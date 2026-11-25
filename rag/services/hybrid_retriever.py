"""
Hybrid Legal Retrieval Engine with Reciprocal Rank Fusion (RRF).

Combines:
1. Dense Vector Search: FastEmbed BGE-small-en-v1.5 + pgvector Cosine Distance (<=>).
2. Sparse Full-Text Search: PostgreSQL search_vector + websearch_to_tsquery.
3. Exact Statutory Matching: Section number and statute title regex boosting.
4. Reciprocal Rank Fusion (RRF, k=60): Rank-based fusion without score calibration.
5. Strict Quarantine: Excludes any unverified legal documents.
6. Temporal and Jurisdiction Awareness: Prioritizes applicable state and historical versions.
"""
import re
from typing import List, Tuple, Dict, Any, Optional
from datetime import date
from django.db import connection
from django.db.models import Q, F

from rag.models import LegalProvisionChunk, VerificationStatus
from rag.services.vectorizer import embed_query
from rag.services.temporal_router import apply_temporal_filter


RRF_K = 60
EXACT_MATCH_WEIGHT = 1.5
VECTOR_MATCH_WEIGHT = 1.0
FTS_MATCH_WEIGHT = 1.0

RE_SECTION_QUERY = re.compile(r'\b(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)\b', re.IGNORECASE)


def extract_section_query(query: str) -> Optional[str]:
    """Extracts explicit section numbers like 'Section 84A' from query."""
    m = RE_SECTION_QUERY.search(query)
    if m:
        return f"Section {m.group(1).upper()}"
    return None


def hybrid_legal_search(
    query: str,
    jurisdictions: Optional[List[str]] = None,
    is_historical: bool = False,
    target_date: Optional[date] = None,
    limit: int = 5,
    include_unverified: bool = False,
    unverified_only: bool = False,
) -> List[Tuple[LegalProvisionChunk, float]]:
    """
    Executes hybrid retrieval combining vector similarity, full-text search,
    and exact statutory matching fused via Reciprocal Rank Fusion (k=60).

    Returns:
        List of (LegalProvisionChunk, rrf_score) sorted by descending score.
    """
    if not query or not query.strip():
        return []

    # 1. Base QuerySet with Strict Quarantine / Unverified Mode Separation
    if unverified_only:
        base_qs = LegalProvisionChunk.objects.select_related(
            'document', 'amendment_record', 'amendment_record__amending_act'
        ).filter(
            document__is_verified_authentic=False,
            document__verification_status=VerificationStatus.PENDING,
        )
    elif include_unverified:
        base_qs = LegalProvisionChunk.objects.select_related(
            'document', 'amendment_record', 'amendment_record__amending_act'
        ).filter(
            document__verification_status__in=[
                VerificationStatus.VERIFIED,
                VerificationStatus.PENDING,
            ]
        )
    else:
        # Default: Verified Authentic Only
        base_qs = LegalProvisionChunk.objects.select_related(
            'document', 'amendment_record', 'amendment_record__amending_act'
        ).filter(
            document__is_verified_authentic=True,
            document__verification_status=VerificationStatus.VERIFIED,
        )

    # 2. Jurisdiction Filter
    primary_jurisdiction = jurisdictions[0] if jurisdictions else None
    if jurisdictions:
        base_qs = base_qs.filter(document__jurisdiction__in=jurisdictions)

    # 3. Temporal Filter (Current vs Historical)
    base_qs = apply_temporal_filter(base_qs, is_historical=is_historical, target_date=target_date)

    # Candidate tracking dict: chunk_id -> (chunk_obj, rrf_score)
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, LegalProvisionChunk] = {}

    # -------------------------------------------------------------
    # Pipeline A: Exact Section & Title Match (High Precision Tier)
    # -------------------------------------------------------------
    explicit_sec = extract_section_query(query)
    if explicit_sec:
        exact_qs = base_qs.filter(
            Q(section_number__iexact=explicit_sec) |
            Q(section_number__icontains=explicit_sec)
        )[:15]

        for rank, chunk in enumerate(exact_qs, start=1):
            cid = str(chunk.id)
            chunk_map[cid] = chunk
            score = EXACT_MATCH_WEIGHT / (RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + score

    # -------------------------------------------------------------
    # Pipeline B: Dense Vector Similarity Search (FastEmbed + pgvector)
    # -------------------------------------------------------------
    try:
        query_vec = embed_query(query)
        if connection.vendor == 'postgresql':
            from pgvector.django import CosineDistance
            # Strict semantic distance threshold (cosine_dist <= 0.35)
            vec_qs = list(
                base_qs.annotate(
                    cosine_dist=CosineDistance('embedding', query_vec)
                ).filter(cosine_dist__lte=0.35).order_by('cosine_dist')[:25]
            )
        else:
            # Fallback for SQLite unit tests (where pgvector is unavailable):
            # Rank chunks by query term overlap count
            stop_words = {
                'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'this', 'that',
                'these', 'those', 'under', 'from', 'with', 'have', 'does', 'were', 'been',
                'about', 'into', 'over', 'after', 'than', 'more', 'some', 'such', 'many',
                'most', 'other', 'only', 'shall', 'should', 'would', 'could', 'cannot',
                'the', 'for', 'and', 'are', 'was', 'per', 'how', 'why', 'can', 'may',
                'its', 'any', 'all', 'not', 'but', 'nor'
            }
            meaningful_words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2 and w not in stop_words]

            scored_candidates = []
            if meaningful_words:
                min_overlap = 2 if len(meaningful_words) >= 2 else 1
                for candidate in base_qs:
                    searchable_text = f"{candidate.content} {candidate.section_title} {candidate.document.short_title} {candidate.document.title}".lower()
                    candidate_tokens = set(re.findall(r'\w+', searchable_text))
                    overlap = sum(1 for w in meaningful_words if w in candidate_tokens)
                    if overlap >= min_overlap:
                        scored_candidates.append((candidate, overlap))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                vec_qs = [c[0] for c in scored_candidates[:25]]
            else:
                vec_qs = []

        for rank, chunk in enumerate(vec_qs, start=1):
            cid = str(chunk.id)
            chunk_map[cid] = chunk
            score = VECTOR_MATCH_WEIGHT / (RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + score
    except Exception as e:
        # Vector search fallback if vector service fails
        pass

    # -------------------------------------------------------------
    # Pipeline C: PostgreSQL Full-Text Search (tsvector & GIN Index)
    # -------------------------------------------------------------
    if connection.vendor == 'postgresql':
        try:
            # Clean search query of boolean syntax hazards
            clean_q = re.sub(r'[^\w\s]', ' ', query).strip()
            if clean_q:
                # Use raw SQL / extra to query GIN index with websearch_to_tsquery
                fts_qs = base_qs.extra(
                    where=["rag_legal_provision_chunk.search_vector @@ websearch_to_tsquery('english', %s)"],
                    params=[clean_q],
                    select={'rank': "ts_rank_cd(rag_legal_provision_chunk.search_vector, websearch_to_tsquery('english', %s))"},
                    select_params=[clean_q],
                ).order_by('-rank')[:25]

                for rank, chunk in enumerate(fts_qs, start=1):
                    cid = str(chunk.id)
                    chunk_map[cid] = chunk
                    score = FTS_MATCH_WEIGHT / (RRF_K + rank)
                    scores[cid] = scores.get(cid, 0.0) + score
        except Exception:
            # Fallback if FTS parsing encounters syntax issue
            fts_fallback = base_qs.filter(
                Q(content__icontains=query) | Q(section_title__icontains=query)
            )[:20]
            for rank, chunk in enumerate(fts_fallback, start=1):
                cid = str(chunk.id)
                chunk_map[cid] = chunk
                score = FTS_MATCH_WEIGHT / (RRF_K + rank)
                scores[cid] = scores.get(cid, 0.0) + score
    else:
        # SQLite unit tests FTS fallback
        fts_qs = base_qs.filter(
            Q(content__icontains=query) | Q(section_title__icontains=query)
        )[:20]
        for rank, chunk in enumerate(fts_qs, start=1):
            cid = str(chunk.id)
            chunk_map[cid] = chunk
            score = FTS_MATCH_WEIGHT / (RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + score

    # -------------------------------------------------------------
    # Jurisdiction Priority Re-ranking
    # -------------------------------------------------------------
    if primary_jurisdiction:
        for cid, chunk in chunk_map.items():
            if chunk.document.jurisdiction == primary_jurisdiction:
                # Slight priority boost to primary state jurisdiction over general federal law
                scores[cid] = scores.get(cid, 0.0) + 0.005

    # Sort descending by fused RRF score
    ranked_results = [
        (chunk_map[cid], scores[cid])
        for cid in sorted(scores, key=lambda k: scores[k], reverse=True)
    ]

    return ranked_results[:limit]
