"""
Unified Nigerian Legal Retrieval and Context Generation Service.

Orchestrates:
1. Jurisdiction Detection (State vs Federal hierarchy)
2. Temporal Intent Analysis (Current in-force vs Historical effective date)
3. Hybrid Retrieval with Reciprocal Rank Fusion (RRF k=60)
4. Deterministic Citation Generation & Amendment Lineage Notice
5. Prompt Context Assembly for DeepSeek Grounded Ingestion
"""
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from rag.models import LegalProvisionChunk
from rag.services.jurisdiction_service import detect_jurisdiction
from rag.services.temporal_router import analyze_temporal_intent, format_statutory_lineage_notice
from rag.services.hybrid_retriever import hybrid_legal_search
from rag.services.citation_service import build_structured_citation


RE_LEGAL_INTENT = re.compile(
    r'\b(?:law|act|statute|section|decree|provision|court|judge|legal|jurisdiction|'
    r'cama|evidence|constitution|tax|taxation|landlord|tenant|tenancy|premises|'
    r'incorporation|company|director|offence|penalty|fine|amendment|gazette|'
    r'nigerian law|laws of the federation|lfn)\b',
    re.IGNORECASE
)


def is_legal_query(query: str) -> bool:
    """Detects whether a user prompt concerns Nigerian law or statutory provisions."""
    return bool(RE_LEGAL_INTENT.search(query))


def retrieve_legal_context(
    query: str,
    explicit_jurisdiction: Optional[str] = None,
    limit: int = 4,
    allow_unverified: bool = False,
) -> Dict[str, Any]:
    """
    High-level entrypoint for legal context retrieval.

    Supports:
    1. Verified Authentic Retrieval (Default): Searches exclusively documents verified by legal counsel.
    2. Controlled Unverified Retrieval (allow_unverified=True): When no verified provisions exist,
       retrieves from quarantined official agency reprints with prominent disclaimers and section-level citations.

    Returns:
        {
            'is_legal_query': bool,
            'has_verified_results': bool,
            'has_unverified_results': bool,
            'is_unverified_mode': bool,
            'prompt_context': str,
            'citations': List[Dict[str, Any]],
            'active_citation_tokens': Set[str],
            'jurisdictions': List[str],
            'is_historical': bool,
            'target_date': Optional[str],
        }
    """
    query_is_legal = is_legal_query(query)

    if not query_is_legal:
        return {
            'is_legal_query': False,
            'has_verified_results': False,
            'has_unverified_results': False,
            'is_unverified_mode': False,
            'prompt_context': "",
            'citations': [],
            'active_citation_tokens': set(),
            'jurisdictions': [],
            'is_historical': False,
            'target_date': None,
        }

    # 1. Jurisdiction Detection
    if explicit_jurisdiction:
        primary_jur = explicit_jurisdiction
        search_jurs = [explicit_jurisdiction]
    else:
        primary_jur, search_jurs = detect_jurisdiction(query)

    # 2. Temporal Analysis (Current vs Historical)
    is_hist, target_date, target_year = analyze_temporal_intent(query)

    # 3. Hybrid Search via RRF (Verified Authentic Only First)
    ranked_chunks = hybrid_legal_search(
        query=query,
        jurisdictions=search_jurs,
        is_historical=is_hist,
        target_date=target_date,
        limit=limit,
        include_unverified=False,
    )

    if not ranked_chunks:
        # Check if controlled unverified retrieval is enabled
        if allow_unverified:
            unverified_chunks = hybrid_legal_search(
                query=query,
                jurisdictions=search_jurs,
                is_historical=is_hist,
                target_date=target_date,
                limit=limit,
                unverified_only=True,
            )
            if unverified_chunks:
                citations: List[Dict[str, Any]] = []
                active_tokens: Set[str] = set()
                context_blocks: List[str] = []

                exact_unverified_notice = (
                    "This is an official-agency-hosted reprint that has not yet been "
                    "independently authenticated against the original Official Gazette or reviewed by legal counsel."
                )

                for rank, (chunk, score) in enumerate(unverified_chunks, start=1):
                    cit_data = build_structured_citation(chunk)
                    cit_data['unverified_disclaimer'] = exact_unverified_notice
                    citations.append(cit_data)
                    token = cit_data['citation_token']
                    active_tokens.add(cit_data['citation_id'])
                    active_tokens.add(token)

                    source_label = cit_data.get('source_type', 'OFFICIAL_AGENCY_HOSTED_REPRINT').replace('_', ' ').title()

                    block = (
                        f"--- UNVERIFIED PROVISION {rank} {token} ---\n"
                        f"Statute: {chunk.document.title} ({chunk.document.short_title}) [UNVERIFIED AGENCY REPRINT]\n"
                        f"Citation: {chunk.citation_label}\n"
                        f"Jurisdiction: {chunk.document.jurisdiction}\n"
                        f"Verification Status: PENDING_VERIFICATION (is_verified_authentic=False)\n"
                        f"Source Classification: {source_label}\n"
                        f"Publisher / Imprint: {cit_data['publisher']}\n"
                        f"Official Gazette Reference: {cit_data['official_gazette_ref']}\n\n"
                        f"STATUTORY TEXT:\n{chunk.content}\n"
                    )
                    context_blocks.append(block)

                unverified_prompt_context = (
                    "====================================================================\n"
                    "CONTROLLED UNVERIFIED STATUTORY REFERENCE (PENDING COUNSEL AUDIT)\n"
                    "====================================================================\n"
                    "IMPORTANT NOTICE: The following provisions were retrieved from an\n"
                    f"UNVERIFIED OFFICIAL-AGENCY-HOSTED REPRINT. {exact_unverified_notice}\n"
                    "This source is currently held in quarantine (PENDING_VERIFICATION).\n\n"
                    "MANDATORY INSTRUCTIONS FOR YOUR ANSWER:\n"
                    "1. UNVERIFIED-SOURCE DISCLOSURE AT BEGINNING:\n"
                    "   You MUST begin your response with this exact disclosure verbatim at the very top:\n"
                    f"   > **Notice**: {exact_unverified_notice}\n\n"
                    "2. NO REPETITIVE DISCLAIMERS AT THE END:\n"
                    "   Place the disclosure ONCE at the beginning. Do NOT add repetitive generic disclaimers,\n"
                    "   boilerplate caveats, or duplicate notices at the end of your answer unless genuinely necessary.\n\n"
                    "3. ACCURATE SUBSECTION REFERENCES & GROUNDING:\n"
                    "   Ground your explanation strictly in the statutory text provided below.\n"
                    "   Preserve accurate subsection references (e.g. Section 18(1), Section 18(2)).\n\n"
                    "4. STATUTORY CITATIONS:\n"
                    "   You MUST cite each provision using its exact token, e.g. [CITATION: CIT-...].\n"
                    "   These tokens will be rendered as clickable section-level citations.\n\n"
                    "5. NON-AUTHORITATIVE STATUS:\n"
                    "   NEVER represent these provisions or your answer as counsel-verified, authenticated,\n"
                    "   or legally authoritative.\n"
                    "====================================================================\n\n"
                    + "\n\n".join(context_blocks)
                    + "\n====================================================================\n"
                )

                return {
                    'is_legal_query': True,
                    'has_verified_results': False,
                    'has_unverified_results': True,
                    'is_unverified_mode': True,
                    'prompt_context': unverified_prompt_context,
                    'citations': citations,
                    'active_citation_tokens': active_tokens,
                    'jurisdictions': search_jurs,
                    'is_historical': is_hist,
                    'target_date': str(target_date) if target_date else None,
                }

        return {
            'is_legal_query': True,
            'has_verified_results': False,
            'has_unverified_results': False,
            'is_unverified_mode': False,
            'prompt_context': "",
            'citations': [],
            'active_citation_tokens': set(),
            'jurisdictions': search_jurs,
            'is_historical': is_hist,
            'target_date': str(target_date) if target_date else None,
        }

    # 4. Build Structured Citations and Prompt Text (VERIFIED AUTHENTIC)
    citations: List[Dict[str, Any]] = []
    active_tokens: Set[str] = set()
    context_blocks: List[str] = []

    for rank, (chunk, score) in enumerate(ranked_chunks, start=1):
        cit_data = build_structured_citation(chunk)
        citations.append(cit_data)
        token = cit_data['citation_token']
        active_tokens.add(cit_data['citation_id'])
        active_tokens.add(token)

        # Lineage and amendment notice
        lineage = format_statutory_lineage_notice(chunk)
        lineage_line = f"\n  {lineage['lineage_text']}" if lineage['lineage_text'] else ""

        block = (
            f"--- PROVISION {rank} {token} ---\n"
            f"Statute: {chunk.document.title} ({chunk.document.short_title})\n"
            f"Citation: {chunk.citation_label}\n"
            f"Jurisdiction: {chunk.document.jurisdiction}\n"
            f"Version / Status: {chunk.document.version_label} | {chunk.document.status}\n"
            f"Effective From: {chunk.document.effective_from} (To: {chunk.document.effective_to or 'Present'})\n"
            f"Gazette Provenance: {cit_data['official_gazette_ref']} | {cit_data['publisher']}{lineage_line}\n\n"
            f"STATUTORY TEXT:\n{chunk.content}\n"
        )
        context_blocks.append(block)

    prompt_context = (
        "====================================================================\n"
        "OFFICIAL NIGERIAN STATUTORY KNOWLEDGE BASE (AUTHENTIC & VERIFIED)\n"
        "====================================================================\n"
        "The following statutory provisions were retrieved from the verified Nigerian legal knowledge base.\n"
        "RULES FOR LEGAL GROUNDING:\n"
        "1. Base your answer strictly on the legal text provided below.\n"
        "2. You MUST cite each legal provision using its exact token, e.g. [CITATION: CIT-...].\n"
        "3. Do NOT invent or alter section numbers, statute titles, or gazette references.\n"
        "4. Note any amendment or historical repeal notices and advise the user accordingly.\n"
        "====================================================================\n\n"
        + "\n\n".join(context_blocks)
        + "\n====================================================================\n"
    )

    return {
        'is_legal_query': True,
        'has_verified_results': True,
        'has_unverified_results': False,
        'is_unverified_mode': False,
        'prompt_context': prompt_context,
        'citations': citations,
        'active_citation_tokens': active_tokens,
        'jurisdictions': search_jurs,
        'is_historical': is_hist,
        'target_date': str(target_date) if target_date else None,
    }


def retrieve_unverified_legal_context(
    query: str,
    explicit_jurisdiction: Optional[str] = None,
    limit: int = 4,
) -> Dict[str, Any]:
    """
    Explicit entrypoint for controlled unverified statutory retrieval.
    """
    return retrieve_legal_context(
        query=query,
        explicit_jurisdiction=explicit_jurisdiction,
        limit=limit,
        allow_unverified=True,
    )


def get_trusted_legal_chunks(
    query_text: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    target_date: Optional[Any] = None,
    limit: int = 10,
):
    """
    Backwards-compatible wrapper returning QuerySet of trusted chunks.
    """
    from rag.services.hybrid_retriever import hybrid_legal_search
    if not query_text:
        return LegalProvisionChunk.objects.filter(
            document__is_verified_authentic=True,
            document__verification_status='VERIFIED_AUTHENTIC',
        )[:limit]

    results = hybrid_legal_search(
        query=query_text,
        jurisdictions=[jurisdiction] if jurisdiction else None,
        is_historical=target_date is not None,
        target_date=target_date,
        limit=limit,
    )
    chunk_ids = [c[0].id for c in results]
    return LegalProvisionChunk.objects.filter(id__in=chunk_ids)
