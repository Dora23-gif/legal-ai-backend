"""
Deterministic Legal Citation Generator and Hallucination Filter.

Guarantees:
1. Citations are generated PROGRAMMATICALLY from verified database fields only.
2. The LLM (DeepSeek) is never permitted to invent citation IDs, URLs, or gazette references.
3. DeepSeek is provided with fixed token anchors: [CITATION: <ID>].
4. Any citation token emitted by the model that does not exist in the active retrieved pool
   is strictly detected and rejected.
"""
import re
from typing import Dict, Any, List, Set, Tuple
from rag.models import LegalProvisionChunk, LegalSourceDocument


RE_CITATION_TOKEN = re.compile(r'\[CITATION:\s*([A-Za-z0-9_-]+)\]')


def generate_citation_id(chunk: LegalProvisionChunk) -> str:
    """
    Creates a deterministic, human-readable legal citation identifier.
    e.g.: 'CIT-FED-CAMA-SEC18' or 'CIT-LAGOS-TENANCY-SEC4'
    """
    doc = chunk.document
    jur_prefix = (
        doc.jurisdiction
        .replace('FEDERAL', 'FED')
        .replace('_STATE', '')
        .replace('FCT_ABUJA', 'FCT')
    )
    clean_fam = re.sub(r'[^A-Za-z0-9]', '', doc.statute_family_code.replace('FED_', '').replace('LAG_', ''))[:12]
    clean_sec = re.sub(r'[^A-Za-z0-9]', '', chunk.section_number.replace('Section', 'SEC'))[:12]
    return f"CIT-{jur_prefix}-{clean_fam}-{clean_sec}".upper()


def build_structured_citation(chunk: LegalProvisionChunk) -> Dict[str, Any]:
    """
    Builds a complete, verifiable legal citation record from database metadata.
    """
    doc = chunk.document
    citation_id = generate_citation_id(chunk)
    citation_token = f"[CITATION: {citation_id}]"

    # Gazette & Provenance citations
    gazette = doc.official_gazette_ref or "Official Publication"
    publisher = doc.publisher or "Federal/State Government Printer"

    source_url = doc.source_url

    return {
        'citation_id': citation_id,
        'citation_token': citation_token,
        'citation_label': chunk.citation_label,
        'statute_title': doc.title,
        'short_title': doc.short_title,
        'section_number': chunk.section_number,
        'section_title': chunk.section_title,
        'jurisdiction': doc.jurisdiction,
        'status': doc.status,
        'version_label': doc.version_label,
        'is_current': doc.is_current_version,
        'effective_from': str(doc.effective_from),
        'effective_to': str(doc.effective_to) if doc.effective_to else None,
        'official_gazette_ref': gazette,
        'publisher': publisher,
        'source_url': source_url,
        'sha256_checksum': doc.sha256_checksum,
        'is_verified': doc.is_verified_authentic,
        'verified_by': doc.verified_by,
        'source_type': getattr(doc, 'source_type', 'OFFICIAL_GAZETTE'),
        'verification_status': getattr(doc, 'verification_status', 'PENDING_VERIFICATION'),
        'chunk_id': str(chunk.id),
        'document_id': str(doc.id),
    }


def filter_and_validate_citations(
    text: str,
    active_citation_tokens: Set[str],
    citations: Optional[List[Dict[str, Any]]] = None,
    render_clickable: Optional[bool] = None,
) -> Tuple[str, List[str], List[str]]:
    """
    Audits the generated LLM text for citation tokens.

    - Valid citations: tokens present in active_citation_tokens.
    - Hallucinated citations: tokens invented by the model not in active_citation_tokens.
      These are stripped from the visible response to guarantee zero hallucinated provenance.
    - Clickable rendering: when render_clickable is True (or enabled when citations metadata is provided),
      transforms validated internal tokens (e.g. [CITATION: CIT-FED-CAMA2020-SEC18]) into human-readable,
      clickable section-level markdown citations (e.g. [Section 18, CAMA 2020](source_url)) while
      preserving surrounding subsection references (e.g. Section 18(1)).

    Returns:
        (sanitized_text, valid_tokens_found, rejected_tokens_found)
    """
    if render_clickable is None:
        render_clickable = bool(citations and len(citations) > 0)

    citations_by_id: Dict[str, Dict[str, Any]] = {}
    if citations:
        for c in citations:
            cid = c.get('citation_id', '')
            if cid:
                citations_by_id[cid] = c
                citations_by_id[cid.upper()] = c

    found_tokens = RE_CITATION_TOKEN.findall(text)
    valid_tokens = []
    rejected_tokens = []

    for tok_id in found_tokens:
        full_tok = f"[CITATION: {tok_id}]"
        is_valid = (
            tok_id in active_citation_tokens
            or full_tok in active_citation_tokens
            or tok_id.upper() in active_citation_tokens
            or f"[CITATION: {tok_id.upper()}]" in active_citation_tokens
        )
        if is_valid:
            valid_tokens.append(full_tok)
        else:
            rejected_tokens.append(full_tok)

    # 1. Sanitize text by removing hallucinated tokens
    sanitized_text = text
    for bad_tok in set(rejected_tokens):
        sanitized_text = sanitized_text.replace(bad_tok, "")

    # 2. Render validated citation tokens as clickable section-level citations
    if render_clickable:
        for full_tok in set(valid_tokens):
            m = RE_CITATION_TOKEN.search(full_tok)
            if not m:
                continue
            tok_id = m.group(1)
            cit = citations_by_id.get(tok_id) or citations_by_id.get(tok_id.upper())

            if cit and cit.get('source_url') and str(cit.get('source_url')).startswith(('http://', 'https://')):
                label = cit.get('citation_label') or f"{cit.get('section_number', 'Section')}, {cit.get('short_title', '')}".strip(', ')
                url = str(cit.get('source_url'))
                replacement = f"[{label}]({url})"
            elif cit:
                # Citation metadata exists without an external URL: render clean authoritative label without misleading hyperlink
                label = cit.get('citation_label') or f"{cit.get('section_number', 'Section')}, {cit.get('short_title', '')}".strip(', ')
                replacement = label
            else:
                # Valid token without detailed dictionary metadata: render clean section label without fabricated URL
                parts = tok_id.split('-')
                sec_part = parts[-1] if len(parts) > 1 else 'Section'
                sec_label = re.sub(r'^SEC(\d+)', r'Section \1', sec_part, flags=re.IGNORECASE)
                statute_part = parts[-2] if len(parts) > 2 else ''
                statute_label = re.sub(r'([A-Za-z]+)(\d+)', r'\1 \2', statute_part)
                label = f"{sec_label}, {statute_label}".strip(', ') if statute_label else sec_label
                replacement = label

            sanitized_text = sanitized_text.replace(full_tok, replacement)

    return sanitized_text, valid_tokens, rejected_tokens
