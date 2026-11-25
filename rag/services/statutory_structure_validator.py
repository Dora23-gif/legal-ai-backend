"""
Statutory Structure & Completeness Validator (Phase 4).

Validates that an extracted LegalSourceDocument exhibits 100% structural fidelity
to its statutory enactment model prior to legal counsel verification:
1. Exact sequential sections (e.g. Sections 1 through 168 for FCCPA 2018).
2. Exact statutory parts (e.g. Parts I through XVIII for FCCPA 2018), verified
   exclusively from structural metadata and headings, never from inline prose.
3. Independent schedule verification (First Schedule and Second Schedule), verified
   exclusively from structural metadata and headings, never from inline prose.
4. Distinguishes legitimate section chunking (multi-chunk sections and subsections)
   from genuine duplicate provisions and duplicate section content.
5. Detection and reporting of missing, duplicate, malformed, negative, and out-of-bounds sections.
6. Strict architectural separation of structural completeness auditing from
   legal counsel authenticity verification.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import hashlib
import re

from rag.models import LegalSourceDocument, LegalProvisionChunk
from rag.source_register import get_statute_registry_entry


ROMAN_TO_INT: Dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
    "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
    "XIX": 19, "XX": 20, "XXI": 21, "XXII": 22
}
INT_TO_ROMAN: Dict[int, str] = {v: k for k, v in ROMAN_TO_INT.items()}

# Ordinal Schedule Canonical Mapping (First Schedule through Fifteenth Schedule)
ORDINAL_SCHEDULE_MAP = {
    "first": "First Schedule", "1st": "First Schedule",
    "second": "Second Schedule", "2nd": "Second Schedule",
    "third": "Third Schedule", "3rd": "Third Schedule",
    "fourth": "Fourth Schedule", "4th": "Fourth Schedule",
    "fifth": "Fifth Schedule", "5th": "Fifth Schedule",
    "sixth": "Sixth Schedule", "6th": "Sixth Schedule",
    "seventh": "Seventh Schedule", "7th": "Seventh Schedule",
    "eighth": "Eighth Schedule", "8th": "Eighth Schedule",
    "ninth": "Ninth Schedule", "9th": "Ninth Schedule",
    "tenth": "Tenth Schedule", "10th": "Tenth Schedule",
    "eleventh": "Eleventh Schedule", "11th": "Eleventh Schedule",
    "twelfth": "Twelfth Schedule", "12th": "Twelfth Schedule",
    "thirteenth": "Thirteenth Schedule", "13th": "Thirteenth Schedule",
    "fourteenth": "Fourteenth Schedule", "14th": "Fourteenth Schedule",
    "fifteenth": "Fifteenth Schedule", "15th": "Fifteenth Schedule",
}
RE_ORDINAL_SCHEDULE_PATTERN = (
    r'\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|'
    r'seventh|7th|eighth|8th|ninth|9th|tenth|10th|eleventh|11th|'
    r'twelfth|12th|thirteenth|13th|fourteenth|14th|fifteenth|15th)\s+schedule\b'
)


@dataclass
class StructuralCompletenessReport:
    """Detailed diagnostic report of statutory structural completeness."""
    statute_family_code: str
    document_id: str
    is_structurally_complete: bool
    total_chunks: int

    # Sections
    expected_sections_count: int
    found_sections: List[str]
    missing_sections: List[str]
    duplicate_sections: List[str]
    malformed_sections: List[str]

    # Parts
    expected_parts: List[str]
    found_parts: List[str]
    missing_parts: List[str]

    # Schedules
    expected_schedules: List[str]
    found_schedules: List[str]
    missing_schedules: List[str]

    # Diagnostic errors
    errors: List[str] = field(default_factory=list)

    def format_detailed_report(self) -> str:
        """Renders human-readable audit diagnostics for CLI and logs."""
        lines = [
            "=" * 60,
            "STATUTORY STRUCTURAL COMPLETENESS AUDIT REPORT",
            "=" * 60,
            f"Statute Family Code:   {self.statute_family_code}",
            f"Document UUID:         {self.document_id}",
            f"Total Chunks Audited:  {self.total_chunks}",
            f"Audit Outcome:         {'PASS - 100% COMPLETE' if self.is_structurally_complete else 'FAIL - STRUCTURALLY INCOMPLETE'}",
            "",
            f"[1. SECTIONS AUDIT] (Target: {self.expected_sections_count} sequential sections)",
            f"  Distinct Sections Found: {len(self.found_sections)}",
            f"  Missing Sections ({len(self.missing_sections)}): {self.missing_sections if len(self.missing_sections) <= 15 else self.missing_sections[:15] + ['... (total ' + str(len(self.missing_sections)) + ' missing)']}",
            f"  Duplicate Sections:      {self.duplicate_sections or 'None'}",
            f"  Malformed / Out-of-Bounds: {self.malformed_sections or 'None'}",
            "",
            f"[2. PARTS AUDIT] (Target: {len(self.expected_parts)} parts: {', '.join(self.expected_parts) if self.expected_parts else 'None'})",
            f"  Parts Found:             {', '.join(self.found_parts) if self.found_parts else 'None'}",
            f"  Missing Parts ({len(self.missing_parts)}): {', '.join(self.missing_parts) if self.missing_parts else 'None'}",
            "",
            f"[3. SCHEDULES AUDIT] (Target: {', '.join(self.expected_schedules) if self.expected_schedules else 'None'})",
            f"  Schedules Found:         {', '.join(self.found_schedules) if self.found_schedules else 'None'}",
            f"  Missing Schedules:       {', '.join(self.missing_schedules) if self.missing_schedules else 'None'}",
        ]
        if self.errors:
            lines.append("")
            lines.append("[DIAGNOSTIC ERRORS]")
            for err in self.errors:
                lines.append(f"  * {err}")
        lines.append("=" * 60)
        return "\n".join(lines)


def canonicalize_part_token(token: str) -> Optional[str]:
    """Converts a token (Letter A-G, Roman or digit) into canonical 'Part {X}' format."""
    tok = token.strip().upper()
    if tok in ("A", "B", "C", "D", "E", "F", "G"):
        return f"Part {tok}"
    if tok.isdigit():
        n = int(tok)
        return f"Part {INT_TO_ROMAN.get(n, tok)}"
    elif tok in ROMAN_TO_INT:
        return f"Part {tok}"
    return None


def extract_structural_part_identifier(chunk: LegalProvisionChunk) -> Optional[str]:
    """
    Extracts a canonical Part identifier (e.g. 'Part V' or 'Part A') ONLY from reliable
    structural metadata or structural headings.

    Hardened Guard: Does NOT match arbitrary inline statutory prose like
    'Subject to Part V' or 'referred to in Part A'.
    """
    # 1. Dedicated structural metadata field on the model
    if chunk.part_number:
        m = re.search(r'\bPart\s+([A-G]|[IVXLCDM]+|\d+)\b', chunk.part_number, re.IGNORECASE)
        if m:
            return canonicalize_part_token(m.group(1))

    # 2. Structural section_number (e.g. chunk represents the Part divider)
    if chunk.section_number:
        m = re.match(r'^\s*Part\s+([A-G]|[IVXLCDM]+|\d+)(?:\s*[\.:\-\—]|$)', chunk.section_number, re.IGNORECASE)
        if m:
            return canonicalize_part_token(m.group(1))

    # 3. Structural section_title (heading beginning with Part)
    if chunk.section_title:
        m = re.match(r'^\s*Part\s+([A-G]|[IVXLCDM]+|\d+)\b(?:\s*[\:\-\—\.\)]|\s+[A-Z])', chunk.section_title, re.IGNORECASE)
        if m:
            return canonicalize_part_token(m.group(1))

    # 4. Standalone structural heading at the VERY START of chunk content:
    # Must be on its own line or followed by an uppercase title line
    if chunk.content:
        m = re.match(r'^\s*PART\s+([A-G]|[IVXLCDM]+|\d+)\b(?:\s*[\:\-\—\.\n]|\s+[A-Z\s]{3,}\n)', chunk.content, re.IGNORECASE)
        if m:
            return canonicalize_part_token(m.group(1))

    return None


def extract_structural_schedule_identifier(chunk: LegalProvisionChunk) -> Optional[str]:
    """
    Extracts a canonical Schedule identifier (e.g. 'First Schedule' through 'Fifteenth Schedule')
    ONLY from reliable structural metadata or structural headings.

    Hardened Guard: Does NOT match arbitrary inline statutory prose like
    'as specified in the First Schedule'.
    """
    # 1. Structural section_number
    if chunk.section_number:
        m = re.search(RE_ORDINAL_SCHEDULE_PATTERN, chunk.section_number, re.IGNORECASE)
        if m:
            return ORDINAL_SCHEDULE_MAP.get(m.group(1).lower())

    # 2. Structural section_title heading
    if chunk.section_title:
        m = re.search(RE_ORDINAL_SCHEDULE_PATTERN, chunk.section_title, re.IGNORECASE)
        if m:
            return ORDINAL_SCHEDULE_MAP.get(m.group(1).lower())

    # 3. Citation label
    if chunk.citation_label:
        m = re.search(RE_ORDINAL_SCHEDULE_PATTERN, chunk.citation_label, re.IGNORECASE)
        if m:
            return ORDINAL_SCHEDULE_MAP.get(m.group(1).lower())

    # 4. Standalone structural heading at start of content
    if chunk.content:
        m = re.match(
            r'^\s*(?:THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH)\s+SCHEDULE\b',
            chunk.content,
            re.IGNORECASE
        )
        if m:
            return ORDINAL_SCHEDULE_MAP.get(m.group(1).lower())

    return None


def parse_section_identifiers(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses a raw provision string into:
      (parent_section, precise_provision_id, error_message)

    Distinguishes:
      - Parent section: e.g. "Section 15"
      - Precise provision: e.g. "Section 15(1)", "Section 15(2)", "Section 15"
      - Error: malformed, negative, or unrecognized patterns

    Examples:
      "Section 1"       -> ("Section 1", "Section 1", None)
      "Section 15(1)"   -> ("Section 15", "Section 15(1)", None)
      "Section 15(2)"   -> ("Section 15", "Section 15(2)", None)
      "s. 15"           -> ("Section 15", "Section 15", None)
      "15"              -> ("Section 15", "Section 15", None)
      "First Schedule"  -> (None, "First Schedule", None)
      "Second Schedule" -> (None, "Second Schedule", None)
      "Section -5"      -> (None, None, "Negative section number: Section -5")
      "Section ???"     -> (None, None, "Malformed section identifier: Section ???")
      "Section Invalid" -> (None, None, "Malformed section identifier: Section Invalid")
    """
    if not raw_text:
        return None, None, "Empty section text"
    raw = str(raw_text).strip()

    # Check schedules
    sched_m = re.search(r'^\s*(?:the\s+)?' + RE_ORDINAL_SCHEDULE_PATTERN, raw, re.IGNORECASE)
    if sched_m:
        return None, ORDINAL_SCHEDULE_MAP.get(sched_m.group(1).lower()), None
    if re.search(r'\bschedule\b', raw, re.IGNORECASE):
        return None, raw.strip(), None

    # First attempt exact section match
    m = re.match(
        r'^\s*(?:section\b|sec\b\.?|s\b\.\s*|s\b\s+)?\s*(-?\d+[a-zA-Z]?)(?:\s*(\([^\)]+\)[^\s]*))?(?:\s*[\.:\-\,]|$)',
        raw,
        re.IGNORECASE
    )
    if m:
        num_str = m.group(1).upper()
        subsec = m.group(2) or ''
        if num_str.startswith('-'):
            return None, None, f"Negative section number: {raw}"
        parent = f"Section {num_str}"
        precise = f"Section {num_str}{subsec}" if subsec else parent
        return parent, precise, None

    # If it begins with a section keyword but didn't match the valid numeric pattern, it's malformed
    if re.match(r'^\s*(?:section\b|sec\b|s\.)', raw, re.IGNORECASE):
        return None, None, f"Malformed section identifier: {raw}"

    return None, None, f"Unrecognized provision format: {raw}"


def normalize_section_identifier(text: str) -> Optional[str]:
    """Backward-compatible helper returning parent section or schedule."""
    parent, precise, _ = parse_section_identifiers(text)
    return parent or precise


def compute_content_overlap_metrics(text1: str, text2: str) -> Tuple[float, float]:
    """
    Computes content overlap metrics between two text fragments:
    1. word_overlap: |W1 & W2| / min(|W1|, |W2|) (token vocabulary overlap)
    2. shingle_overlap: |S1 & S2| / min(|S1|, |S2|) (3-gram consecutive phrase overlap)

    Returns (word_overlap_ratio, shingle_overlap_ratio) in range [0.0, 1.0].
    """
    words1 = re.findall(r'\b[a-zA-Z0-9_]+\b', text1.lower())
    words2 = re.findall(r'\b[a-zA-Z0-9_]+\b', text2.lower())

    if not words1 or not words2:
        return 0.0, 0.0

    set1 = set(words1)
    set2 = set(words2)
    min_words = min(len(set1), len(set2))
    word_ratio = len(set1.intersection(set2)) / float(min_words) if min_words > 0 else 0.0

    shingles1 = set(tuple(words1[i:i+3]) for i in range(max(0, len(words1) - 2)))
    shingles2 = set(tuple(words2[i:i+3]) for i in range(max(0, len(words2) - 2)))
    min_shingles = min(len(shingles1), len(shingles2))
    shingle_ratio = len(shingles1.intersection(shingles2)) / float(min_shingles) if min_shingles > 0 else 0.0

    return word_ratio, shingle_ratio


def validate_statutory_completeness(doc: LegalSourceDocument) -> StructuralCompletenessReport:
    """
    Performs rigorous structural completeness auditing on a LegalSourceDocument.

    Validates:
    1. Every sequential parent section from Section 1 to Section N (e.g. 1 to 168 for FCCPA 2018).
    2. Every statutory Part (e.g. Parts I through XVIII for FCCPA 2018) via structural metadata/headings.
    3. Independent Schedule verification (First Schedule and Second Schedule) via structural metadata/headings.
    4. Exact normalized section matching (Section 1 never matches Section 10 or 100).
    5. Distinguishes legitimate section chunking (multi-chunk sections and subsections)
       from genuine duplicate provisions and duplicate section content.
    6. Detection of malformed, negative, or out-of-bounds sections (e.g. Section 0, Section 169).

    NOTE: Structural completeness is strictly independent from legal counsel authorization.
    A complete structural audit does NOT automatically authenticate or verify the document.
    """
    registry_entry = get_statute_registry_entry(doc.statute_family_code)
    gaz_details = registry_entry.get("official_gazette_details", {}) if registry_entry else {}
    completeness_spec = gaz_details.get("extraction_completeness_spec", {})

    expected_sections_count = completeness_spec.get("expected_sections_count", 0)
    expected_parts = completeness_spec.get("expected_parts_list", [])
    expected_schedules = completeness_spec.get("expected_schedules", [])

    # If no completeness specification is defined for this statute family, pass gracefully
    if not expected_sections_count and not expected_parts and not expected_schedules:
        return StructuralCompletenessReport(
            statute_family_code=doc.statute_family_code,
            document_id=str(doc.id),
            is_structurally_complete=True,
            total_chunks=doc.chunks.count(),
            expected_sections_count=0,
            found_sections=[],
            missing_sections=[],
            duplicate_sections=[],
            malformed_sections=[],
            expected_parts=[],
            found_parts=[],
            missing_parts=[],
            expected_schedules=[],
            found_schedules=[],
            missing_schedules=[],
            errors=[],
        )

    chunks = list(doc.chunks.all().order_by('chunk_index'))
    total_chunks = len(chunks)

    # Accounting structures
    parent_sections_found: Set[str] = set()
    found_parts_set: Set[str] = set()
    found_schedules_set: Set[str] = set()
    malformed_sections: List[str] = []

    # Provision grouping for duplicate detection:
    # key: (parent_section, precise_provision_id)
    # value: list of chunk info dicts
    provision_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    parent_section_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in chunks:
        # 1. Structural Schedule Detection (Hardened: ignores inline prose)
        sched_id = extract_structural_schedule_identifier(chunk)
        if sched_id:
            found_schedules_set.add(sched_id)

        # 2. Structural Part Detection (Hardened: ignores inline prose)
        part_id = extract_structural_part_identifier(chunk)
        if part_id:
            found_parts_set.add(part_id)

        # 3. Section and Subsection Parsing
        raw_sec = chunk.section_number or chunk.citation_label or ""
        parent_sec, precise_id, parse_err = parse_section_identifiers(raw_sec)

        if parse_err:
            malformed_sections.append(parse_err)
            continue

        if precise_id and "Schedule" in precise_id:
            # Handled by schedule detector
            continue

        if parent_sec:
            # Check numerical bounds (e.g. Section 0, Section 169)
            m = re.match(r'^Section\s+(\d+)$', parent_sec)
            if m:
                num = int(m.group(1))
                if num < 1:
                    malformed_sections.append(f"{parent_sec} (out-of-bounds: section number must be >= 1)")
                    continue
                if expected_sections_count and num > expected_sections_count:
                    malformed_sections.append(
                        f"{parent_sec} (out-of-bounds: exceeds statute maximum of {expected_sections_count})"
                    )
                    continue

                parent_sections_found.add(parent_sec)

                # Record for duplicate provision and relationship analysis
                raw_content = (chunk.content or "").strip()
                content_fp = hashlib.sha256(raw_content.lower().encode('utf-8')).hexdigest()[:16]
                chunk_record = {
                    "chunk_id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "content": raw_content,
                    "content_fp": content_fp,
                    "precise_id": precise_id,
                    "parent_sec": parent_sec,
                }
                key = (parent_sec, precise_id)
                provision_groups.setdefault(key, []).append(chunk_record)
                parent_section_chunks.setdefault(parent_sec, []).append(chunk_record)

    # Analyze duplicates vs. legitimate chunking
    # 1. Identical duplicate provision content is rejected.
    # 2. Legitimate sequential chunks belonging to one section are accepted.
    # 3. Different subsections are accepted.
    # 4. Partially overlapping chunks with the same provision identifier are detected.
    # 5. Different SHA-256 fingerprints are NOT automatically treated as proof that chunks are legitimate.
    # 6. Chunk relationships and provision identifiers are considered alongside content fingerprints.
    duplicate_sections: List[str] = []

    # 1. Intra-provision group evaluation (chunks sharing the exact same provision identifier)
    for (parent_sec, precise_id), group in provision_groups.items():
        if len(group) > 1:
            fingerprints = [c["content_fp"] for c in group]
            # Case A: Identical content extracted multiple times for the same section/subsection
            if len(set(fingerprints)) < len(group):
                duplicate_sections.append(
                    f"Duplicate provision content detected: {precise_id} ({len(group)} occurrences)"
                )
            # Case B: Specific subsection identifier (e.g. Section 15(1)) duplicated across distinct chunks
            elif precise_id != parent_sec:
                duplicate_sections.append(
                    f"Duplicate subsection identifier: {precise_id} ({len(group)} occurrences)"
                )

            # Case C: Check for partial overlap between distinct chunks sharing the same provision identifier.
            # Distinct SHA-256 fingerprints are NOT automatically proof that two chunks are legitimate!
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    ci = group[i]
                    cj = group[j]
                    if ci["content_fp"] == cj["content_fp"]:
                        continue  # Already handled in Case A
                    w_ratio, s_ratio = compute_content_overlap_metrics(ci["content"], cj["content"])
                    w_len_i = len(re.findall(r'\b[a-zA-Z0-9_]+\b', ci["content"]))
                    w_len_j = len(re.findall(r'\b[a-zA-Z0-9_]+\b', cj["content"]))
                    if s_ratio >= 0.35 or (w_ratio >= 0.50 and w_len_i >= 6 and w_len_j >= 6):
                        overlap_pct = max(w_ratio, s_ratio) * 100
                        msg = (
                            f"Partially overlapping duplicate chunks detected: {precise_id} "
                            f"(chunks #{ci['chunk_index']} and #{cj['chunk_index']} have {overlap_pct:.1f}% content overlap)"
                        )
                        if msg not in duplicate_sections:
                            duplicate_sections.append(msg)

    # 2. Inter-provision / parent section evaluation (cross-subsection duplicate and overlap checks)
    for parent_sec, p_chunks in parent_section_chunks.items():
        if len(p_chunks) > 1:
            for i in range(len(p_chunks)):
                for j in range(i + 1, len(p_chunks)):
                    ci = p_chunks[i]
                    cj = p_chunks[j]
                    if ci["precise_id"] == cj["precise_id"]:
                        continue  # Already evaluated in provision_groups above

                    # Case D: Identical content across different subsections
                    if ci["content_fp"] == cj["content_fp"]:
                        msg = (
                            f"Duplicate content across provisions: {ci['precise_id']} and {cj['precise_id']} "
                            f"(chunks #{ci['chunk_index']} and #{cj['chunk_index']} have identical text)"
                        )
                        if msg not in duplicate_sections:
                            duplicate_sections.append(msg)
                    else:
                        w_ratio, s_ratio = compute_content_overlap_metrics(ci["content"], cj["content"])
                        w_len_i = len(re.findall(r'\b[a-zA-Z0-9_]+\b', ci["content"]))
                        w_len_j = len(re.findall(r'\b[a-zA-Z0-9_]+\b', cj["content"]))
                        if s_ratio >= 0.35 or (w_ratio >= 0.50 and w_len_i >= 6 and w_len_j >= 6):
                            overlap_pct = max(w_ratio, s_ratio) * 100
                            msg = (
                                f"Partially overlapping content across provisions: {ci['precise_id']} and {cj['precise_id']} "
                                f"(chunks #{ci['chunk_index']} and #{cj['chunk_index']} have {overlap_pct:.1f}% content overlap)"
                            )
                            if msg not in duplicate_sections:
                                duplicate_sections.append(msg)

    # Analyze missing sequential parent sections (1 to N)
    found_sections = sorted(
        parent_sections_found,
        key=lambda s: int(re.match(r'^Section\s+(\d+)', s).group(1)) if re.match(r'^Section\s+(\d+)', s) else 9999
    )

    missing_sections = []
    if expected_sections_count > 0:
        for i in range(1, expected_sections_count + 1):
            expected_sec = f"Section {i}"
            if expected_sec not in parent_sections_found:
                missing_sections.append(expected_sec)

    # Analyze missing parts
    missing_parts = [p for p in expected_parts if p not in found_parts_set]
    found_parts = [p for p in expected_parts if p in found_parts_set]

    # Analyze missing schedules
    missing_schedules = [s for s in expected_schedules if s not in found_schedules_set]
    found_schedules = [s for s in expected_schedules if s in found_schedules_set]

    # Aggregate diagnostic errors
    errors = []
    if missing_sections:
        errors.append(
            f"Missing {len(missing_sections)} expected statutory section(s): "
            f"{missing_sections[:10] if len(missing_sections) <= 10 else missing_sections[:10] + ['... (total ' + str(len(missing_sections)) + ')']}"
        )
    if duplicate_sections:
        errors.append(f"Duplicate provision(s) detected: {duplicate_sections}")
    if malformed_sections:
        errors.append(f"Malformed / out-of-bounds section identifier(s) detected: {malformed_sections}")
    if missing_parts:
        errors.append(f"Missing {len(missing_parts)} statutory Part(s): {missing_parts}")
    if missing_schedules:
        errors.append(f"Missing {len(missing_schedules)} Schedule(s): {missing_schedules}")

    is_structurally_complete = (len(errors) == 0)

    return StructuralCompletenessReport(
        statute_family_code=doc.statute_family_code,
        document_id=str(doc.id),
        is_structurally_complete=is_structurally_complete,
        total_chunks=total_chunks,
        expected_sections_count=expected_sections_count,
        found_sections=found_sections,
        missing_sections=missing_sections,
        duplicate_sections=duplicate_sections,
        malformed_sections=malformed_sections,
        expected_parts=expected_parts,
        found_parts=found_parts,
        missing_parts=missing_parts,
        expected_schedules=expected_schedules,
        found_schedules=found_schedules,
        missing_schedules=missing_schedules,
        errors=errors,
    )
