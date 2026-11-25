"""
Structure-Aware Nigerian Statutory Chunker.

Guarantees:
1. Distinguishes actual legislative Sections from internal numbered lists.
2. Preserves Part and Chapter structural hierarchies.
3. Keeps subsections (1), (2), (a), (b), (i), (ii) as cohesive units within their section.
4. Generates standard legal citation labels (e.g. "Section 84A, Evidence Act 2011").
5. Produces well-bounded chunk tokens for dense vector embedding.
"""
import re
from typing import List, Dict, Any, Optional


# Regex patterns for Nigerian legislative hierarchy
RE_PART = re.compile(
    r'^(?:PART|Part)\s+([IVXLCDM\d]+|[A-Z])\s*[-–—:]?\s*(.*)$',
    re.IGNORECASE
)

RE_CHAPTER = re.compile(
    r'^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+)\s*[-–—:]?\s*(.*)$',
    re.IGNORECASE
)

# True Section patterns:
# e.g.: "Section 18. Right to form a company"
#       "Section 84A. Admissibility of electronic records"
RE_SECTION_EXPLICIT = re.compile(
    r'^Section\s+(\d+[A-Z]?)\.\s+([A-Z][^\n]{2,150})$'
)

# Numbered list items inside a section body (MUST NOT be treated as sections):
# e.g. "(1)", "(2)", "(a)", "(b)", "(i)", "(ii)", "1.", "2."
RE_INTERNAL_LIST = re.compile(
    r'^(?:\(\d+[a-z]?\)|\([a-z]\)|\([ivxlcdm]+\))\s+',
    re.IGNORECASE
)


def _extract_sec_number_int(sec_str: str) -> Optional[int]:
    """Extracts integer value from 'Section 18', '84A', etc."""
    m = re.search(r'\d+', sec_str)
    return int(m.group(0)) if m else None


class StatutoryChunk:
    def __init__(
        self,
        part_number: str,
        part_title: str,
        chapter_number: str,
        section_number: str,
        section_title: str,
        content: str,
        citation_label: str,
        chunk_index: int,
    ):
        self.part_number = part_number
        self.part_title = part_title
        self.chapter_number = chapter_number
        self.section_number = section_number
        self.section_title = section_title
        self.content = content.strip()
        self.citation_label = citation_label
        self.chunk_index = chunk_index
        # Approximate token count (words * 1.33)
        self.token_count = max(1, int(len(self.content.split()) * 1.33))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'part_number': self.part_number,
            'part_title': self.part_title,
            'chapter_number': self.chapter_number,
            'section_number': self.section_number,
            'section_title': self.section_title,
            'content': self.content,
            'citation_label': self.citation_label,
            'chunk_index': self.chunk_index,
            'token_count': self.token_count,
        }


ORDINAL_SCHEDULES_CHUNK = {
    "first": "First Schedule", "second": "Second Schedule", "third": "Third Schedule",
    "fourth": "Fourth Schedule", "fifth": "Fifth Schedule", "sixth": "Sixth Schedule",
    "seventh": "Seventh Schedule", "eighth": "Eighth Schedule", "ninth": "Ninth Schedule",
    "tenth": "Tenth Schedule", "eleventh": "Eleventh Schedule", "twelfth": "Twelfth Schedule",
    "thirteenth": "Thirteenth Schedule", "fourteenth": "Fourteenth Schedule", "fifteenth": "Fifteenth Schedule",
}


def _chunk_sequential_gazette_statute(
    text: str,
    statute_short_title: str,
    arr_match: re.Match,
    enacted_match: re.Match,
) -> List[StatutoryChunk]:
    """Chunks large Nigerian Gazette statutes with an Arrangement of Sections and sequential sections."""
    arrangement_map = {}
    arr_segment = text[arr_match.end():enacted_match.start()]
    for line in arr_segment.split('\n'):
        m = re.match(r'^\s*(\d+)\.?\s+([A-Z][^\n\.\;]+)', line.strip())
        if m:
            s_num = int(m.group(1))
            s_title = m.group(2).strip()
            if 1 <= s_num <= 1000 and s_num not in arrangement_map:
                arrangement_map[s_num] = s_title

    max_expected_sections = max(arrangement_map.keys()) if arrangement_map else 870

    body_text = text[enacted_match.start():]
    lines = body_text.split('\n')

    chunks: List[StatutoryChunk] = []
    chunk_idx = 1
    current_part_num = "Part A"
    current_part_title = ""
    current_chap_num = ""

    curr_expected_sec = 1
    sec_lines: List[str] = []

    in_schedules = False
    current_sched_num = ""
    sched_lines: List[str] = []

    def flush_sec():
        nonlocal chunk_idx, sec_lines, curr_expected_sec
        if sec_lines and curr_expected_sec > 1:
            sec_int = curr_expected_sec - 1
            sec_str = f"Section {sec_int}"
            title = arrangement_map.get(sec_int, f"Provisions of {sec_str}")
            content = "\n".join(sec_lines).strip()
            chunks.append(StatutoryChunk(
                part_number=current_part_num,
                part_title=current_part_title,
                chapter_number=current_chap_num,
                section_number=sec_str,
                section_title=title,
                content=content,
                citation_label=f"{sec_str}, {statute_short_title}",
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        sec_lines = []

    def flush_sched():
        nonlocal chunk_idx, sched_lines, current_sched_num
        if current_sched_num and sched_lines:
            content = "\n".join(sched_lines).strip()
            chunks.append(StatutoryChunk(
                part_number="Schedules",
                part_title=f"Statutory Schedules to {statute_short_title}",
                chapter_number="",
                section_number=current_sched_num,
                section_title=f"{current_sched_num} to {statute_short_title}",
                content=content,
                citation_label=f"{current_sched_num}, {statute_short_title}",
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        sched_lines = []

    for line in lines:
        line_clean = line.strip()

        if curr_expected_sec > max_expected_sections and not in_schedules:
            flush_sec()
            in_schedules = True

        if not in_schedules:
            m_part = re.match(r'^\s*PART\s+([A-G])\b[\s\uFFFD\u2013\u2014\-\–\—\:]*(.*)$', line_clean, re.IGNORECASE)
            if m_part and not re.search(r'\b(?:of the|schedule|apply|pursuant|under)\b', line_clean, re.IGNORECASE):
                current_part_num = f"Part {m_part.group(1).upper()}"
                current_part_title = m_part.group(2).strip()
                continue

            pattern = rf'^{curr_expected_sec}\.[\s\uFFFD\u2013\u2014\-\–\—\.]*'
            if re.match(pattern, line_clean):
                flush_sec()
                curr_expected_sec += 1
                sec_lines = [line_clean]
            else:
                if sec_lines:
                    sec_lines.append(line_clean)
        else:
            m_sched = re.match(
                r'^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH)\s+SCHEDULE\b',
                line_clean,
                re.IGNORECASE
            )
            if m_sched:
                flush_sched()
                sched_tok = m_sched.group(1).lower()
                current_sched_num = ORDINAL_SCHEDULES_CHUNK.get(sched_tok, f"{m_sched.group(1).title()} Schedule")
                sched_lines = [line_clean]
            elif "SCHEDULE TO THE" in line_clean and "BILL" in line_clean:
                flush_sched()
                current_sched_num = "Authentication Schedule"
                sched_lines = [line_clean]
            else:
                if current_sched_num:
                    sched_lines.append(line_clean)

    if not in_schedules:
        flush_sec()
    else:
        flush_sched()

    return chunks


def chunk_nigerian_statute(
    text: str,
    statute_short_title: str = "Nigerian Statute"
) -> List[StatutoryChunk]:
    """
    Parses a Nigerian statute text into discrete, section-level chunks.
    Disambiguates top-level sections from internal lists.
    """
    arr_match = re.search(r'ARRANGEMENT OF S\s*ECTIONS', text, re.IGNORECASE)
    enacted_match = re.search(r'(?:ENACTED\s+by\s+the\s+National\s+Assembly|BE\s+IT\s+ENACTED)', text, re.IGNORECASE)

    if arr_match and enacted_match:
        return _chunk_sequential_gazette_statute(text, statute_short_title, arr_match, enacted_match)

    lines = text.split('\n')
    chunks: List[StatutoryChunk] = []

    current_part_num = ""
    current_part_title = ""
    current_chap_num = ""
    current_sec_num = ""
    current_sec_title = ""
    current_sec_lines: List[str] = []
    chunk_counter = 1

    def flush_current_section():
        nonlocal chunk_counter, current_sec_lines
        if current_sec_num and current_sec_lines:
            content_text = "\n".join(current_sec_lines).strip()
            if content_text:
                citation = f"{current_sec_num}, {statute_short_title}"
                chunk = StatutoryChunk(
                    part_number=current_part_num,
                    part_title=current_part_title,
                    chapter_number=current_chap_num,
                    section_number=current_sec_num,
                    section_title=current_sec_title,
                    content=content_text,
                    citation_label=citation,
                    chunk_index=chunk_counter,
                )
                chunks.append(chunk)
                chunk_counter += 1
        current_sec_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            if current_sec_lines:
                current_sec_lines.append("")
            i += 1
            continue

        # 1. Check for PART boundary
        part_match = RE_PART.match(line)
        if part_match:
            current_part_num = f"Part {part_match.group(1).upper()}"
            current_part_title = part_match.group(2).strip()
            # If title is on next line
            if not current_part_title and (i + 1 < len(lines)) and not lines[i + 1].strip().startswith(('Section', 'Chapter', 'Part')):
                i += 1
                current_part_title = lines[i].strip()
            i += 1
            continue

        # 2. Check for CHAPTER boundary
        chap_match = RE_CHAPTER.match(line)
        if chap_match:
            current_chap_num = f"Chapter {chap_match.group(1)}"
            i += 1
            continue

        # 3. Check if line is an internal list item (e.g. "(1)", "(a)", "(i)")
        # Must NOT treat as a new section!
        if RE_INTERNAL_LIST.match(line):
            current_sec_lines.append(line)
            i += 1
            continue

        # 4. Check for Explicit Section boundary ("Section 18. Right to...")
        sec_match = RE_SECTION_EXPLICIT.match(line)
        if sec_match:
            flush_current_section()
            sec_num = sec_match.group(1)
            sec_title = sec_match.group(2).strip()

            current_sec_num = f"Section {sec_num}"
            current_sec_title = sec_title
            current_sec_lines = [f"{current_sec_num}. {sec_title}"]
            i += 1
            continue

        # 5. Check alternate section format without "Section" keyword (e.g. "18. Right to form a company")
        alt_sec_match = re.match(r'^(\d+[A-Z]?)\.\s+([A-Z][^\n]{2,120})$', line)
        if alt_sec_match:
            cand_sec_num = alt_sec_match.group(1)
            cand_title = alt_sec_match.group(2).strip()
            cand_int = _extract_sec_number_int(cand_sec_num)
            curr_int = _extract_sec_number_int(current_sec_num) if current_sec_num else None

            # Disambiguation:
            # If candidate title ends with a colon/semicolon, or if cand_int <= curr_int,
            # it is an internal numbered subsection (e.g. "1. In any proceeding..."), NOT a new section.
            is_subsection = False
            if cand_title.endswith((':', ';')):
                is_subsection = True
            elif curr_int is not None and cand_int is not None and cand_int <= curr_int:
                is_subsection = True

            if is_subsection:
                current_sec_lines.append(line)
                i += 1
                continue

            # It is a true top-level Section
            flush_current_section()
            current_sec_num = f"Section {cand_sec_num}"
            current_sec_title = cand_title
            current_sec_lines = [f"{current_sec_num}. {cand_title}"]
            i += 1
            continue

        # Default: normal text belonging to current section
        if not current_sec_num:
            # Preamble, Long Title, or Enactment Clause
            current_sec_num = "Preamble"
            current_sec_title = "Long Title & Enactment Formula"
            current_sec_lines = [line]
        else:
            current_sec_lines.append(line)

        i += 1

    # Flush final section
    flush_current_section()

    # Fallback: if no formal sections were matched (e.g. scanned/plain legal notice), chunk by paragraphs
    if not chunks and text.strip():
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for idx, p in enumerate(paragraphs, start=1):
            chunks.append(StatutoryChunk(
                part_number="",
                part_title="",
                chapter_number="",
                section_number=f"Paragraph {idx}",
                section_title=p[:60] + "..." if len(p) > 60 else p,
                content=p,
                citation_label=f"Paragraph {idx}, {statute_short_title}",
                chunk_index=idx,
            ))

    return chunks
