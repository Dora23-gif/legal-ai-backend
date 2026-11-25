"""
DOCX Legal Document Extractor with In-Order Paragraph and Table Traversal.

Guarantees:
1. Walks the DOCX element body in strict sequential document order.
2. Interleaves paragraphs, headings, and tables exactly as drafted.
3. Converts DOCX tables into readable Markdown pipe tables in-place.
4. Preserves statutory schedules, fine tables, and corporate forms in context.
"""
import io
import re
from typing import Dict, Any, List, Tuple
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table


def _format_table_as_markdown(table: Table) -> str:
    """Converts a python-docx Table into clean Markdown table format."""
    rows_data: List[List[str]] = []
    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            cell_text = " ".join(cell.text.split())
            row_cells.append(cell_text.replace("|", "\\|"))
        rows_data.append(row_cells)

    if not rows_data:
        return ""

    num_cols = max(len(r) for r in rows_data) if rows_data else 0
    if num_cols == 0:
        return ""

    # Pad any uneven rows
    for r in rows_data:
        while len(r) < num_cols:
            r.append("")

    lines = []
    # Header row
    lines.append("| " + " | ".join(rows_data[0]) + " |")
    # Separator row
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    # Body rows
    for r in rows_data[1:]:
        lines.append("| " + " | ".join(r) + " |")

    return "\n" + "\n".join(lines) + "\n"


def extract_docx_content(raw_bytes: bytes) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """
    Extracts text and tables in exact document order from raw DOCX bytes.

    Returns:
        (extracted_text, extraction_status, errors_list, metadata)
    """
    try:
        doc = Document(io.BytesIO(raw_bytes))
    except Exception as e:
        return "", "FAILED", [f"DOCX parsing error: {e}"], {}

    blocks: List[str] = []
    errors: List[str] = []
    paragraph_count = 0
    table_count = 0

    try:
        # Traverse body elements in sequential document order
        for child in doc.element.body.iterchildren():
            # Check local tag name
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

            if tag == 'p':
                p = Paragraph(child, doc)
                text = p.text.strip()
                if text:
                    paragraph_count += 1
                    style_name = p.style.name.lower() if p.style else ""
                    if 'heading 1' in style_name:
                        blocks.append(f"\n# {text}\n")
                    elif 'heading 2' in style_name:
                        blocks.append(f"\n## {text}\n")
                    elif 'heading 3' in style_name:
                        blocks.append(f"\n### {text}\n")
                    else:
                        blocks.append(text)

            elif tag == 'tbl':
                tbl = Table(child, doc)
                table_md = _format_table_as_markdown(tbl)
                if table_md.strip():
                    table_count += 1
                    blocks.append(table_md)

    except Exception as e:
        errors.append(f"Error during DOCX in-order element traversal: {e}")

    full_text = "\n\n".join(blocks).strip()

    if not full_text:
        extraction_status = "FAILED"
        errors.append("DOCX contains zero readable text or tables.")
    elif errors:
        extraction_status = "MIXED_PARTIAL_TEXT"
    else:
        extraction_status = "COMPLETE_TEXT"

    metadata = {
        'paragraph_count': paragraph_count,
        'table_count': table_count,
        'total_character_count': len(full_text),
        'format': 'docx',
    }

    return full_text, extraction_status, errors, metadata
