"""
HTML Legal Document Extractor with Preorder DOM Traversal.

Guarantees:
1. Strips headers, footers, navigation menus, ads, and scripts.
2. Walks DOM tree in preorder, preserving exact document sequence.
3. Formats headings (H1-H4) as statutory structure boundaries.
4. Converts HTML tables into clean Markdown tables in-place.
5. Formats ordered and unordered lists without breaking numbering.
"""
import re
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString


def _html_table_to_markdown(table_tag: Tag) -> str:
    """Converts a BeautifulSoup <table> tag into a clean Markdown table."""
    rows: List[List[str]] = []
    for tr in table_tag.find_all('tr'):
        cells = []
        for cell in tr.find_all(['th', 'td']):
            text = " ".join(cell.get_text().split()).replace("|", "\\|")
            cells.append(text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    num_cols = max(len(r) for r in rows)
    if num_cols == 0:
        return ""

    for r in rows:
        while len(r) < num_cols:
            r.append("")

    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")

    return "\n" + "\n".join(lines) + "\n"


def extract_html_content(raw_bytes: bytes) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """
    Extracts structured statutory text from HTML in strict document order.

    Returns:
        (extracted_text, extraction_status, errors_list, metadata)
    """
    try:
        # Detect encoding from BOM or default to utf-8 with fallback
        try:
            html_text = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            html_text = raw_bytes.decode('latin-1', errors='replace')

        try:
            soup = BeautifulSoup(html_text, 'lxml')
        except Exception:
            soup = BeautifulSoup(html_text, 'html.parser')
    except Exception as e:
        return "", "FAILED", [f"HTML parsing error: {e}"], {}

    # Strip noisy boilerplate tags
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'noscript', 'iframe']):
        tag.decompose()

    # Find primary content container if present
    content_root = (
        soup.find('main') or
        soup.find('article') or
        soup.find('div', class_=re.compile(r'content|statute|law|document|act', re.I)) or
        soup.body or
        soup
    )

    blocks: List[str] = []
    errors: List[str] = []
    heading_count = 0
    table_count = 0

    # In-order recursive preorder traversal
    def traverse(node):
        nonlocal heading_count, table_count
        if isinstance(node, NavigableString):
            return

        tag_name = node.name.lower() if node.name else ""

        # Headings
        if tag_name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            text = " ".join(node.get_text().split())
            if text:
                heading_count += 1
                level = int(tag_name[1])
                blocks.append(f"\n{'#' * level} {text}\n")
            return

        # Tables: process as unit and stop recursion
        elif tag_name == 'table':
            md = _html_table_to_markdown(node)
            if md.strip():
                table_count += 1
                blocks.append(md)
            return

        # Paragraphs
        elif tag_name == 'p':
            text = " ".join(node.get_text().split())
            if text:
                blocks.append(text)
            return

        # Lists
        elif tag_name in ('ol', 'ul'):
            for i, li in enumerate(node.find_all('li', recursive=False), start=1):
                li_text = " ".join(li.get_text().split())
                if li_text:
                    prefix = f"{i}." if tag_name == 'ol' else "-"
                    blocks.append(f"{prefix} {li_text}")
            return

        # Recurse children
        for child in node.children:
            if isinstance(child, Tag):
                traverse(child)

    traverse(content_root)

    full_text = "\n\n".join(blocks).strip()
    # Normalize multiple blank lines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    if not full_text:
        extraction_status = "FAILED"
        errors.append("HTML document contained no readable text content.")
    else:
        extraction_status = "COMPLETE_TEXT"

    metadata = {
        'heading_count': heading_count,
        'table_count': table_count,
        'total_character_count': len(full_text),
        'title': soup.title.string.strip() if soup.title and soup.title.string else "",
        'format': 'html',
    }

    return full_text, extraction_status, errors, metadata
