"""
PDF Legal Document Extractor with Page-by-Page Scanned Detection.

Guarantees:
1. Page-by-page text density and image audit.
2. Identifies mixed-content PDFs (some pages digital, some scanned images).
3. Explicitly flags scanned pages and records their exact page numbers.
4. Never silently discards pages or marks incomplete extraction as successful.
"""
import io
import re
from typing import Dict, Any, List, Tuple
from pypdf import PdfReader


class ScannedPDFError(ValueError):
    """Raised when a PDF is entirely image-based and requires OCR."""
    pass


def extract_pdf_content(raw_bytes: bytes) -> Tuple[str, str, List[int], List[str], Dict[str, Any]]:
    """
    Extracts text and audits scanned pages from raw PDF bytes.

    Returns:
        (extracted_text, extraction_status, scanned_pages, errors_list, metadata)
    """
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as e:
        return "", "FAILED", [], [f"PDF parsing error: {e}"], {}

    total_pages = len(reader.pages)
    if total_pages == 0:
        return "", "FAILED", [], ["PDF has 0 pages."], {}

    page_texts: List[str] = []
    scanned_pages: List[int] = []
    errors: List[str] = []
    total_chars = 0

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = ""
            errors.append(f"Page {idx}: text extraction exception: {e}")

        # Clean excess line breaks and page numbers
        cleaned_page = re.sub(r'\r\n|\r', '\n', text).strip()
        char_count = len(cleaned_page)
        total_chars += char_count

        # Check for image objects on the page
        has_images = False
        try:
            has_images = len(page.images) > 0
        except Exception:
            pass

        # If page has negligible text (< 40 chars) but has images, it's a scanned page
        if char_count < 40 and has_images:
            scanned_pages.append(idx)
            page_texts.append(f"\n[PAGE {idx}: SCANNED IMAGE - OCR REQUIRED]\n")
        elif char_count == 0:
            scanned_pages.append(idx)
            page_texts.append(f"\n[PAGE {idx}: EMPTY / BLANK PAGE]\n")
        else:
            page_texts.append(f"\n--- PAGE {idx} ---\n{cleaned_page}")

    full_text = "\n".join(page_texts).strip()

    # Determine extraction status
    if len(scanned_pages) == total_pages or total_chars < 50:
        extraction_status = "NEEDS_OCR"
        errors.append(
            f"All {total_pages} page(s) lack an embedded text layer (total characters: {total_chars}). "
            "Full optical character recognition (OCR) is required."
        )
    elif len(scanned_pages) > 0:
        extraction_status = "MIXED_PARTIAL_TEXT"
        errors.append(
            f"Document contains {len(scanned_pages)} scanned image page(s) out of {total_pages} total pages: "
            f"Pages {scanned_pages}. Text on these pages could not be extracted digitally."
        )
    else:
        extraction_status = "COMPLETE_TEXT"

    metadata = {
        'total_pages': total_pages,
        'scanned_pages_count': len(scanned_pages),
        'scanned_pages': scanned_pages,
        'total_character_count': total_chars,
        'average_chars_per_page': round(total_chars / total_pages, 1) if total_pages else 0,
        'format': 'pdf',
    }

    return full_text, extraction_status, scanned_pages, errors, metadata
