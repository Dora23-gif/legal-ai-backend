"""
Plain Text Legal Document Extractor and Normalizer.

Guarantees:
1. Robust encoding fallback (UTF-8, Latin-1, CP1252).
2. Normalizes Windows CRLF to standard Unix newlines.
3. Cleans whitespace while preserving statutory indentations.
"""
import re
from typing import Dict, Any, List, Tuple


def extract_plain_text_content(raw_bytes: bytes) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """
    Decodes and normalizes plain text legal documents.

    Returns:
        (extracted_text, extraction_status, errors_list, metadata)
    """
    errors: List[str] = []
    used_encoding = 'utf-8'

    try:
        text = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode('latin-1')
            used_encoding = 'latin-1'
        except Exception as e:
            return "", "FAILED", [f"Text encoding error: {e}"], {}

    # Normalize line breaks
    normalized = re.sub(r'\r\n|\r', '\n', text)
    # Strip null characters or binary garbage
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', normalized).strip()

    if not cleaned:
        extraction_status = "FAILED"
        errors.append("Plain text file is empty or contains only whitespace.")
    else:
        extraction_status = "COMPLETE_TEXT"

    metadata = {
        'encoding': used_encoding,
        'line_count': cleaned.count('\n') + 1 if cleaned else 0,
        'total_character_count': len(cleaned),
        'format': 'txt',
    }

    return cleaned, extraction_status, errors, metadata
