"""
Nigerian Statutory Legal Document Ingestion Service.

Guarantees:
1. Multi-format support: PDF, DOCX, HTML, TXT (from file bytes or protected URLs).
2. Cryptographic SHA-256 deduplication and idempotent ingestion.
3. Accurate extraction quality auditing: preserves scanned-page numbers, partial extractions, and errors.
4. FastEmbed local CPU vector generation (384 dimensions).
5. Strict Quarantine: Newly ingested documents are ALWAYS saved as PENDING_VERIFICATION
   and is_verified_authentic=False until manually approved by legal counsel.
"""
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from django.db import transaction, connection

from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    SourceType,
    VerificationStatus,
    ExtractionStatus,
)
from rag.services.downloader import (
    safe_download_legal_source_url,
    validate_file_signature,
    SSRFSecurityError,
    InvalidFileSignatureError,
)
from rag.services.extractors.pdf_extractor import extract_pdf_content
from rag.services.extractors.docx_extractor import extract_docx_content
from rag.services.extractors.html_extractor import extract_html_content
from rag.services.extractors.text_extractor import extract_plain_text_content
from rag.services.chunker import chunk_nigerian_statute, StatutoryChunk
from rag.services.vectorizer import embed_texts


class DuplicateLegalSourceError(ValueError):
    """Raised when an identical document (by SHA-256 hash) has already been ingested."""
    pass


class IngestionError(RuntimeError):
    """Raised when an unrecoverable error occurs during ingestion."""
    pass


EXTRACTOR_VERSION = "2.0.0"


def ingest_legal_source_document(
    raw_bytes: Optional[bytes] = None,
    source_url: Optional[str] = None,
    statute_family_code: str = "",
    title: str = "",
    short_title: str = "",
    jurisdiction: str = Jurisdiction.FEDERAL,
    category: str = DocumentCategory.PRIMARY_ACT,
    version_label: str = "As Enacted",
    is_current_version: bool = True,
    effective_from: Optional[Any] = None,
    effective_to: Optional[Any] = None,
    status: str = DocumentStatus.ACTIVE,
    publisher: str = "Federal Government Press, Nigeria",
    source_type: str = SourceType.OFFICIAL_GAZETTE,
    official_gazette_ref: str = "",
    original_filename: str = "",
    generate_embeddings: bool = True,
) -> Tuple[LegalSourceDocument, int, List[str]]:
    """
    Ingests, parses, chunks, and vectorizes a Nigerian statute into the legal knowledge base.

    Returns:
        (saved_document, chunk_count, warnings_list)
    """
    warnings: List[str] = []

    # 1. Acquire source raw bytes
    if source_url:
        try:
            downloaded_bytes, detected_fmt, dl_meta = safe_download_legal_source_url(source_url)
            raw_bytes = downloaded_bytes
            if not original_filename:
                original_filename = source_url.split('/')[-1].split('?')[0] or f"source.{detected_fmt}"
        except (SSRFSecurityError, InvalidFileSignatureError, ValueError) as e:
            raise IngestionError(f"Secure download failed for URL '{source_url}': {e}")
    elif raw_bytes is None:
        raise ValueError("Either raw_bytes or source_url must be provided for legal ingestion.")

    # 2. Cryptographic SHA-256 Checksum & Duplicate Audit
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    existing_doc = LegalSourceDocument.objects.filter(sha256_checksum=sha256_hash).first()
    if existing_doc:
        raise DuplicateLegalSourceError(
            f"Document with identical cryptographic SHA-256 hash ({sha256_hash[:16]}...) "
            f"already exists in the legal repository: '{existing_doc.title}' "
            f"(Short title: {existing_doc.short_title}, ID: {existing_doc.id})."
        )

    # Release database connection during CPU-intensive extraction and embedding
    connection.close()

    # 3. Detect File Signature / Format
    detected_format = validate_file_signature(raw_bytes)

    # 4. Extract Text & Audit Completeness
    extracted_text = ""
    extraction_status = ExtractionStatus.COMPLETE_TEXT
    scanned_pages: List[int] = []
    extraction_errors: List[str] = []
    extractor_meta: Dict[str, Any] = {}

    if detected_format == 'pdf':
        extracted_text, extraction_status, scanned_pages, extraction_errors, extractor_meta = extract_pdf_content(raw_bytes)
        if extraction_status == "NEEDS_OCR":
            warnings.append("Document is entirely composed of scanned images. OCR pipeline is required.")
        elif extraction_status == "MIXED_PARTIAL_TEXT":
            warnings.append(f"Document contains {len(scanned_pages)} scanned page(s) ({scanned_pages}) which require OCR.")
    elif detected_format == 'docx':
        extracted_text, extraction_status, extraction_errors, extractor_meta = extract_docx_content(raw_bytes)
    elif detected_format == 'html':
        extracted_text, extraction_status, extraction_errors, extractor_meta = extract_html_content(raw_bytes)
    elif detected_format == 'txt':
        extracted_text, extraction_status, extraction_errors, extractor_meta = extract_plain_text_content(raw_bytes)

    # 5. Structure-Aware Statutory Parsing & Chunking
    chunks: List[StatutoryChunk] = []
    if extracted_text.strip() and extraction_status != "NEEDS_OCR":
        chunks = chunk_nigerian_statute(extracted_text, statute_short_title=short_title or title)

    # 6. Dense Vector Embeddings Generation (FastEmbed local CPU BGE-small-en-v1.5)
    embeddings: List[List[float]] = []
    if chunks and generate_embeddings:
        chunk_contents = [c.content for c in chunks]
        embeddings = embed_texts(chunk_contents, batch_size=32)

    # 7. Atomic Database Ingestion
    # Ensure fresh active connection for atomic persistence
    connection.close()
    with transaction.atomic():
        doc = LegalSourceDocument.objects.create(
            statute_family_code=statute_family_code or short_title.upper().replace(' ', '_'),
            title=title,
            short_title=short_title or title[:120],
            jurisdiction=jurisdiction,
            category=category,
            version_label=version_label,
            is_current_version=is_current_version,
            effective_from=effective_from,
            effective_to=effective_to,
            status=status,
            publisher=publisher,
            source_type=source_type,
            official_gazette_ref=official_gazette_ref,
            source_url=source_url,
            original_filename=original_filename or f"document.{detected_format}",
            sha256_checksum=sha256_hash,
            # Quarantine Enforcement: ALWAYS start as unverified
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            verification_reason="",
            verification_notes="Awaiting manual counsel verification and source provenance audit.",
            extraction_status=extraction_status,
            scanned_pages=scanned_pages,
            extraction_errors=extraction_errors,
            extractor_version=EXTRACTOR_VERSION,
        )

        chunk_instances: List[LegalProvisionChunk] = []
        for idx, chunk in enumerate(chunks):
            emb = embeddings[idx] if (embeddings and idx < len(embeddings)) else [0.0] * 384
            chunk_instances.append(
                LegalProvisionChunk(
                    document=doc,
                    part_number=chunk.part_number,
                    part_title=chunk.part_title,
                    chapter_number=chunk.chapter_number,
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    citation_label=chunk.citation_label,
                    embedding=emb,
                    token_count=chunk.token_count,
                    chunk_index=chunk.chunk_index,
                )
            )

        if chunk_instances:
            LegalProvisionChunk.objects.bulk_create(chunk_instances, batch_size=200)

    return doc, len(chunk_instances), warnings
