"""
Django Management Command: ingest_statute

Ingests a Nigerian legal source document (PDF, DOCX, HTML, or TXT)
from a local file path or secure URL.

All ingested documents are placed in strict QUARANTINE (PENDING_VERIFICATION)
and cannot enter the retrieval knowledge base until audited with verify_statute.
"""
import os
import sys
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError

from rag.models import Jurisdiction, DocumentCategory, DocumentStatus, SourceType
from rag.services.ingestion_service import (
    ingest_legal_source_document,
    DuplicateLegalSourceError,
    IngestionError,
)


class Command(BaseCommand):
    help = "Ingest a Nigerian statute or legal instrument into quarantine storage."

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, help="Absolute or relative path to legal source document file")
        parser.add_argument('--url', type=str, help="Secure HTTPS URL to legal source document")
        parser.add_argument('--statute-code', type=str, required=True, help="Statute family code (e.g. 'FED_CAMA', 'FED_EVIDENCE')")
        parser.add_argument('--title', type=str, required=True, help="Full statute title")
        parser.add_argument('--short-title', type=str, required=True, help="Short title (e.g. 'CAMA 2020')")
        parser.add_argument('--jurisdiction', type=str, default=Jurisdiction.FEDERAL, choices=Jurisdiction.values)
        parser.add_argument('--category', type=str, default=DocumentCategory.PRIMARY_ACT, choices=DocumentCategory.values)
        parser.add_argument('--version-label', type=str, default="As Enacted")
        parser.add_argument('--effective-from', type=str, required=True, help="Commencement date (YYYY-MM-DD)")
        parser.add_argument('--effective-to', type=str, default=None, help="Repeal/amendment date (YYYY-MM-DD) or omit")
        parser.add_argument('--status', type=str, default=DocumentStatus.ACTIVE, choices=DocumentStatus.values)
        parser.add_argument('--publisher', type=str, default="Federal Government Press, Nigeria")
        parser.add_argument('--source-type', type=str, default=SourceType.OFFICIAL_GAZETTE, choices=SourceType.values)
        parser.add_argument('--gazette-ref', type=str, default="", help="Official Gazette volume/number citation")
        parser.add_argument('--no-embeddings', action='store_true', help="Skip dense vector embedding generation")

    def handle(self, *args, **options):
        file_path = options.get('file')
        source_url = options.get('url')

        if not file_path and not source_url:
            raise CommandError("Either --file or --url must be provided.")

        if file_path and source_url:
            raise CommandError("Provide either --file or --url, not both.")

        raw_bytes = None
        orig_filename = ""

        if file_path:
            if not os.path.exists(file_path):
                raise CommandError(f"File not found: {file_path}")
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            orig_filename = os.path.basename(file_path)

        eff_from_str = options.get('effective_from') or options.get('effective-from')
        eff_to_str = options.get('effective_to') or options.get('effective-to')
        statute_code = options.get('statute_code') or options.get('statute-code')
        short_title = options.get('short_title') or options.get('short-title')
        version_label = options.get('version_label') or options.get('version-label') or "As Enacted"
        source_type = options.get('source_type') or options.get('source-type') or SourceType.OFFICIAL_GAZETTE
        gazette_ref = options.get('gazette_ref') or options.get('gazette-ref') or ""
        no_embeddings = options.get('no_embeddings') or options.get('no-embeddings', False)

        try:
            eff_from = datetime.strptime(eff_from_str, '%Y-%m-%d').date()
            eff_to = datetime.strptime(eff_to_str, '%Y-%m-%d').date() if eff_to_str else None
        except ValueError as e:
            raise CommandError(f"Invalid date format (must be YYYY-MM-DD): {e}")

        self.stdout.write(f"Starting ingestion of '{short_title}'...")

        try:
            doc, chunk_count, warnings = ingest_legal_source_document(
                raw_bytes=raw_bytes,
                source_url=source_url,
                statute_family_code=statute_code,
                title=options['title'],
                short_title=short_title,
                jurisdiction=options['jurisdiction'],
                category=options['category'],
                version_label=version_label,
                effective_from=eff_from,
                effective_to=eff_to,
                status=options['status'],
                publisher=options['publisher'],
                source_type=source_type,
                official_gazette_ref=gazette_ref,
                original_filename=orig_filename,
                generate_embeddings=not no_embeddings,
            )
        except DuplicateLegalSourceError as e:
            raise CommandError(f"[DUPLICATE REJECTED] {e}")
        except IngestionError as e:
            raise CommandError(f"[INGESTION FAILED] {e}")
        except Exception as e:
            raise CommandError(f"Unexpected ingestion failure: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"\n[INGESTION SUCCESSFUL - QUARANTINED]\n"
            f"Document ID: {doc.id}\n"
            f"Short Title: {doc.short_title}\n"
            f"SHA-256 Hash: {doc.sha256_checksum}\n"
            f"Extraction Status: {doc.extraction_status}\n"
            f"Chunks Generated: {chunk_count}\n"
            f"Quarantine Status: {doc.verification_status} (is_verified_authentic={doc.is_verified_authentic})\n"
            f"Note: This document CANNOT be retrieved until verified via:\n"
            f"  python manage.py verify_statute --document-id {doc.id} --counsel-name \"Counsel Name\" --reason \"Reason\""
        ))

        if warnings:
            self.stdout.write(self.style.WARNING("\nWarnings during ingestion:"))
            for w in warnings:
                self.stdout.write(self.style.WARNING(f"  - {w}"))
