"""
Django Management Command: verify_statute

Enforces manual legal counsel audit for quarantined legal documents.
No document is permitted into the trusted knowledge base without
an explicit audit trail recording counsel name, verification timestamp, and legal rationale.
"""
import uuid
from django.utils import timezone
from django.core.management.base import BaseCommand, CommandError

from rag.models import LegalSourceDocument, VerificationStatus, ExtractionStatus


class Command(BaseCommand):
    help = "Manually verify or reject a quarantined legal source document."

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            required=True,
            help="UUID of the LegalSourceDocument to verify or reject"
        )
        parser.add_argument(
            '--counsel-name',
            type=str,
            required=True,
            help="Full name or Bar roll number of reviewing legal counsel"
        )
        parser.add_argument(
            '--reason',
            type=str,
            required=True,
            help="Detailed legal rationale / gazette verification justification"
        )
        parser.add_argument(
            '--reject',
            action='store_true',
            help="Mark document as REJECTED (inauthentic, altered, or unverified)"
        )
        parser.add_argument(
            '--notes',
            type=str,
            default="",
            help="Additional audit observations or counsel notes"
        )

    def handle(self, *args, **options):
        doc_id_str = (options.get('document_id') or options.get('document-id') or '').strip()
        counsel_name = (options.get('counsel_name') or options.get('counsel-name') or '').strip()
        reason = options.get('reason', '').strip()
        is_reject = options.get('reject', False)
        notes = options.get('notes', '').strip()

        # Strict parameter validation
        if not counsel_name:
            raise CommandError("Reviewing legal counsel name (--counsel-name) cannot be empty.")

        if not reason or len(reason) < 10:
            raise CommandError(
                "A substantive legal justification (--reason) of at least 10 characters is required for audit provenance."
            )

        try:
            doc_uuid = uuid.UUID(doc_id_str)
        except ValueError:
            raise CommandError(f"Invalid UUID format for --document-id: '{doc_id_str}'")

        try:
            doc = LegalSourceDocument.objects.get(id=doc_uuid)
        except LegalSourceDocument.DoesNotExist:
            raise CommandError(f"LegalSourceDocument with ID '{doc_id_str}' was not found in database.")

        # Update verification audit trail
        now = timezone.now()
        doc.verified_at = now
        doc.verified_by = counsel_name
        doc.verification_reason = reason
        doc.verification_notes = notes

        if is_reject:
            doc.is_verified_authentic = False
            doc.verification_status = VerificationStatus.REJECTED
            doc.save()
            self.stdout.write(self.style.ERROR(
                f"\n[DOCUMENT REJECTED - EXCLUDED FROM RETRIEVAL]\n"
                f"Document ID: {doc.id}\n"
                f"Title: {doc.short_title}\n"
                f"Status: REJECTED\n"
                f"Audited by: {counsel_name}\n"
                f"Rejection Reason: {reason}\n"
                f"Audited At: {now.isoformat()}\n"
            ))
        else:
            # Strict Incomplete Extraction Guard:
            # Documents flagged NEEDS_OCR, MIXED_PARTIAL_TEXT, or FAILED, or documents
            # with unextracted scanned pages cannot be verified as complete statutory sources.
            if doc.extraction_status != ExtractionStatus.COMPLETE_TEXT:
                raise CommandError(
                    f"Cannot verify document '{doc.id}': Extraction status is '{doc.extraction_status}' (not 'COMPLETE_TEXT'). "
                    f"Incomplete statutory sources or documents requiring OCR/repair cannot be admitted into the trusted knowledge base."
                )

            if doc.scanned_pages and len(doc.scanned_pages) > 0:
                raise CommandError(
                    f"Cannot verify document '{doc.id}': Document contains {len(doc.scanned_pages)} scanned page(s) ({doc.scanned_pages}). "
                    f"Missing or rasterized pages must be extracted and inspected before verification."
                )

            # Statutory Boundary & Structural Completeness Audit:
            # Performs exact sequential validation across all sections (e.g. 1..168),
            # all statutory Parts (e.g. Parts I..XVIII), and independent Schedules.
            # Strictly distinguishes structural completeness from legal counsel authentication.
            from rag.services.statutory_structure_validator import validate_statutory_completeness
            completeness_report = validate_statutory_completeness(doc)
            if not completeness_report.is_structurally_complete:
                self.stdout.write(self.style.ERROR(f"\n{completeness_report.format_detailed_report()}\n"))
                err_summary = "; ".join(completeness_report.errors[:4])
                if len(completeness_report.errors) > 4:
                    err_summary += f" (...and {len(completeness_report.errors) - 4} more structural defect(s))"
                raise CommandError(
                    f"Cannot verify document '{doc.id}': Document failed statutory structural completeness audit. {err_summary}"
                )

            doc.is_verified_authentic = True
            doc.verification_status = VerificationStatus.VERIFIED
            doc.save()
            self.stdout.write(self.style.SUCCESS(
                f"\n[DOCUMENT VERIFIED AUTHENTIC - ADMITTED TO KNOWLEDGE BASE]\n"
                f"Document ID: {doc.id}\n"
                f"Title: {doc.short_title}\n"
                f"Status: VERIFIED_AUTHENTIC\n"
                f"Verified by: {counsel_name}\n"
                f"Legal Rationale: {reason}\n"
                f"Audit Timestamp: {now.isoformat()}\n"
                f"Eligible Provision Chunks: {doc.chunks.count()}\n"
            ))
