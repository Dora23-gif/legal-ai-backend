"""
Django Management Command: inspect_statute

Read-only inspection CLI for legal source documents and statutory provision chunks.
Guarantees:
1. Zero database mutations (read-only execution).
2. Zero exposure of database credentials or environment secrets.
3. Detailed reporting of extraction completeness, scanned pages, and verification status.
"""
import uuid
from typing import Optional
from django.core.management.base import BaseCommand, CommandError
from rag.models import LegalSourceDocument, LegalProvisionChunk, StatuteAmendmentRecord


class Command(BaseCommand):
    help = "Read-only inspection of ingested legal source documents and extracted provisions."

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            default=None,
            help="UUID of the specific LegalSourceDocument to inspect"
        )
        parser.add_argument(
            '--family-code',
            type=str,
            default=None,
            help="Statute family code to inspect (e.g. 'CAMA-2020', 'EVIDENCE-ACT-2011')"
        )
        parser.add_argument(
            '--list',
            action='store_true',
            help="List all legal source documents currently in the database"
        )
        parser.add_argument(
            '--show-sections',
            type=int,
            default=5,
            help="Number of extracted sections to preview (default: 5)"
        )

    def handle(self, *args, **options):
        doc_id_str = options.get('document_id')
        family_code = options.get('family_code')
        is_list = options.get('list', False)
        show_sections = options.get('show_sections', 5)

        if is_list:
            self._list_documents()
            return

        if doc_id_str:
            self._inspect_by_id(doc_id_str.strip(), show_sections)
            return

        if family_code:
            self._inspect_by_family(family_code.strip(), show_sections)
            return

        raise CommandError(
            "Please provide an inspection target: --document-id <UUID>, --family-code <CODE>, or --list."
        )

    def _list_documents(self):
        docs = LegalSourceDocument.objects.all().order_by('-created_at')
        total = docs.count()
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n=== LEGAL SOURCE DOCUMENTS IN DATABASE ({total} total) ==="
        ))

        if total == 0:
            self.stdout.write(self.style.WARNING("No legal source documents found in the database."))
            return

        for doc in docs:
            ver_tag = (
                self.style.SUCCESS("[VERIFIED]")
                if doc.is_verified_authentic
                else self.style.WARNING(f"[{doc.verification_status}]")
            )
            curr_tag = "[CURRENT]" if doc.is_current_version else "[HISTORICAL]"
            self.stdout.write(
                f"- ID: {doc.id}\n"
                f"  Title: {doc.short_title} ({doc.version_label}) {curr_tag} {ver_tag}\n"
                f"  Family: {doc.statute_family_code} | Jurisdiction: {doc.jurisdiction}\n"
                f"  Extraction: {doc.extraction_status} | Scanned Pages: {len(doc.scanned_pages or [])} | Chunks: {doc.chunks.count()}\n"
                f"  Commenced: {doc.effective_from} | Gazette: {doc.official_gazette_ref or 'N/A'}\n"
            )

    def _inspect_by_id(self, doc_id_str: str, show_sections: int):
        try:
            doc_uuid = uuid.UUID(doc_id_str)
        except ValueError:
            raise CommandError(f"Invalid UUID format: '{doc_id_str}'")

        try:
            doc = LegalSourceDocument.objects.get(id=doc_uuid)
        except LegalSourceDocument.DoesNotExist:
            raise CommandError(f"LegalSourceDocument with ID '{doc_id_str}' not found.")

        self._print_detailed_document(doc, show_sections)

    def _inspect_by_family(self, family_code: str, show_sections: int):
        docs = LegalSourceDocument.objects.filter(statute_family_code__iexact=family_code).order_by('-effective_from')
        if not docs.exists():
            raise CommandError(f"No documents found for statute family code: '{family_code}'.")

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\n=== STATUTE FAMILY: {family_code} ({docs.count()} version(s) found) ==="
        ))
        for doc in docs:
            self._print_detailed_document(doc, show_sections)

    def _print_detailed_document(self, doc: LegalSourceDocument, show_sections: int):
        self.stdout.write(self.style.MIGRATE_HEADING("\n" + "=" * 70))
        self.stdout.write(self.style.MIGRATE_LABEL(f"DOCUMENT: {doc.title}"))
        self.stdout.write("=" * 70)

        # 1. Core Identification
        self.stdout.write(
            f"  ID:                     {doc.id}\n"
            f"  Statute Family Code:    {doc.statute_family_code}\n"
            f"  Short Title:            {doc.short_title}\n"
            f"  Jurisdiction:           {doc.jurisdiction}\n"
            f"  Category:               {doc.category}\n"
            f"  Version Label:          {doc.version_label}\n"
            f"  Is Current Version:     {doc.is_current_version}\n"
            f"  Status:                 {doc.status}\n"
        )

        # 2. Chronological & Effective Dates
        self.stdout.write(self.style.MIGRATE_LABEL("  [CHRONOLOGICAL DATES & LINEAGE]"))
        self.stdout.write(
            f"  Commencement Date:      {doc.effective_from}\n"
            f"  Effective To (Repeal):  {doc.effective_to or 'In Force (Active)'}\n"
            f"  Superseded By:          {doc.superseded_by_id or 'None'}\n"
        )

        # 3. Provenance & Integrity
        self.stdout.write(self.style.MIGRATE_LABEL("  [PROVENANCE & DOCUMENT INTEGRITY]"))
        self.stdout.write(
            f"  Publisher:              {doc.publisher}\n"
            f"  Source Type:            {doc.source_type}\n"
            f"  Official Gazette Ref:   {doc.official_gazette_ref or 'N/A'}\n"
            f"  Source URL:             {doc.source_url or 'N/A'}\n"
            f"  Original Filename:      {doc.original_filename}\n"
            f"  SHA-256 Checksum:       {doc.sha256_checksum}\n"
            f"  Integrity Note:         SHA-256 verifies file integrity against tampering/corruption;\n"
            f"                          legal authenticity requires counsel audit against official gazettes.\n"
        )

        # 4. Extraction Audit
        self.stdout.write(self.style.MIGRATE_LABEL("  [EXTRACTION AUDIT]"))
        scanned_count = len(doc.scanned_pages or [])
        errors_count = len(doc.extraction_errors or [])
        self.stdout.write(
            f"  Extraction Status:      {doc.extraction_status}\n"
            f"  Extractor Version:      {doc.extractor_version}\n"
            f"  Scanned Pages:          {scanned_count} page(s) {doc.scanned_pages if scanned_count else ''}\n"
            f"  Extraction Errors:      {errors_count} error(s) {doc.extraction_errors if errors_count else ''}\n"
        )

        # 5. Verification & Quarantine Audit
        self.stdout.write(self.style.MIGRATE_LABEL("  [VERIFICATION & QUARANTINE STATUS]"))
        ver_status_display = (
            self.style.SUCCESS(f"{doc.verification_status} (AUTHENTIC)")
            if doc.is_verified_authentic
            else self.style.WARNING(f"{doc.verification_status} (QUARANTINED - EXCLUDED FROM RETRIEVAL)")
        )
        self.stdout.write(
            f"  Verification Status:    {ver_status_display}\n"
            f"  Verified Authentic:     {doc.is_verified_authentic}\n"
            f"  Verified At:            {doc.verified_at.isoformat() if doc.verified_at else 'N/A'}\n"
            f"  Verified By:            {doc.verified_by or 'N/A'}\n"
            f"  Verification Reason:    {doc.verification_reason or 'N/A'}\n"
            f"  Verification Notes:     {doc.verification_notes or 'N/A'}\n"
        )

        # 6. Statutory Structural Completeness Audit (Read-Only)
        from rag.services.statutory_structure_validator import validate_statutory_completeness
        comp_report = validate_statutory_completeness(doc)
        if comp_report.expected_sections_count > 0:
            self.stdout.write(self.style.MIGRATE_LABEL("  [STATUTORY STRUCTURAL COMPLETENESS AUDIT]"))
            status_text = (
                self.style.SUCCESS("PASS (100% Structural Completeness)")
                if comp_report.is_structurally_complete
                else self.style.WARNING(
                    f"INCOMPLETE ({len(comp_report.missing_sections)} missing sections, {len(comp_report.missing_parts)} missing parts)"
                )
            )
            self.stdout.write(
                f"  Structural Status:      {status_text}\n"
                f"  Sections Accounted:     {len(comp_report.found_sections)} of {comp_report.expected_sections_count}\n"
                f"  Parts Accounted:        {len(comp_report.found_parts)} of {len(comp_report.expected_parts)}\n"
                f"  Schedules Accounted:    {len(comp_report.found_schedules)} of {len(comp_report.expected_schedules)}\n"
                f"  Duplicate Sections:     {len(comp_report.duplicate_sections)}\n"
                f"  Malformed Sections:     {len(comp_report.malformed_sections)}\n"
            )

        # 7. Statutory Provisions & Chunks
        chunks = doc.chunks.all().order_by('chunk_index')
        total_chunks = chunks.count()
        self.stdout.write(self.style.MIGRATE_LABEL(f"  [PROVISION CHUNKS ({total_chunks} total)]"))

        if total_chunks > 0 and show_sections > 0:
            preview_chunks = chunks[:show_sections]
            for ch in preview_chunks:
                self.stdout.write(
                    f"    - [{ch.citation_label}] {ch.section_number}: {ch.section_title or 'Untitled'} "
                    f"({ch.token_count} tokens) [Index: {ch.chunk_index}]"
                )
            if total_chunks > show_sections:
                self.stdout.write(f"      ... and {total_chunks - show_sections} more provision chunk(s).")

        # 7. Amendment Relationships
        amendments_received = doc.amendments_received.all()
        amendments_made = doc.amendments_made.all()

        if amendments_received.exists() or amendments_made.exists():
            self.stdout.write(self.style.MIGRATE_LABEL("  [AMENDMENT RELATIONSHIPS]"))
            for ar in amendments_received:
                two_way_active = ar.base_statute.is_verified_authentic and ar.amending_act.is_verified_authentic
                active_str = "[ACTIVE IN RETRIEVAL]" if two_way_active else "[DORMANT/PENDING VERIFICATION]"
                self.stdout.write(
                    f"    * AMENDED BY: {ar.amending_act.short_title} {active_str}\n"
                    f"      Link ID: {ar.id}\n"
                    f"      Commencement Date: {ar.commencement_date}\n"
                    f"      Affected Sections: {ar.affected_sections}\n"
                    f"      Summary: {ar.summary_of_changes}\n"
                )
            for am in amendments_made:
                two_way_active = am.base_statute.is_verified_authentic and am.amending_act.is_verified_authentic
                active_str = "[ACTIVE IN RETRIEVAL]" if two_way_active else "[DORMANT/PENDING VERIFICATION]"
                self.stdout.write(
                    f"    * AMENDS BASE STATUTE: {am.base_statute.short_title} {active_str}\n"
                    f"      Link ID: {am.id}\n"
                    f"      Commencement Date: {am.commencement_date}\n"
                    f"      Affected Sections: {am.affected_sections}\n"
                    f"      Summary: {am.summary_of_changes}\n"
                )

        self.stdout.write("-" * 70 + "\n")
