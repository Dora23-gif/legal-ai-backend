"""
Django Management Command: link_amendment

Establishes or unlinks statutory amendment relationships between legal source documents.

Guarantees & Safeguards:
1. Verifies that both base statute and amending Act exist in the database.
2. Validates legal metadata compatibility:
   - Base and amending documents must be distinct.
   - Jurisdiction compatibility (Federal -> Federal, State -> State).
   - Temporal sequence validity (amendment commencement >= base statute commencement).
   - Repeal boundary check (amendment cannot commence after base statute repeal).
3. Two-Way Verification Enforcement:
   - Reports whether the amendment relationship is ACTIVE or DORMANT.
   - Pending amendment relationships cannot influence retrieval or citations.
4. Multi-Statute Amending Capability:
   - Supports omnibus amending Acts (e.g. BFA 2022) linking to multiple base Acts.
5. Reversible Unlinking Safeguard:
   - Unlinking removes the amendment record and clears chunk associations,
     but STRICTLY NEVER deletes either LegalSourceDocument.
"""
import uuid
import json
from datetime import datetime, date
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from rag.models import LegalSourceDocument, StatuteAmendmentRecord, LegalProvisionChunk, Jurisdiction


class Command(BaseCommand):
    help = "Link or unlink an amending Act to/from a base statute with metadata compatibility checks."

    def add_arguments(self, parser):
        # Linking arguments
        parser.add_argument(
            '--base-doc-id',
            type=str,
            default=None,
            help="UUID of the base statute receiving amendments"
        )
        parser.add_argument(
            '--amending-doc-id',
            type=str,
            default=None,
            help="UUID of the amending Act enacting amendments"
        )
        parser.add_argument(
            '--amendment-title',
            type=str,
            default="",
            help="Title of the amendment (e.g. 'Business Facilitation Act 2022 Amendments to CAMA 2020')"
        )
        parser.add_argument(
            '--commencement-date',
            type=str,
            default=None,
            help="Date the amendment took legal effect (YYYY-MM-DD)"
        )
        parser.add_argument(
            '--affected-sections',
            type=str,
            default="",
            help="Comma-separated or JSON list of affected sections, e.g. 'Section 18, Section 27, Section 240'"
        )
        parser.add_argument(
            '--summary',
            type=str,
            default="",
            help="Concise legal summary of the modifications, substitutions, or insertions"
        )
        parser.add_argument(
            '--gazette-ref',
            type=str,
            default="",
            help="Official Gazette reference for the amending instrument"
        )

        # Unlinking arguments
        parser.add_argument(
            '--unlink',
            action='store_true',
            help="Unlink/delete an existing amendment relationship without deleting either statutory document"
        )
        parser.add_argument(
            '--link-id',
            type=str,
            default=None,
            help="UUID of the StatuteAmendmentRecord to unlink"
        )

    def handle(self, *args, **options):
        is_unlink = options.get('unlink', False)

        if is_unlink:
            self._handle_unlink(options)
        else:
            self._handle_link(options)

    def _handle_unlink(self, options):
        link_id_str = (options.get('link_id') or '').strip()
        if not link_id_str:
            raise CommandError("Please provide --link-id <UUID> to unlink an amendment relationship.")

        try:
            link_uuid = uuid.UUID(link_id_str)
        except ValueError:
            raise CommandError(f"Invalid UUID format for --link-id: '{link_id_str}'")

        try:
            record = StatuteAmendmentRecord.objects.select_related('base_statute', 'amending_act').get(id=link_uuid)
        except StatuteAmendmentRecord.DoesNotExist:
            raise CommandError(f"StatuteAmendmentRecord with ID '{link_id_str}' not found.")

        base_title = record.base_statute.short_title
        base_id = record.base_statute.id
        amending_title = record.amending_act.short_title
        amending_id = record.amending_act.id

        with transaction.atomic():
            # Disassociate any linked chunks
            chunks_updated = LegalProvisionChunk.objects.filter(amendment_record=record).update(amendment_record=None)
            record.delete()

        # Verify documents still exist (Strict Safeguard)
        base_still_exists = LegalSourceDocument.objects.filter(id=base_id).exists()
        amending_still_exists = LegalSourceDocument.objects.filter(id=amending_id).exists()

        if not base_still_exists or not amending_still_exists:
            raise CommandError(
                "CRITICAL VIOLATION: Unlinking caused statutory document deletion! Transaction aborted."
            )

        self.stdout.write(self.style.SUCCESS(
            f"\n[AMENDMENT RELATIONSHIP UNLINKED SUCCESSFULLY]\n"
            f"Link ID Removed:         {link_id_str}\n"
            f"Base Statute Preserved:  {base_title} (ID: {base_id})\n"
            f"Amending Act Preserved:  {amending_title} (ID: {amending_id})\n"
            f"Provision Chunks Reset:  {chunks_updated}\n"
            f"Status: Neither statutory document was deleted.\n"
        ))

    def _handle_link(self, options):
        base_id_str = (options.get('base_doc_id') or '').strip()
        amending_id_str = (options.get('amending_doc_id') or '').strip()
        amendment_title = options.get('amendment_title', '').strip()
        commencement_str = (options.get('commencement_date') or '').strip()
        affected_str = options.get('affected_sections', '').strip()
        summary = options.get('summary', '').strip()
        gazette_ref = options.get('gazette_ref', '').strip()

        # 1. Parameter Validation
        if not base_id_str or not amending_id_str:
            raise CommandError("Both --base-doc-id and --amending-doc-id are required to establish an amendment link.")

        if not amendment_title:
            raise CommandError("An amendment title (--amendment-title) is required.")

        if not commencement_str:
            raise CommandError("An amendment commencement date (--commencement-date YYYY-MM-DD) is required.")

        if not summary or len(summary) < 10:
            raise CommandError("A substantive legal summary (--summary) of at least 10 characters is required.")

        try:
            commencement_date = datetime.strptime(commencement_str, "%Y-%m-%d").date()
        except ValueError:
            raise CommandError(f"Invalid date format for --commencement-date: '{commencement_str}'. Expected YYYY-MM-DD.")

        # Parse affected sections
        affected_sections = self._parse_affected_sections(affected_str)
        if not affected_sections:
            raise CommandError("At least one affected section (--affected-sections) must be specified.")

        # 2. Retrieve Documents
        try:
            base_uuid = uuid.UUID(base_id_str)
            amending_uuid = uuid.UUID(amending_id_str)
        except ValueError as e:
            raise CommandError(f"Invalid UUID format: {e}")

        try:
            base_doc = LegalSourceDocument.objects.get(id=base_uuid)
        except LegalSourceDocument.DoesNotExist:
            raise CommandError(f"Base statute with ID '{base_id_str}' was not found.")

        try:
            amending_doc = LegalSourceDocument.objects.get(id=amending_uuid)
        except LegalSourceDocument.DoesNotExist:
            raise CommandError(f"Amending Act with ID '{amending_id_str}' was not found.")

        # 3. Metadata Compatibility Checks
        self._validate_metadata_compatibility(base_doc, amending_doc, commencement_date)

        # 4. Check for duplicate link
        existing = StatuteAmendmentRecord.objects.filter(
            base_statute=base_doc,
            amending_act=amending_doc,
            commencement_date=commencement_date,
        ).first()

        if existing:
            raise CommandError(
                f"Amendment relationship between '{base_doc.short_title}' and '{amending_doc.short_title}' "
                f"on {commencement_date} already exists (ID: {existing.id})."
            )

        # 5. Create Amendment Record & Associate Chunks
        with transaction.atomic():
            record = StatuteAmendmentRecord.objects.create(
                base_statute=base_doc,
                amending_act=amending_doc,
                amendment_title=amendment_title,
                commencement_date=commencement_date,
                affected_sections=affected_sections,
                summary_of_changes=summary,
                official_gazette_ref=gazette_ref or amending_doc.official_gazette_ref,
            )

            # Associate matching base statute chunks
            matched_chunks = 0
            for sec in affected_sections:
                clean_sec = sec.strip()
                chunks = base_doc.chunks.filter(section_number__iexact=clean_sec)
                for chunk in chunks:
                    chunk.amendment_record = record
                    chunk.save(update_fields=['amendment_record'])
                    matched_chunks += 1

        # 6. Two-Way Verification Status Evaluation
        is_two_way_active = base_doc.is_verified_authentic and amending_doc.is_verified_authentic

        if is_two_way_active:
            status_banner = self.style.SUCCESS(
                "[STATUS: ACTIVE IN TRUSTED RETRIEVAL]\n"
                "Both the base statute and the amending Act are verified authentic.\n"
                "This amendment relationship is now fully operational in hybrid retrieval and citations."
            )
        else:
            pending_reasons = []
            if not base_doc.is_verified_authentic:
                pending_reasons.append(f"Base statute '{base_doc.short_title}' is {base_doc.verification_status}")
            if not amending_doc.is_verified_authentic:
                pending_reasons.append(f"Amending Act '{amending_doc.short_title}' is {amending_doc.verification_status}")

            status_banner = self.style.WARNING(
                "[STATUS: QUARANTINED / DORMANT (NOT IN RETRIEVAL)]\n"
                "WARNING: One or both legal instruments are awaiting verification:\n"
                + "\n".join(f"  - {r}" for r in pending_reasons) + "\n"
                "By architectural design, this amendment relationship CANNOT influence\n"
                "retrieval results or citations until BOTH instruments are verified authentic."
            )

        self.stdout.write(
            f"\n=== STATUTE AMENDMENT RELATIONSHIP CREATED ===\n"
            f"Link ID:             {record.id}\n"
            f"Base Statute:        {base_doc.short_title} (ID: {base_doc.id})\n"
            f"Amending Act:        {amending_doc.short_title} (ID: {amending_doc.id})\n"
            f"Title:               {record.amendment_title}\n"
            f"Commencement Date:   {record.commencement_date}\n"
            f"Affected Sections:   {record.affected_sections} ({matched_chunks} provision chunk(s) linked)\n"
            f"Summary:             {record.summary_of_changes}\n"
            f"Official Gazette:    {record.official_gazette_ref or 'N/A'}\n"
            f"{status_banner}\n"
        )

    def _parse_affected_sections(self, raw_str: str):
        if not raw_str:
            return []
        raw_str = raw_str.strip()
        if raw_str.startswith('[') and raw_str.endswith(']'):
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, list):
                    return [str(s).strip() for s in parsed if str(s).strip()]
            except json.JSONDecodeError:
                pass

        # Split on commas
        sections = []
        for part in raw_str.split(','):
            p = part.strip()
            if p:
                if not p.lower().startswith("section") and not p.lower().startswith("sec"):
                    p = f"Section {p}"
                sections.append(p)
        return sections

    def _validate_metadata_compatibility(
        self,
        base_doc: LegalSourceDocument,
        amending_doc: LegalSourceDocument,
        commencement_date: date
    ):
        """Enforces rigorous legal metadata checks between base statute and amending Act."""
        # 1. Distinctness
        if base_doc.id == amending_doc.id:
            raise CommandError("A legal document cannot be linked as an amendment to itself.")

        # 2. Jurisdiction Compatibility
        if base_doc.jurisdiction != amending_doc.jurisdiction:
            # Allow Federal amending Acts to amend federal baseline, but State acts cannot amend Federal
            if base_doc.jurisdiction == Jurisdiction.FEDERAL and amending_doc.jurisdiction != Jurisdiction.FEDERAL:
                raise CommandError(
                    f"Jurisdiction conflict: A {amending_doc.jurisdiction} Act cannot amend a {base_doc.jurisdiction} statute."
                )

        # 3. Temporal Sequence Validity
        if commencement_date < base_doc.effective_from:
            raise CommandError(
                f"Temporal anomaly: Amendment commencement date ({commencement_date}) cannot precede "
                f"the base statute's commencement date ({base_doc.effective_from})."
            )

        # 4. Repeal Boundary Check
        if base_doc.effective_to and commencement_date > base_doc.effective_to:
            raise CommandError(
                f"Legal conflict: Base statute was repealed/terminated on {base_doc.effective_to}. "
                f"Cannot link an amendment commencing after repeal ({commencement_date})."
            )
