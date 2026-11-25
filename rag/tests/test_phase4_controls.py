"""
Unit and Integration Tests for Phase 4 Controls and Safeguards.

Tests:
1. Source Registry Metadata Structure (5 dates, confirmed references, uninvented hashes).
2. Read-only Statute Inspection Command (inspect_statute).
3. Amendment Linking & Reversible Unlinking (link_amendment).
4. Two-Way Verification Enforcement in Retrieval & Lineage Notices.
5. Incomplete Extraction Quarantine Guard in verify_statute (NEEDS_OCR / MIXED_PARTIAL_TEXT).
6. Multi-Statute Amending Act Capability.
"""
import uuid
from io import StringIO
from datetime import date
from django.test import TestCase
from django.core.management import call_command
from django.core.management.base import CommandError

from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    StatuteAmendmentRecord,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    SourceType,
    VerificationStatus,
    ExtractionStatus,
)
from rag.source_register import (
    NIGERIAN_LEGAL_SOURCE_REGISTRY,
    MetadataVerificationLevel,
    get_statute_registry_entry,
    list_all_registry_entries,
    get_expected_amendments,
)
from rag.services.temporal_router import format_statutory_lineage_notice
from rag.services.hybrid_retriever import hybrid_legal_search


class SourceRegistryMetadataTests(TestCase):
    """Verifies that the legal source registry strictly adheres to Phase 4 rules."""

    def test_all_six_priority_instruments_present(self):
        entries = list_all_registry_entries()
        self.assertEqual(len(entries), 6)
        expected_families = {
            "CAMA-2020", "BFA-2022", "EVIDENCE-ACT-2011",
            "EVIDENCE-AMENDMENT-2023", "NDPA-2023", "FCCPA-2018"
        }
        actual_families = {e["statute_family_code"] for e in entries}
        self.assertEqual(actual_families, expected_families)

    def test_sha256_checksums_are_not_invented_or_prepopulated(self):
        """Hashes must be None until official authentic files are received."""
        for entry in list_all_registry_entries():
            self.assertIsNone(
                entry.get("sha256_checksum"),
                f"Registry entry {entry['statute_family_code']} must not have a prepopulated SHA-256 hash."
            )
            self.assertIn("integrity_verification_method", entry)

    def test_five_separate_dates_and_years_are_distinct(self):
        for entry in list_all_registry_entries():
            self.assertIsInstance(entry["title_year"], int)
            self.assertIsInstance(entry["enactment_year"], int)
            self.assertIsInstance(entry["presidential_assent_date"], date)
            if entry["gazette_publication_date"] is not None:
                self.assertIsInstance(entry["gazette_publication_date"], date)
            self.assertIsInstance(entry["commencement_date"], date)
            # Enactment year must equal or precede assent date year
            self.assertLessEqual(entry["enactment_year"], entry["presidential_assent_date"].year)

    def test_confirmed_gazette_details_level(self):
        for entry in list_all_registry_entries():
            if entry["metadata_status"] == MetadataVerificationLevel.CONFIRMED_GAZETTE_REFERENCE:
                gaz = entry["official_gazette_details"]
                self.assertTrue(gaz["gazette_number"])
                self.assertTrue(gaz["volume"])
                self.assertTrue(gaz["full_gazette_ref"])

    def test_cama_and_bfa_amendment_specifications(self):
        cama_amends = get_expected_amendments("CAMA-2020")
        self.assertEqual(len(cama_amends), 1)
        self.assertEqual(cama_amends[0]["amending_act_family"], "BFA-2022")
        self.assertIn("Section 18", cama_amends[0]["affected_sections"])

        bfa_amends = get_expected_amendments("BFA-2022")
        self.assertGreaterEqual(len(bfa_amends), 2)  # Multi-statute amending capability

    def test_reconciled_gazette_notices_and_pages(self):
        """Verifies reconciled Government Notice numbers and Gazette pagination."""
        cama = get_statute_registry_entry("CAMA-2020")
        self.assertEqual(cama["official_gazette_details"]["government_notice_number"], "Govt. Notice No. 119")
        self.assertEqual(cama["official_gazette_details"]["gazette_number"], "No. 124")
        self.assertEqual(cama["official_gazette_details"]["volume"], "Vol. 107")

        evidence = get_statute_registry_entry("EVIDENCE-ACT-2011")
        self.assertEqual(evidence["official_gazette_details"]["government_notice_number"], "Govt. Notice No. 162")
        self.assertEqual(evidence["official_gazette_details"]["gazette_number"], "No. 80")
        self.assertEqual(evidence["official_gazette_details"]["volume"], "Vol. 98")

        ndpa = get_statute_registry_entry("NDPA-2023")
        self.assertEqual(ndpa["official_gazette_details"]["government_notice_number"], "Govt. Notice No. 82")
        self.assertEqual(ndpa["official_gazette_details"]["gazette_number"], "No. 119")
        self.assertEqual(ndpa["official_gazette_details"]["page_range"], "pages A719 - A758")

        fccpa = get_statute_registry_entry("FCCPA-2018")
        self.assertEqual(fccpa["official_gazette_details"]["government_notice_number"], "Govt. Notice No. 5")
        self.assertEqual(fccpa["official_gazette_details"]["gazette_number"], "No. 18")
        self.assertEqual(fccpa["official_gazette_details"]["volume"], "Vol. 106")
        self.assertEqual(fccpa["official_gazette_details"]["cover_stated_page_range"], "pages A1 - A95")
        self.assertEqual(fccpa["official_gazette_details"]["actual_printed_page_range"], "pages A1 - A96")
        self.assertIn("A1 - A96", fccpa["official_gazette_details"]["page_range"])
        self.assertIn("Page A96", fccpa["official_gazette_details"]["assent_commencement_page"])
        self.assertEqual(fccpa["source_type"], "REGULATOR")
        self.assertEqual(fccpa["commencement_date"], date(2019, 1, 30))
        self.assertIn("30th Day of January, 2019", fccpa["commencement_provision_notes"])
        self.assertIn("pagination_discrepancy_notes", fccpa["official_gazette_details"])

        bfa = get_statute_registry_entry("BFA-2022")
        self.assertIn("Pending Direct Cover Verification", bfa["official_gazette_details"]["government_notice_number"])
        self.assertIn("Pending Direct Cover Verification", bfa["official_gazette_details"]["page_range"])
        self.assertEqual(bfa["metadata_status"], MetadataVerificationLevel.PENDING_PRINT_AUDIT)

    def test_evidence_act_commencement_and_assent_distinction(self):
        """Verifies express commencement date and distinct Clerk/President timeline."""
        evidence = get_statute_registry_entry("EVIDENCE-ACT-2011")
        self.assertEqual(evidence["commencement_date"], date(2011, 6, 3))
        self.assertEqual(evidence["presidential_assent_date"], date(2011, 6, 3))
        notes = evidence.get("assent_authentication_notes", "")
        self.assertIn("2 June 2011", notes)
        self.assertIn("3 June 2011", notes)
        self.assertIn("3rd Day of June, 2011", notes)

    def test_sha256_two_step_verification_methodology(self):
        """Ensures all 6 entries mandate two-step file integrity and page-by-page visual/textual verification."""
        for entry in list_all_registry_entries():
            method = entry["integrity_verification_method"]
            self.assertIn("SHA-256", method)
            self.assertIn("Two-step verification", method)
            self.assertIn("Page-by-page", method)
            # Must not confuse file hashing with byte comparison of physical paper
            self.assertNotIn("byte-level SHA-256 integrity comparison between NDPC official PDF and the physical Gazette",
                             entry["official_gazette_details"].get("remaining_verification_requirement", ""))

    def test_provenance_and_category_separation(self):
        """Verifies legal instrument category is distinct from digital copy provenance."""
        valid_provenances = {
            "ORIGINAL_GAZETTE_AUTHENTIC_PRINT",
            "OFFICIAL_AGENCY_HOSTED_REPRINT",
            "THIRD_PARTY_REFERENCE_COPY",
            "PENDING_PRINT_AUDIT",
        }
        for entry in list_all_registry_entries():
            self.assertEqual(entry["category"], "PRIMARY_ACT")
            self.assertIn(entry["metadata_status"].value, valid_provenances)
            for host in entry["official_gazette_details"].get("hosting_institutions", []):
                self.assertIn(host["provenance"], valid_provenances)
                self.assertTrue(host["url"].startswith("http"))


class InspectStatuteCommandTests(TestCase):
    """Verifies that inspect_statute is read-only, non-destructive, and informative."""

    def setUp(self):
        self.doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-CAMA-2020",
            title="Synthetic Companies and Allied Matters Act, 2020",
            short_title="Synthetic CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=date(2020, 8, 7),
            publisher="Federal Government Printer",
            source_type=SourceType.OFFICIAL_GAZETTE,
            official_gazette_ref="Gazette No. 124, Vol. 107",
            original_filename="synthetic_cama_2020.pdf",
            sha256_checksum="a" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
            scanned_pages=[],
            extractor_version="2.0.0",
        )
        self.chunk = LegalProvisionChunk.objects.create(
            document=self.doc,
            section_number="Section 18",
            section_title="Right to form a company",
            content="Any two or more persons may form and incorporate a company.",
            citation_label="Section 18, Synthetic CAMA 2020",
            embedding=[0.01] * 384,
            token_count=15,
            chunk_index=0,
        )

    def test_list_documents(self):
        out = StringIO()
        call_command('inspect_statute', list=True, stdout=out)
        output = out.getvalue()
        self.assertIn("Synthetic CAMA 2020", output)
        self.assertIn("PENDING_VERIFICATION", output)
        self.assertIn("COMPLETE_TEXT", output)

    def test_inspect_by_document_id(self):
        out = StringIO()
        call_command('inspect_statute', document_id=str(self.doc.id), stdout=out)
        output = out.getvalue()
        self.assertIn(str(self.doc.id), output)
        self.assertIn("Synthetic Companies and Allied Matters Act, 2020", output)
        self.assertIn("Section 18: Right to form a company", output)
        self.assertIn("SHA-256 verifies file integrity against tampering", output)

    def test_inspect_by_family_code(self):
        out = StringIO()
        call_command('inspect_statute', family_code="SYN-CAMA-2020", stdout=out)
        output = out.getvalue()
        self.assertIn("STATUTE FAMILY: SYN-CAMA-2020", output)
        self.assertIn("Synthetic CAMA 2020", output)

    def test_command_does_not_modify_database_records(self):
        initial_updated_at = self.doc.updated_at
        out = StringIO()
        call_command('inspect_statute', document_id=str(self.doc.id), stdout=out)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.updated_at, initial_updated_at)
        self.assertFalse(self.doc.is_verified_authentic)

    def test_invalid_uuid_raises_command_error(self):
        with self.assertRaises(CommandError) as ctx:
            call_command('inspect_statute', document_id="invalid-uuid")
        self.assertIn("Invalid UUID format", str(ctx.exception))


class LinkAmendmentCommandTests(TestCase):
    """Verifies link_amendment rules, metadata compatibility, and reversible unlinking."""

    def setUp(self):
        self.base_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-CAMA",
            title="Synthetic CAMA 2020",
            short_title="Synthetic CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=date(2020, 8, 7),
            publisher="Federal Government Printer",
            source_type=SourceType.OFFICIAL_GAZETTE,
            original_filename="synthetic_cama.pdf",
            sha256_checksum="b" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )
        self.chunk18 = LegalProvisionChunk.objects.create(
            document=self.base_doc,
            section_number="Section 18",
            section_title="Formation of company",
            content="Any two or more persons may form a company.",
            citation_label="Section 18, Synthetic CAMA 2020",
            embedding=[0.02] * 384,
            token_count=12,
            chunk_index=0,
        )
        self.amending_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-BFA",
            title="Synthetic Business Facilitation Act 2022",
            short_title="Synthetic BFA 2022",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=date(2023, 2, 8),
            publisher="Federal Government Printer",
            source_type=SourceType.OFFICIAL_GAZETTE,
            original_filename="synthetic_bfa.pdf",
            sha256_checksum="c" * 64,
            is_verified_authentic=False,  # Quarantined / Pending
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )

    def test_successful_linking_and_dormant_status_when_one_unverified(self):
        out = StringIO()
        call_command(
            'link_amendment',
            base_doc_id=str(self.base_doc.id),
            amending_doc_id=str(self.amending_doc.id),
            amendment_title="BFA 2022 Amendments to CAMA 2020",
            commencement_date="2023-02-08",
            affected_sections="Section 18, Section 240",
            summary="Permits single-member private companies.",
            stdout=out,
        )
        output = out.getvalue()
        self.assertIn("STATUTE AMENDMENT RELATIONSHIP CREATED", output)
        self.assertIn("QUARANTINED / DORMANT", output)

        record = StatuteAmendmentRecord.objects.get(base_statute=self.base_doc, amending_act=self.amending_doc)
        self.assertEqual(record.affected_sections, ["Section 18", "Section 240"])

        # Check provision chunk association
        self.chunk18.refresh_from_db()
        self.assertEqual(self.chunk18.amendment_record, record)

    def test_active_status_when_both_documents_are_verified(self):
        self.amending_doc.is_verified_authentic = True
        self.amending_doc.verification_status = VerificationStatus.VERIFIED
        self.amending_doc.save()

        out = StringIO()
        call_command(
            'link_amendment',
            base_doc_id=str(self.base_doc.id),
            amending_doc_id=str(self.amending_doc.id),
            amendment_title="BFA 2022 Amendments to CAMA 2020",
            commencement_date="2023-02-08",
            affected_sections="Section 18",
            summary="Permits single-member private companies.",
            stdout=out,
        )
        output = out.getvalue()
        self.assertIn("STATUS: ACTIVE IN TRUSTED RETRIEVAL", output)

    def test_disallow_self_amendment(self):
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'link_amendment',
                base_doc_id=str(self.base_doc.id),
                amending_doc_id=str(self.base_doc.id),
                amendment_title="Invalid Self Amendment",
                commencement_date="2023-02-08",
                affected_sections="Section 1",
                summary="Self amending attempt",
            )
        self.assertIn("cannot be linked as an amendment to itself", str(ctx.exception))

    def test_temporal_anomaly_rejection(self):
        """Amendment commencement date cannot precede base statute commencement date."""
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'link_amendment',
                base_doc_id=str(self.base_doc.id),
                amending_doc_id=str(self.amending_doc.id),
                amendment_title="Anomalous Amendment",
                commencement_date="2018-01-01",  # Prior to CAMA 2020 (2020-08-07)
                affected_sections="Section 18",
                summary="Temporal anomaly test.",
            )
        self.assertIn("Temporal anomaly", str(ctx.exception))

    def test_jurisdiction_conflict_rejection(self):
        state_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-LAGOS-LAW",
            title="Synthetic Lagos State Law",
            short_title="Synthetic Lagos Law",
            jurisdiction=Jurisdiction.LAGOS_STATE,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            effective_from=date(2021, 1, 1),
            publisher="Lagos State Printing Corporation",
            original_filename="lagos.pdf",
            sha256_checksum="d" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )
        with self.assertRaises(CommandError) as ctx:
            call_command(
                'link_amendment',
                base_doc_id=str(self.base_doc.id),  # Federal
                amending_doc_id=str(state_doc.id),    # Lagos State
                amendment_title="State Law Amending Federal Act",
                commencement_date="2023-01-01",
                affected_sections="Section 1",
                summary="Invalid cross-jurisdiction amendment.",
            )
        self.assertIn("Jurisdiction conflict", str(ctx.exception))

    def test_reversible_unlinking_safeguard(self):
        """Unlinking removes amendment record and resets chunks, preserving both documents."""
        record = StatuteAmendmentRecord.objects.create(
            base_statute=self.base_doc,
            amending_act=self.amending_doc,
            amendment_title="Link to Remove",
            commencement_date=date(2023, 2, 8),
            affected_sections=["Section 18"],
            summary_of_changes="Summary of test changes",
        )
        self.chunk18.amendment_record = record
        self.chunk18.save()

        out = StringIO()
        call_command('link_amendment', unlink=True, link_id=str(record.id), stdout=out)
        output = out.getvalue()

        self.assertIn("AMENDMENT RELATIONSHIP UNLINKED SUCCESSFULLY", output)
        self.assertIn("Neither statutory document was deleted", output)

        # Confirm amendment record is deleted
        self.assertFalse(StatuteAmendmentRecord.objects.filter(id=record.id).exists())

        # Confirm chunks are disassociated
        self.chunk18.refresh_from_db()
        self.assertIsNone(self.chunk18.amendment_record)

        # STRICT GUARANTEE: Both legal documents MUST still exist
        self.assertTrue(LegalSourceDocument.objects.filter(id=self.base_doc.id).exists())
        self.assertTrue(LegalSourceDocument.objects.filter(id=self.amending_doc.id).exists())

    def test_multi_statute_amending_act_support(self):
        second_base_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-IMMIGRATION",
            title="Synthetic Immigration Act 2015",
            short_title="Synthetic Immigration Act",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            effective_from=date(2015, 5, 25),
            publisher="Federal Government Printer",
            original_filename="immigration.pdf",
            sha256_checksum="e" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )

        # Link 1: BFA -> CAMA
        rec1 = StatuteAmendmentRecord.objects.create(
            base_statute=self.base_doc,
            amending_act=self.amending_doc,
            amendment_title="BFA Amendments to CAMA",
            commencement_date=date(2023, 2, 8),
            affected_sections=["Section 18"],
            summary_of_changes="CAMA modifications.",
        )

        # Link 2: BFA -> Immigration Act
        rec2 = StatuteAmendmentRecord.objects.create(
            base_statute=second_base_doc,
            amending_act=self.amending_doc,
            amendment_title="BFA Amendments to Immigration Act",
            commencement_date=date(2023, 2, 8),
            affected_sections=["Section 20"],
            summary_of_changes="Visa facilitation.",
        )

        # Verify amending_doc has 2 distinct amendment records
        self.assertEqual(self.amending_doc.amendments_made.count(), 2)
        base_statute_ids = set(self.amending_doc.amendments_made.values_list('base_statute_id', flat=True))
        self.assertEqual(base_statute_ids, {self.base_doc.id, second_base_doc.id})


class TwoWayVerificationEnforcementTests(TestCase):
    """Verifies that pending amendment relationships CANNOT influence retrieval or lineage notices."""

    def setUp(self):
        self.base_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-CAMA-VER",
            title="Synthetic Verified CAMA 2020",
            short_title="Synthetic CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=date(2020, 8, 7),
            publisher="Federal Government Printer",
            original_filename="cama.pdf",
            sha256_checksum="f" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )
        self.chunk = LegalProvisionChunk.objects.create(
            document=self.base_doc,
            section_number="Section 18",
            section_title="Formation of company",
            content="Any two or more persons may form and incorporate a company.",
            citation_label="Section 18, Synthetic CAMA 2020",
            embedding=[0.05] * 384,
            token_count=15,
            chunk_index=0,
        )
        self.amending_doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-BFA-PEND",
            title="Synthetic Unverified BFA 2022",
            short_title="Synthetic BFA 2022",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=date(2023, 2, 8),
            publisher="Federal Government Printer",
            original_filename="bfa.pdf",
            sha256_checksum="1" * 64,
            is_verified_authentic=False,  # PENDING VERIFICATION!
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
        )
        self.amend_rec = StatuteAmendmentRecord.objects.create(
            base_statute=self.base_doc,
            amending_act=self.amending_doc,
            amendment_title="BFA Amendments to CAMA",
            commencement_date=date(2023, 2, 8),
            affected_sections=["Section 18"],
            summary_of_changes="Introduced single-member company provisions.",
        )
        self.chunk.amendment_record = self.amend_rec
        self.chunk.save()

    def test_lineage_notice_omitted_when_amending_act_is_pending_verification(self):
        """When amending Act is NOT verified authentic, amendment notice MUST NOT be generated."""
        notice = format_statutory_lineage_notice(self.chunk)
        self.assertFalse(notice['is_amended'])
        self.assertEqual(notice['provision_nature'], 'ORIGINAL')
        self.assertEqual(notice['lineage_text'], "")
        self.assertIsNone(notice['amending_act_title'])

    def test_lineage_notice_active_only_when_both_instruments_verified(self):
        """When amending Act is verified, amendment notice becomes active."""
        self.amending_doc.is_verified_authentic = True
        self.amending_doc.verification_status = VerificationStatus.VERIFIED
        self.amending_doc.save()

        # Reload chunk to clear cached relations
        chunk = LegalProvisionChunk.objects.select_related(
            'document', 'amendment_record__base_statute', 'amendment_record__amending_act'
        ).get(id=self.chunk.id)

        notice = format_statutory_lineage_notice(chunk)
        self.assertTrue(notice['is_amended'])
        self.assertEqual(notice['provision_nature'], 'AMENDED')
        self.assertEqual(notice['amending_act_title'], 'Synthetic BFA 2022')
        self.assertIn("AMENDMENT NOTICE: Section 18 was amended by Synthetic BFA 2022", notice['lineage_text'])


class IncompleteExtractionQuarantineTests(TestCase):
    """Verifies verify_statute refuses to verify incomplete or scanned-page documents."""

    def test_rejects_verification_of_needs_ocr_document(self):
        doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-OCR",
            title="Scanned Document Needing OCR",
            short_title="Scanned Doc",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2020, 1, 1),
            publisher="Federal Government Printer",
            original_filename="scanned.pdf",
            sha256_checksum="2" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.NEEDS_OCR,
            scanned_pages=[1, 2, 3],
        )

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Legal Counsel Esq.",
                reason="Attempting to verify scanned document.",
            )
        self.assertIn("not 'COMPLETE_TEXT'", str(ctx.exception))
        doc.refresh_from_db()
        self.assertFalse(doc.is_verified_authentic)

    def test_rejects_verification_of_mixed_partial_text_document(self):
        doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-MIXED",
            title="Partially Scanned Document",
            short_title="Mixed Doc",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2020, 1, 1),
            publisher="Federal Government Printer",
            original_filename="mixed.pdf",
            sha256_checksum="3" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.MIXED_PARTIAL_TEXT,
            scanned_pages=[10],
        )

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Legal Counsel Esq.",
                reason="Attempting to verify mixed partial document.",
            )
        self.assertIn("not 'COMPLETE_TEXT'", str(ctx.exception))
        doc.refresh_from_db()
        self.assertFalse(doc.is_verified_authentic)

    def test_rejects_verification_if_scanned_pages_list_is_not_empty(self):
        doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-PAGES",
            title="Document With Unextracted Scanned Pages",
            short_title="Pages Doc",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2020, 1, 1),
            publisher="Federal Government Printer",
            original_filename="pages.pdf",
            sha256_checksum="4" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,  # Misclassified status
            scanned_pages=[4, 5],  # But scanned pages exist!
        )

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Legal Counsel Esq.",
                reason="Attempting to verify document with scanned pages.",
            )
        self.assertIn("scanned page(s)", str(ctx.exception))
        doc.refresh_from_db()
        self.assertFalse(doc.is_verified_authentic)

    def test_complete_text_verification_succeeds(self):
        doc = LegalSourceDocument.objects.create(
            statute_family_code="SYN-COMPLETE",
            title="Complete Digital Text Document",
            short_title="Complete Doc",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2020, 1, 1),
            publisher="Federal Government Printer",
            original_filename="complete.pdf",
            sha256_checksum="5" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
            scanned_pages=[],
        )

        out = StringIO()
        call_command(
            'verify_statute',
            document_id=str(doc.id),
            counsel_name="Senior Advocate of Nigeria",
            reason="Verified against Federal Republic of Nigeria Official Gazette No. 12.",
            stdout=out,
        )
        self.assertIn("DOCUMENT VERIFIED AUTHENTIC", out.getvalue())
        doc.refresh_from_db()
        self.assertTrue(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(doc.verified_by, "Senior Advocate of Nigeria")

    def _create_mock_fccpa_doc(self):
        """Helper to create a synthetic FCCPA document for structural testing."""
        return LegalSourceDocument.objects.create(
            statute_family_code="FCCPA-2018",
            title="Federal Competition and Consumer Protection Act, 2018",
            short_title="FCCPA 2018",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2019, 1, 30),
            publisher="Federal Government Printer",
            source_type=SourceType.REGULATOR,
            original_filename="fccpc_fccpa_2018.pdf",
            sha256_checksum="6" * 64,
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            extraction_status=ExtractionStatus.COMPLETE_TEXT,
            scanned_pages=[],
        )

    def _populate_fccpa_chunks(self, doc, omit_sections=None, omit_parts=None, omit_schedules=None, duplicate_section=None):
        """Helper to populate mock chunks across all 18 Parts, 168 sections, and 2 Schedules."""
        omit_sections = set(omit_sections or [])
        omit_parts = set(omit_parts or [])
        omit_schedules = set(omit_schedules or [])

        part_ranges = [
            ("Part I", 1, 2), ("Part II", 3, 16), ("Part III", 17, 26), ("Part IV", 27, 38),
            ("Part V", 39, 58), ("Part VI", 59, 69), ("Part VII", 70, 79), ("Part VIII", 80, 91),
            ("Part IX", 92, 106), ("Part X", 107, 113), ("Part XI", 114, 133), ("Part XII", 134, 145),
            ("Part XIII", 146, 150), ("Part XIV", 151, 152), ("Part XV", 153, 153),
            ("Part XVI", 154, 155), ("Part XVII", 156, 156), ("Part XVIII", 157, 168)
        ]
        chunks_to_create = []
        chunk_idx = 0
        for part_name, start_sec, end_sec in part_ranges:
            is_omitted = part_name in omit_parts
            part_val = "" if is_omitted else part_name
            for sec_num in range(start_sec, end_sec + 1):
                sec_name = f"Section {sec_num}"
                if sec_name in omit_sections:
                    continue
                part_mention = "generic_part" if is_omitted else part_name
                chunks_to_create.append(LegalProvisionChunk(
                    document=doc,
                    part_number=part_val,
                    section_number=sec_name,
                    section_title=f"Provisions of {sec_name}",
                    content=f"Statutory text of {sec_name} in {part_mention}.",
                    citation_label=f"{sec_name}, FCCPA 2018",
                    embedding=[0.01] * 384,
                    token_count=15,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
                if duplicate_section and sec_name == duplicate_section:
                    chunks_to_create.append(LegalProvisionChunk(
                        document=doc,
                        part_number=part_val,
                        section_number=sec_name,
                        section_title=f"Duplicate provisions of {sec_name}",
                        content=f"Statutory text of {sec_name} in {part_mention}.",  # Identical content to test duplicate extraction
                        citation_label=f"{sec_name}, FCCPA 2018 (duplicate)",
                        embedding=[0.01] * 384,
                        token_count=15,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1

        if "First Schedule" not in omit_schedules:
            chunks_to_create.append(LegalProvisionChunk(
                document=doc,
                part_number="Schedules",
                section_number="First Schedule",
                section_title="Professional Associations",
                content="List of professional associations under Section 167(14).",
                citation_label="First Schedule, FCCPA 2018",
                embedding=[0.01] * 384,
                token_count=12,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

        if "Second Schedule" not in omit_schedules:
            chunks_to_create.append(LegalProvisionChunk(
                document=doc,
                part_number="Schedules",
                section_number="Second Schedule",
                section_title="Transitional Provisions and Supplementary Matters",
                content="Transitional provisions and transfer of assets from Consumer Protection Council.",
                citation_label="Second Schedule, FCCPA 2018",
                embedding=[0.01] * 384,
                token_count=15,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

        LegalProvisionChunk.objects.bulk_create(chunks_to_create)

    def test_exact_section_normalization_never_substring_matches(self):
        """Verifies Section 1 never falsely matches Section 10 or Section 100 via substring."""
        from rag.services.statutory_structure_validator import (
            normalize_section_identifier,
            validate_statutory_completeness,
        )
        # Normalization unit check
        self.assertEqual(normalize_section_identifier("Section 1"), "Section 1")
        self.assertEqual(normalize_section_identifier("Section 10"), "Section 10")
        self.assertEqual(normalize_section_identifier("Section 100"), "Section 100")
        self.assertNotEqual(normalize_section_identifier("Section 1"), normalize_section_identifier("Section 10"))
        self.assertNotEqual(normalize_section_identifier("Section 1"), normalize_section_identifier("Section 100"))

        doc = self._create_mock_fccpa_doc()
        # Omit Section 1, but include Section 10 and Section 100
        self._populate_fccpa_chunks(doc, omit_sections=["Section 1"])

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertIn("Section 1", report.missing_sections)
        self.assertIn("Section 10", report.found_sections)
        self.assertIn("Section 100", report.found_sections)

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Bar Council Auditor",
                reason="Attempting to verify FCCPA document missing Section 1.",
            )
        self.assertIn("Section 1", str(ctx.exception))
        self.assertIn("failed statutory structural completeness audit", str(ctx.exception))

    def test_detects_missing_intermediate_section(self):
        """Verifies detection of an intermediate section omitted from extraction (Section 42)."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Omit intermediate Section 42
        self._populate_fccpa_chunks(doc, omit_sections=["Section 42"])

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertIn("Section 42", report.missing_sections)

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Auditing extraction with missing Section 42.",
            )
        self.assertIn("Section 42", str(ctx.exception))

    def test_detects_missing_part(self):
        """Verifies detection when a statutory Part (Part V) is omitted from extraction."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Omit Part V
        self._populate_fccpa_chunks(doc, omit_parts=["Part V"])

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertIn("Part V", report.missing_parts)

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Auditing extraction with missing Part V.",
            )
        self.assertIn("Part V", str(ctx.exception))

    def test_detects_missing_schedule(self):
        """Verifies independent detection of missing First or Second Schedule."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Omit Second Schedule
        self._populate_fccpa_chunks(doc, omit_schedules=["Second Schedule"])

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertIn("Second Schedule", report.missing_schedules)
        self.assertIn("First Schedule", report.found_schedules)

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Auditing extraction with missing Second Schedule.",
            )
        self.assertIn("Second Schedule", str(ctx.exception))

    def test_detects_duplicate_section_identifiers(self):
        """Verifies detection and rejection of duplicate section chunks (two Section 15 chunks)."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Duplicate Section 15
        self._populate_fccpa_chunks(doc, duplicate_section="Section 15")

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(any("Section 15" in d for d in report.duplicate_sections))

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Auditing extraction containing duplicate Section 15.",
            )
        self.assertIn("Section 15", str(ctx.exception))
        self.assertIn("Duplicate", str(ctx.exception))

    def test_complete_statutory_extraction_passes_structural_completeness(self):
        """Verifies a 100% complete FCCPA extraction passes structural check and can be verified by counsel."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Complete: all 168 sections, all 18 Parts, First Schedule, Second Schedule
        self._populate_fccpa_chunks(doc)

        report = validate_statutory_completeness(doc)
        self.assertTrue(report.is_structurally_complete)
        self.assertEqual(len(report.missing_sections), 0)
        self.assertEqual(len(report.missing_parts), 0)
        self.assertEqual(len(report.missing_schedules), 0)
        self.assertEqual(len(report.duplicate_sections), 0)
        self.assertEqual(len(report.found_sections), 168)
        self.assertEqual(len(report.found_parts), 18)
        self.assertEqual(len(report.found_schedules), 2)

        # Document is NOT verified yet (structural completeness is separate from legal authenticity)
        doc.refresh_from_db()
        self.assertFalse(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.PENDING)

        # Now counsel authorizes verification
        out = StringIO()
        call_command(
            'verify_statute',
            document_id=str(doc.id),
            counsel_name="Senior Advocate of Nigeria",
            reason="Verified complete authentic text across all 18 Parts, 168 sections, and 2 Schedules against Gazette No. 18.",
            stdout=out,
        )
        self.assertIn("DOCUMENT VERIFIED AUTHENTIC", out.getvalue())
        doc.refresh_from_db()
        self.assertTrue(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(doc.verified_by, "Senior Advocate of Nigeria")

    def test_detects_out_of_bounds_section_zero(self):
        """Verifies Section 0 is detected as out-of-bounds (section numbers must be >= 1)."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        self._populate_fccpa_chunks(doc)
        # Add an invalid Section 0 chunk
        LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section 0",
            section_title="Preamble / Invalid Section 0",
            content="Invalid statutory section number zero.",
            citation_label="Section 0, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=10,
            chunk_index=999,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(any("Section 0" in m and "out-of-bounds" in m for m in report.malformed_sections))

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Testing Section 0 boundary check.",
            )
        self.assertIn("Section 0", str(ctx.exception))

    def test_detects_out_of_bounds_section_169(self):
        """Verifies Section 169 is detected as out-of-bounds for FCCPA (max 168 sections)."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        self._populate_fccpa_chunks(doc)
        # Add an invalid Section 169 chunk
        LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section 169",
            section_title="Phantom Extra Section",
            content="Statutory provision exceeding maximum 168 sections.",
            citation_label="Section 169, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=10,
            chunk_index=1000,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(any("Section 169" in m and "exceeds statute maximum" in m for m in report.malformed_sections))

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Testing Section 169 boundary check.",
            )
        self.assertIn("Section 169", str(ctx.exception))

    def test_detects_malformed_section_identifiers(self):
        """Verifies malformed or negative section identifiers (e.g. 'Section ???', 'Section -5') are rejected."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        self._populate_fccpa_chunks(doc)
        # Add malformed chunks
        LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section ???",
            section_title="Corrupt Section Identifier",
            content="Text with corrupt section identifier.",
            citation_label="Section ???, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=10,
            chunk_index=1001,
        )
        LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section -5",
            section_title="Negative Section Identifier",
            content="Text with negative section number.",
            citation_label="Section -5, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=10,
            chunk_index=1002,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(any("Section ???" in m for m in report.malformed_sections))
        self.assertTrue(any("Section -5" in m for m in report.malformed_sections))

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Testing malformed section identifiers.",
            )
        self.assertIn("failed statutory structural completeness audit", str(ctx.exception))

    def test_multiple_legitimate_chunks_belonging_to_one_section_accepted(self):
        """Verifies a long section split into multiple sequential chunks with distinct text is accepted."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Populate all sections EXCEPT Section 18
        self._populate_fccpa_chunks(doc, omit_sections=["Section 18"])

        # Add Section 18 as two legitimate sequential chunks (e.g. paragraphs a-f, then paragraphs g-m)
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18",
            section_title="Staff of the Commission",
            content="Part 1: The Commission may appoint such other staff as it considers necessary.",
            citation_label="Section 18, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=15,
            chunk_index=50,
        )
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18",
            section_title="Staff of the Commission",
            content="Part 2: The terms and conditions of service shall be determined by the Commission.",
            citation_label="Section 18, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=16,
            chunk_index=51,
        )

        report = validate_statutory_completeness(doc)
        self.assertTrue(report.is_structurally_complete)
        self.assertEqual(len(report.duplicate_sections), 0)
        self.assertIn("Section 18", report.found_sections)
        self.assertEqual(len(report.found_sections), 168)

    def test_different_subsections_belonging_to_one_section_accepted(self):
        """Verifies different subsections of one section (e.g. Section 18(1) and Section 18(2)) are accepted."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Populate all sections EXCEPT Section 18
        self._populate_fccpa_chunks(doc, omit_sections=["Section 18"])

        # Add Section 18 as distinct subsections: Section 18(1) and Section 18(2)
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18(1)",
            section_title="Staff appointments",
            content="Subsection (1): The Commission shall employ such persons as necessary.",
            citation_label="Section 18(1), FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=14,
            chunk_index=52,
        )
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18(2)",
            section_title="Remuneration of staff",
            content="Subsection (2): The terms of employment shall be as approved by the Board.",
            citation_label="Section 18(2), FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=16,
            chunk_index=53,
        )

        report = validate_statutory_completeness(doc)
        self.assertTrue(report.is_structurally_complete)
        self.assertEqual(len(report.duplicate_sections), 0)
        self.assertIn("Section 18", report.found_sections)
        self.assertEqual(len(report.found_sections), 168)

    def test_prose_mention_of_part_or_schedule_does_not_satisfy_audit(self):
        """Verifies mentioning Part V or First Schedule inside ordinary body prose does not satisfy the audit."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Omit Part V and First Schedule from structural extraction
        self._populate_fccpa_chunks(doc, omit_parts=["Part V"], omit_schedules=["First Schedule"])

        # Create a chunk with ordinary prose mentioning Part V and First Schedule
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 30",
            section_title="Incidental Powers",
            content=(
                "Subject to the provisions of Part V of this Act, the Commission may enter into contracts "
                "with any professional body specified in the First Schedule to this Act."
            ),
            citation_label="Section 30, FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=25,
            chunk_index=200,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        # Assert that inline prose mention was NOT accepted as fulfilling Part V or First Schedule
        self.assertIn("Part V", report.missing_parts)
        self.assertIn("First Schedule", report.missing_schedules)

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Verifying prose mention rejection.",
            )
        self.assertIn("Part V", str(ctx.exception))

    def test_detects_partially_overlapping_duplicate_chunks_with_same_identifier(self):
        """Verifies partially overlapping chunks sharing the same provision identifier are detected and rejected.

        Confirms that having different SHA-256 fingerprints is NOT treated as proof of legitimacy
        when chunks have substantial text overlap.
        """
        import hashlib
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        # Populate all sections EXCEPT Section 18
        self._populate_fccpa_chunks(doc, omit_sections=["Section 18"])

        # Add two chunks for Section 18 that have DIFFERENT SHA-256 fingerprints,
        # but have high partial overlap (e.g. Chunk B contains Chunk A plus additional clause)
        chunk_a_text = (
            "The Commission may appoint such other staff as it considers necessary "
            "for the efficient performance of its statutory functions under this Act."
        )
        chunk_b_text = (
            "The Commission may appoint such other staff as it considers necessary "
            "for the efficient performance of its statutory functions under this Act, "
            "and shall determine their remuneration and conditions of service."
        )

        # Confirm SHA-256 fingerprints are strictly different
        fp_a = hashlib.sha256(chunk_a_text.lower().encode('utf-8')).hexdigest()
        fp_b = hashlib.sha256(chunk_b_text.lower().encode('utf-8')).hexdigest()
        self.assertNotEqual(fp_a, fp_b)

        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18",
            section_title="Staff of the Commission",
            content=chunk_a_text,
            citation_label="Section 18, FCCPA 2018 (chunk 1)",
            embedding=[0.01] * 384,
            token_count=18,
            chunk_index=50,
        )
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18",
            section_title="Staff of the Commission",
            content=chunk_b_text,
            citation_label="Section 18, FCCPA 2018 (chunk 2)",
            embedding=[0.01] * 384,
            token_count=26,
            chunk_index=51,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(
            any("Partially overlapping duplicate chunks detected: Section 18" in d for d in report.duplicate_sections)
        )

        with self.assertRaises(CommandError) as ctx:
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Senior Counsel",
                reason="Verifying rejection of partially overlapping duplicate chunks.",
            )
        self.assertIn("Partially overlapping duplicate chunks detected", str(ctx.exception))

    def test_detects_overlapping_content_across_subsections(self):
        """Verifies that if two different subsections share identical or substantially overlapping text, it is flagged."""
        from rag.services.statutory_structure_validator import validate_statutory_completeness

        doc = self._create_mock_fccpa_doc()
        self._populate_fccpa_chunks(doc, omit_sections=["Section 18"])

        # Create Section 18(1) and Section 18(2) with identical text (mis-extracted duplicate)
        shared_text = "The Commission shall employ such persons as necessary for its operations."
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18(1)",
            section_title="Staff of the Commission",
            content=shared_text,
            citation_label="Section 18(1), FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=12,
            chunk_index=52,
        )
        LegalProvisionChunk.objects.create(
            document=doc,
            part_number="Part III",
            section_number="Section 18(2)",
            section_title="Staff of the Commission",
            content=shared_text,
            citation_label="Section 18(2), FCCPA 2018",
            embedding=[0.01] * 384,
            token_count=12,
            chunk_index=53,
        )

        report = validate_statutory_completeness(doc)
        self.assertFalse(report.is_structurally_complete)
        self.assertTrue(
            any("Duplicate content across provisions: Section 18(1) and Section 18(2)" in d for d in report.duplicate_sections)
        )
