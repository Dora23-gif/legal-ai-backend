import datetime
import hashlib
import unittest
from django.test import TestCase
from django.db import IntegrityError, connection, models
from rag.models import (
    LegalSourceDocument,
    StatuteAmendmentRecord,
    LegalProvisionChunk,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    SourceType,
)


class LegalKnowledgeModelTests(TestCase):
    """
    Comprehensive test suite verifying Phase 1 legal knowledge base:
    - Provenance & SHA-256 integrity
    - Current-version uniqueness per statute family
    - Date validity and exclusion constraints
    - Amendment lineage tracking
    - Database-level trigger full-text search vector sync
    - 384-dimensional vector storage & cosine distance
    - Multi-tier retrieval filtering (Current vs Historical/Repealed)
    - Live on-disk storage measurements
    """

    def setUp(self):
        sample_bytes = b"Official Gazette of the Federal Republic of Nigeria No. 124"
        self.sample_sha256 = hashlib.sha256(sample_bytes).hexdigest()
        self.mock_embedding_1 = [0.05] * 384
        self.mock_embedding_2 = [-0.05] * 384

    def test_01_document_creation_and_integrity_checksum(self):
        """Verifies authentic statute document creation with verifiable provenance."""
        doc = LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            is_current_version=True,
            effective_from=datetime.date(2020, 8, 7),
            effective_to=None,
            status=DocumentStatus.ACTIVE,
            publisher="Federal Government Printing Press",
            source_type=SourceType.OFFICIAL_GAZETTE,
            official_gazette_ref="Official Gazette No. 124, Vol. 107",
            source_url="https://cac.gov.ng/cama-2020.pdf",
            original_filename="CAMA_2020_Gazette.pdf",
            sha256_checksum=self.sample_sha256,
            verified_by="Legal Counsel Team",
            verified_at=datetime.datetime.now(datetime.timezone.utc),
        )

        self.assertIsNotNone(doc.id)
        self.assertEqual(len(doc.sha256_checksum), 64)
        self.assertTrue(doc.is_current_version)
        self.assertIn("ACTIVE", str(doc))

    def test_02_unique_current_version_per_statute_family(self):
        """Rejects conflicting current versions within the same statute family and jurisdiction."""
        LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="CAMA 2020 Original",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2020, 8, 7),
            sha256_checksum=self.sample_sha256,
            original_filename="cama_2020.pdf",
        )

        # Attempting to insert another 'is_current_version=True' for FED_CAMA in FEDERAL must fail
        with self.assertRaises(IntegrityError):
            LegalSourceDocument.objects.create(
                statute_family_code="FED_CAMA",
                title="CAMA 2026 Proposed",
                short_title="CAMA 2026",
                jurisdiction=Jurisdiction.FEDERAL,
                is_current_version=True,
                effective_from=datetime.date(2026, 1, 1),
                sha256_checksum=self.sample_sha256,
                original_filename="cama_2026.pdf",
            )

    def test_03_effective_date_order_constraint(self):
        """Rejects documents where effective_to is earlier than effective_from."""
        with self.assertRaises(IntegrityError):
            LegalSourceDocument.objects.create(
                statute_family_code="FED_INVALID_DATES",
                title="Invalid Date Law",
                short_title="IDL 2022",
                jurisdiction=Jurisdiction.FEDERAL,
                is_current_version=False,
                effective_from=datetime.date(2022, 1, 1),
                effective_to=datetime.date(2020, 1, 1),
                sha256_checksum=self.sample_sha256,
                original_filename="invalid.pdf",
            )

    def test_04_statute_amendment_record_and_chunk_linkage(self):
        """Verifies amendment relationships and precise citation generation."""
        cama_2020 = LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2020, 8, 7),
            status=DocumentStatus.AMENDED,
            sha256_checksum=self.sample_sha256,
            original_filename="cama_2020.pdf",
        )

        finance_act_2023 = LegalSourceDocument.objects.create(
            statute_family_code="FED_FINANCE_ACT",
            title="Finance Act, 2023",
            short_title="Finance Act 2023",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2023, 5, 28),
            status=DocumentStatus.ACTIVE,
            sha256_checksum=self.sample_sha256,
            original_filename="finance_act_2023.pdf",
        )

        amendment = StatuteAmendmentRecord.objects.create(
            base_statute=cama_2020,
            amending_act=finance_act_2023,
            amendment_title="Finance Act 2023 CAMA Amendments",
            commencement_date=datetime.date(2023, 5, 28),
            affected_sections=["Section 18"],
            summary_of_changes="Modified company share capital and electronic filings.",
            official_gazette_ref="Gazette No. 89, Vol. 110",
        )

        chunk = LegalProvisionChunk.objects.create(
            document=cama_2020,
            amendment_record=amendment,
            section_number="Section 18",
            section_title="Right to form a company",
            content="As amended by the Finance Act 2023, any two or more persons may form a company...",
            citation_label="Section 18, CAMA 2020 (as amended by Finance Act 2023)",
            embedding=self.mock_embedding_1,
            chunk_index=1,
        )

        self.assertIsNotNone(chunk.amendment_record)
        self.assertEqual(chunk.amendment_record.amending_act.short_title, "Finance Act 2023")
        self.assertIn("Finance Act 2023", chunk.citation_label)

    @unittest.skipUnless(connection.vendor == 'postgresql', "Requires PostgreSQL engine for triggers")
    def test_05_postgresql_trigger_search_vector_automatic_sync(self):
        """Verifies that the PostgreSQL database trigger automatically populates and updates search_vector."""
        from django.contrib.postgres.search import SearchQuery

        doc = LegalSourceDocument.objects.create(
            statute_family_code="RIV_TENANCY",
            title="Rivers State Recovery of Premises Law",
            short_title="Rivers Recovery of Premises Law",
            jurisdiction=Jurisdiction.RIVERS_STATE,
            is_current_version=True,
            effective_from=datetime.date(1999, 5, 29),
            sha256_checksum=self.sample_sha256,
            original_filename="rivers_premises.pdf",
        )

        chunk = LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section 16",
            section_title="Summary Recovery of Possession Prohibited",
            content="No landlord shall take possession of any premises without legal process and an order of court.",
            citation_label="Section 16, Rivers Recovery of Premises Law",
            embedding=self.mock_embedding_1,
            chunk_index=16,
        )

        chunk.refresh_from_db()
        self.assertIsNotNone(chunk.search_vector)

        matches = LegalProvisionChunk.objects.filter(search_vector=SearchQuery("possession"))
        self.assertTrue(matches.filter(id=chunk.id).exists())

        LegalProvisionChunk.objects.filter(id=chunk.id).update(
            content="No landlord shall unlawfully lock out or forcibly eject a tenant."
        )

        updated_matches = LegalProvisionChunk.objects.filter(search_vector=SearchQuery("eject"))
        self.assertTrue(updated_matches.filter(id=chunk.id).exists())

    @unittest.skipUnless(connection.vendor == 'postgresql', "Requires PostgreSQL pgvector extension")
    def test_06_vector_embedding_cosine_distance_query(self):
        """Verifies pgvector stores 384-dim embeddings and performs cosine distance queries."""
        from pgvector.django import CosineDistance

        doc = LegalSourceDocument.objects.create(
            statute_family_code="FED_EVIDENCE",
            title="Evidence Act, 2011",
            short_title="Evidence Act 2011",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2011, 6, 3),
            sha256_checksum=self.sample_sha256,
            original_filename="evidence_act_2011.pdf",
        )

        chunk1 = LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section 84",
            section_title="Admissibility of statements in documents produced by computers",
            content="In any proceeding a statement contained in a document produced by a computer shall be admissible...",
            embedding=self.mock_embedding_1,
            chunk_index=84,
        )

        chunk2 = LegalProvisionChunk.objects.create(
            document=doc,
            section_number="Section 84A",
            section_title="Admissibility of electronic records and digital signatures",
            content="Information in electronic form shall be admissible as evidence...",
            embedding=self.mock_embedding_2,
            chunk_index=85,
        )

        query_vector = [0.05] * 384
        results = LegalProvisionChunk.objects.annotate(
            distance=CosineDistance("embedding", query_vector)
        ).order_by("distance")

        self.assertEqual(results.first().id, chunk1.id)
        self.assertAlmostEqual(results.first().distance, 0.0, places=4)

    def test_07_current_law_vs_historical_repealed_filtering(self):
        """Verifies that current-law queries exclude repealed statutes and historical queries locate the correct era."""
        cama_1990 = LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="Companies and Allied Matters Act, 1990",
            short_title="CAMA 1990",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=False,
            effective_from=datetime.date(1990, 1, 2),
            effective_to=datetime.date(2020, 8, 7),
            status=DocumentStatus.REPEALED,
            sha256_checksum=self.sample_sha256,
            original_filename="cama_1990.pdf",
        )

        cama_2020 = LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2020, 8, 7),
            effective_to=None,
            status=DocumentStatus.ACTIVE,
            superseded_by=None,
            sha256_checksum=self.sample_sha256,
            original_filename="cama_2020.pdf",
        )

        cama_1990.superseded_by = cama_2020
        cama_1990.save()

        chunk_1990 = LegalProvisionChunk.objects.create(
            document=cama_1990,
            section_number="Section 18",
            section_title="Right to form a company",
            content="Under CAMA 1990, two or more persons may form an incorporated company.",
            citation_label="Section 18, CAMA 1990",
            embedding=self.mock_embedding_1,
            chunk_index=1,
        )

        chunk_2020 = LegalProvisionChunk.objects.create(
            document=cama_2020,
            section_number="Section 18",
            section_title="Right to form a company",
            content="Under CAMA 2020, one person or two or more persons may form a company.",
            citation_label="Section 18, CAMA 2020",
            embedding=self.mock_embedding_2,
            chunk_index=1,
        )

        today = datetime.date.today()

        # 1. Current-law query: MUST return only CAMA 2020, NEVER CAMA 1990
        current_results = LegalProvisionChunk.objects.filter(
            document__statute_family_code="FED_CAMA",
            document__jurisdiction=Jurisdiction.FEDERAL,
            document__is_current_version=True,
            document__status__in=[DocumentStatus.ACTIVE, DocumentStatus.AMENDED],
            document__effective_from__lte=today,
        ).filter(
            models.Q(document__effective_to__isnull=True) |
            models.Q(document__effective_to__gt=today)
        )

        self.assertEqual(current_results.count(), 1)
        self.assertEqual(current_results.first().id, chunk_2020.id)

        # 2. Historical query for dispute in 2015: MUST return CAMA 1990
        dispute_date = datetime.date(2015, 6, 1)
        historical_2015 = LegalProvisionChunk.objects.filter(
            document__statute_family_code="FED_CAMA",
            document__jurisdiction=Jurisdiction.FEDERAL,
            document__effective_from__lte=dispute_date,
        ).filter(
            models.Q(document__effective_to__isnull=True) |
            models.Q(document__effective_to__gt=dispute_date)
        )

        self.assertEqual(historical_2015.count(), 1)
        self.assertEqual(historical_2015.first().id, chunk_1990.id)
        self.assertEqual(historical_2015.first().document.status, DocumentStatus.REPEALED)

    @unittest.skipUnless(connection.vendor == 'postgresql', "Requires live PostgreSQL Neon database")
    def test_08_live_storage_benchmark_metrics(self):
        """Inspects live PostgreSQL disk allocation metrics on Neon for rag_legal_provision_chunk."""
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    'rag_legal_provision_chunk' AS table_name,
                    pg_size_pretty(pg_total_relation_size('rag_legal_provision_chunk')) AS total_size,
                    pg_size_pretty(pg_relation_size('rag_legal_provision_chunk')) AS table_heap_size,
                    pg_size_pretty(pg_indexes_size('rag_legal_provision_chunk')) AS indexes_size;
            """)
            row = cursor.fetchone()

        self.assertIsNotNone(row)
        print(f"\n[NEON DISK BENCHMARK] Table: {row[0]} | Total Size: {row[1]} | Heap Size: {row[2]} | Indexes: {row[3]}")
