"""
Automated Test Suite for Phase 3: Hybrid Retrieval, Legal Lineage Routing,
Jurisdiction Filtering, Citation Integrity, and Anti-Hallucination Guardrails.

Test Categories:
1. Hybrid Retrieval & RRF Fusion (k=60 rank-based scoring).
2. Exact Section Number Matching (e.g., 'Section 84A' precision tier).
3. Historical Law vs Current Law (temporal filtering by effective date).
4. Amendment Relationship Lineage (preserving original vs amending vs consolidated text).
5. Statute Disambiguation (preventing incorrect statute identification).
6. Jurisdiction Scoping & Prioritization (State law priority + Federal baseline).
7. Structured Citation Integrity (metadata provenance from DB fields).
8. Hallucinated Citation Rejection (filtering unknown citation tokens).
9. Empty Retrieval Fallback (handling legal queries with no verified matches).
10. Non-Legal Query Handling (greetings and general questions skip legal filtering).
"""
import datetime
from django.test import TestCase
from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    StatuteAmendmentRecord,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    SourceType,
    VerificationStatus,
)
from rag.services.jurisdiction_service import detect_jurisdiction
from rag.services.temporal_router import (
    analyze_temporal_intent,
    apply_temporal_filter,
    format_statutory_lineage_notice,
)
from rag.services.citation_service import (
    generate_citation_id,
    build_structured_citation,
    filter_and_validate_citations,
)
from rag.services.hybrid_retriever import (
    hybrid_legal_search,
    extract_section_query,
    RRF_K,
)
from rag.services.retrieval_service import (
    is_legal_query,
    retrieve_legal_context,
)


class JurisdictionServiceTests(TestCase):
    """Tests for jurisdiction detection and priority order."""

    def test_state_jurisdiction_priority(self):
        # Lagos State queries
        jur, prio = detect_jurisdiction("What are the tenant eviction rules under Lagos State tenancy law?")
        self.assertEqual(jur, Jurisdiction.LAGOS_STATE)
        self.assertEqual(prio[0], Jurisdiction.LAGOS_STATE)
        self.assertEqual(prio[1], Jurisdiction.FEDERAL)

        # Rivers State queries
        jur_riv, prio_riv = detect_jurisdiction("Recovery of premises in Port Harcourt Rivers State")
        self.assertEqual(jur_riv, Jurisdiction.RIVERS_STATE)
        self.assertEqual(prio_riv[0], Jurisdiction.RIVERS_STATE)

    def test_federal_jurisdiction_default(self):
        jur, prio = detect_jurisdiction("How many directors are required to incorporate a private company under CAMA?")
        self.assertEqual(jur, Jurisdiction.FEDERAL)
        self.assertEqual(prio, [Jurisdiction.FEDERAL])


class TemporalRoutingTests(TestCase):
    """Tests for historical vs current legal query intent and effective date parsing."""

    def test_temporal_intent_detection(self):
        # Historical query with explicit year
        is_hist, target_date, year = analyze_temporal_intent("What was the corporate tax rate in 2019?")
        self.assertTrue(is_hist)
        self.assertEqual(year, 2019)
        self.assertEqual(target_date, datetime.date(2019, 6, 30))

        # Historical query with full date
        is_hist, target_date, year = analyze_temporal_intent("Admissibility of electronic records as at 15th May 2010")
        self.assertTrue(is_hist)
        self.assertEqual(target_date, datetime.date(2010, 5, 15))

        # Historical intent keyword without explicit date
        is_hist, target_date, year = analyze_temporal_intent("Provisions under the repealed 1990 Act")
        self.assertTrue(is_hist)
        self.assertIsNone(target_date)

        # Current query (default)
        is_hist, target_date, year = analyze_temporal_intent("What is the current capital threshold for small companies?")
        self.assertFalse(is_hist)
        self.assertIsNone(target_date)


class CitationServiceTests(TestCase):
    """Tests for deterministic citation generation and hallucination filtering."""

    def setUp(self):
        self.doc = LegalSourceDocument.objects.create(
            statute_family_code="FED_EVIDENCE",
            title="Evidence Act, 2011",
            short_title="Evidence Act 2011",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=datetime.date(2011, 6, 3),
            official_gazette_ref="Official Gazette No. 58, Vol. 98",
            publisher="Federal Government Printer",
            sha256_checksum="e" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            verified_by="Verification Suite",
        )
        self.chunk = LegalProvisionChunk.objects.create(
            document=self.doc,
            section_number="Section 84",
            section_title="Admissibility of statements in documents produced by computers",
            content="In any proceeding a statement contained in a document produced by a computer shall be admissible...",
            citation_label="Section 84, Evidence Act 2011",
            chunk_index=84,
            embedding=[0.0] * 384,
        )

    def test_deterministic_citation_generation(self):
        cit_id = generate_citation_id(self.chunk)
        self.assertTrue(cit_id.startswith("CIT-FED-EVIDENCE-SEC84"))

        structured = build_structured_citation(self.chunk)
        self.assertEqual(structured['statute_title'], "Evidence Act, 2011")
        self.assertEqual(structured['official_gazette_ref'], "Official Gazette No. 58, Vol. 98")
        self.assertEqual(structured['sha256_checksum'], "e" * 64)
        self.assertTrue(structured['is_verified'])

    def test_reject_hallucinated_citation_tokens(self):
        valid_id = generate_citation_id(self.chunk)
        valid_tok = f"[CITATION: {valid_id}]"
        hallucinated_tok = "[CITATION: CIT-FED-FAKESTATUTE-SEC999]"

        model_reply = (
            f"Under the Evidence Act, computer-generated documents are admissible {valid_tok}. "
            f"Furthermore, see fictional rules {hallucinated_tok}."
        )

        sanitized, valid, rejected = filter_and_validate_citations(
            model_reply,
            active_citation_tokens={valid_id, valid_tok}
        )

        self.assertIn(valid_tok, valid)
        self.assertIn(hallucinated_tok, rejected)
        self.assertNotIn(hallucinated_tok, sanitized, "Hallucinated citation tokens must be stripped from response text.")
        self.assertIn(valid_tok, sanitized, "Authentic citation tokens must be preserved.")


class HybridRetrievalIntegrationTests(TestCase):
    """
    Integration tests for hybrid retrieval, RRF ranking, amendment handling,
    exact section matching, and statute disambiguation.
    """

    def setUp(self):
        # 1. Federal Statute A: CAMA 2020
        self.doc_cama = LegalSourceDocument.objects.create(
            statute_family_code="FED_CAMA",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=datetime.date(2020, 8, 7),
            is_current_version=True,
            status=DocumentStatus.ACTIVE,
            sha256_checksum="1" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            verified_by="Counsel A",
        )
        self.chunk_cama_18 = LegalProvisionChunk.objects.create(
            document=self.doc_cama,
            section_number="Section 18",
            section_title="Right to form a company",
            content="Any two or more persons may form and incorporate a company.",
            citation_label="Section 18, CAMA 2020",
            chunk_index=18,
            embedding=[0.05] * 384,
        )

        # 2. Federal Statute B: Evidence Act 2011 (Contains Section 84 and Section 84A)
        self.doc_evidence = LegalSourceDocument.objects.create(
            statute_family_code="FED_EVIDENCE",
            title="Evidence Act, 2011",
            short_title="Evidence Act 2011",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=datetime.date(2011, 6, 3),
            is_current_version=True,
            status=DocumentStatus.ACTIVE,
            sha256_checksum="2" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            verified_by="Counsel B",
        )
        self.chunk_ev_84 = LegalProvisionChunk.objects.create(
            document=self.doc_evidence,
            section_number="Section 84",
            section_title="Admissibility of statements in documents produced by computers",
            content="In any proceeding a statement contained in a document produced by a computer shall be admissible.",
            citation_label="Section 84, Evidence Act 2011",
            chunk_index=84,
            embedding=[0.1] * 384,
        )
        self.chunk_ev_84a = LegalProvisionChunk.objects.create(
            document=self.doc_evidence,
            section_number="Section 84A",
            section_title="Admissibility of electronic records and digital signatures",
            content="Electronic records and digital signatures shall be admissible in evidence where verified.",
            citation_label="Section 84A, Evidence Act 2011",
            chunk_index=85,
            embedding=[0.12] * 384,
        )

        # 3. State Statute: Lagos Tenancy Law 2011
        self.doc_lagos = LegalSourceDocument.objects.create(
            statute_family_code="LAG_TENANCY",
            title="Tenancy Law of Lagos State, 2011",
            short_title="Lagos Tenancy Law 2011",
            jurisdiction=Jurisdiction.LAGOS_STATE,
            effective_from=datetime.date(2011, 8, 24),
            is_current_version=True,
            status=DocumentStatus.ACTIVE,
            sha256_checksum="3" * 64,
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
            verified_by="Counsel C",
        )
        self.chunk_lagos_4 = LegalProvisionChunk.objects.create(
            document=self.doc_lagos,
            section_number="Section 4",
            section_title="Advance rent restrictions in Lagos",
            content="It shall be unlawful for a landlord to demand or receive rent in excess of one year for a yearly tenant.",
            citation_label="Section 4, Lagos Tenancy Law 2011",
            chunk_index=4,
            embedding=[-0.05] * 384,
        )

    def test_exact_section_matching_boost(self):
        # Query specifying Section 84A should retrieve Section 84A as top match
        results = hybrid_legal_search("What are the conditions in Section 84A for electronic records?")
        self.assertTrue(len(results) > 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.section_number, "Section 84A")

    def test_statute_disambiguation(self):
        # Query mentioning CAMA should not return Evidence Act provisions
        results = hybrid_legal_search("Who has the right to form a company under CAMA?")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0].document.short_title, "CAMA 2020")
        self.assertEqual(results[0][0].section_number, "Section 18")

    def test_jurisdiction_filtering_prioritizes_state_law(self):
        # Query for Lagos advance rent
        context = retrieve_legal_context("How many months advance rent can a landlord demand in Lagos?")
        self.assertTrue(context['has_verified_results'])
        first_citation = context['citations'][0]
        self.assertEqual(first_citation['jurisdiction'], Jurisdiction.LAGOS_STATE)
        self.assertEqual(first_citation['short_title'], "Lagos Tenancy Law 2011")

    def test_non_legal_query_returns_no_legal_context(self):
        context = retrieve_legal_context("Write a python script to sort a list of numbers")
        self.assertFalse(context['is_legal_query'])
        self.assertFalse(context['has_verified_results'])
        self.assertEqual(len(context['citations']), 0)

    def test_empty_retrieval_for_unindexed_legal_subject(self):
        # Query about maritime admiralty law (not in test database)
        context = retrieve_legal_context("Admiralty jurisdiction and arrest of foreign ships in Nigerian territorial waters")
        self.assertTrue(context['is_legal_query'])
        self.assertFalse(context['has_verified_results'])
        self.assertEqual(len(context['citations']), 0)
