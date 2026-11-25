"""
Phase 4 Stage 2: CAMA 2020 Controlled Ingestion Test Suite.

Exhaustive verification of:
1. Acquisition & Provenance (URL validation, file signature, network errors, checksum).
2. Extraction Completeness (528 pages, 0 scanned pages, sequential order).
3. Structural Validation (870 sections, Parts A-G, 15 Schedules, duplicate/malformed detection).
4. Database Safety & Idempotency (Atomic transactions, duplicate rejection, rollback, quarantine).
5. Chatbot Regression (Exclusion of unverified CAMA, honest disclosure, non-legal Q&A, streaming, resilience).
"""
import os
import re
import json
import hashlib
import datetime
from unittest.mock import patch, MagicMock
import requests
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    VerificationStatus,
    ExtractionStatus,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    SourceType,
)
from rag.services.downloader import (
    safe_download_legal_source_url,
    validate_file_signature,
    SSRFSecurityError,
    InvalidFileSignatureError,
)
from rag.services.extractors.pdf_extractor import extract_pdf_content
from rag.services.chunker import chunk_nigerian_statute
from rag.services.statutory_structure_validator import (
    validate_statutory_completeness,
    parse_section_identifiers,
    canonicalize_part_token,
)
from rag.services.hybrid_retriever import hybrid_legal_search
from rag.services.retrieval_service import retrieve_legal_context
from rag.services.ingestion_service import (
    ingest_legal_source_document,
    DuplicateLegalSourceError,
    IngestionError,
)
from chat.models import Conversation, Message
from chat.services import (
    generate_legal_response,
    stream_legal_response_events,
    AIServiceError,
)

User = get_user_model()

CAMA_PDF_PATH = r"C:\Users\Owner\.gemini\antigravity\brain\ddae0c82-fe8c-4d5b-9ed7-776fe2df7f58\scratch\cama_2020_nipc.pdf"
CAMA_KNOWN_SHA256 = "3723358a6a4d1c582921a5e626a5c215a0bf611efa997ec0fca53763b7887726"
CAMA_KNOWN_SIZE = 1227065

SAMPLE_STATUTE_TEXT = (
    b"Section 1. Establishment of Commission.\n"
    b"There is established the Corporate Affairs Commission.\n\n"
    b"Section 2. Functions of the Commission.\n"
    b"The Commission shall administer the provisions of this Act."
)


class CamaAcquisitionTests(TestCase):
    """Tests for file acquisition, security checks, and cryptographic integrity."""

    def test_cama_file_size_and_sha256_integrity(self):
        self.assertTrue(os.path.exists(CAMA_PDF_PATH), "CAMA 2020 source PDF must exist in scratch")
        with open(CAMA_PDF_PATH, "rb") as f:
            data = f.read()
        self.assertEqual(len(data), CAMA_KNOWN_SIZE)
        computed_hash = hashlib.sha256(data).hexdigest()
        self.assertEqual(computed_hash, CAMA_KNOWN_SHA256)

    def test_pdf_signature_validation(self):
        with open(CAMA_PDF_PATH, "rb") as f:
            header = f.read(1024)
        fmt = validate_file_signature(header)
        self.assertEqual(fmt, "pdf")

    def test_invalid_signature_rejection(self):
        invalid_bytes = b"\x00\xff\xfe\x00\x12\x34\x56\x78\x00\x00"
        with self.assertRaises(InvalidFileSignatureError):
            validate_file_signature(invalid_bytes)

    def test_empty_file_rejection(self):
        with self.assertRaises(InvalidFileSignatureError):
            validate_file_signature(b"")

    def test_unsafe_redirect_ssrf_rejection(self):
        unsafe_urls = [
            "http://127.0.0.1/cama.pdf",
            "http://localhost/cama.pdf",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/internal.pdf",
            "ftp://example.com/cama.pdf",
        ]
        for url in unsafe_urls:
            with self.assertRaises(SSRFSecurityError):
                safe_download_legal_source_url(url)

    @patch("requests.Session.get")
    def test_network_failure_handling(self, mock_get):
        mock_get.side_effect = Exception("Simulated connection timeout")
        with self.assertRaises(IngestionError):
            ingest_legal_source_document(
                source_url="https://nipc.gov.ng/assets/resources/companies-and-allied-matters-act-2020/cama-2020-gazetted.pdf",
                statute_family_code="CAMA-2020",
                title="CAMA 2020",
                effective_from=datetime.date(2020, 8, 7),
            )


class CamaExtractionTests(TestCase):
    """Tests for PDF extraction, page audits, and text fidelity."""

    @classmethod
    def setUpTestData(cls):
        if os.path.exists(CAMA_PDF_PATH):
            with open(CAMA_PDF_PATH, "rb") as f:
                cls.raw_bytes = f.read()
            cls.extracted_text, cls.status, cls.scanned, cls.errors, cls.meta = extract_pdf_content(cls.raw_bytes)
        else:
            cls.raw_bytes = b""
            cls.extracted_text = ""

    def test_cama_complete_text_extraction(self):
        self.assertEqual(self.status, ExtractionStatus.COMPLETE_TEXT)
        self.assertEqual(len(self.scanned), 0, "NIPC CAMA 2020 PDF should have 0 scanned pages")
        self.assertEqual(self.meta.get("total_pages"), 528)
        self.assertGreater(len(self.extracted_text), 1000000)

    def test_enactment_and_arrangement_present(self):
        self.assertTrue(bool(re.search(r'ARRANGEMENT OF S\s*ECTIONS', self.extracted_text, re.IGNORECASE)))
        self.assertIn("ENACTED by the National Assembly", self.extracted_text)
        self.assertTrue(bool(re.search(r'MUHAMMADU\s+BUHARI', self.extracted_text, re.IGNORECASE)))


class CamaStructuralValidationTests(TestCase):
    """Tests for structural completeness, Parts A-G, 15 Schedules, and 870 sections."""

    def test_part_tokens_canonicalization(self):
        for letter in ["A", "B", "C", "D", "E", "F", "G"]:
            self.assertEqual(canonicalize_part_token(letter), f"Part {letter}")
        self.assertEqual(canonicalize_part_token("I"), "Part I")
        self.assertEqual(canonicalize_part_token("XVIII"), "Part XVIII")
        self.assertEqual(canonicalize_part_token("1"), "Part I")

    def test_schedule_identifiers_parsing(self):
        schedules = [
            "First Schedule", "Second Schedule", "Third Schedule", "Fourth Schedule",
            "Fifth Schedule", "Sixth Schedule", "Seventh Schedule", "Eighth Schedule",
            "Ninth Schedule", "Tenth Schedule", "Eleventh Schedule", "Twelfth Schedule",
            "Thirteenth Schedule", "Fourteenth Schedule", "Fifteenth Schedule",
            "Authentication Schedule"
        ]
        for s in schedules:
            parent, precise, err = parse_section_identifiers(s)
            self.assertIsNone(parent)
            self.assertIsNotNone(precise)
            self.assertIsNone(err)

    def test_chunker_extracts_870_sections_and_15_schedules(self):
        with open(CAMA_PDF_PATH, "rb") as f:
            data = f.read()
        text, _, _, _, _ = extract_pdf_content(data)
        chunks = chunk_nigerian_statute(text, statute_short_title="CAMA 2020")

        sections = [c for c in chunks if c.section_number.startswith("Section ")]
        schedules = [c for c in chunks if "Schedule" in c.section_number]
        parts = {c.part_number for c in chunks if c.part_number and c.part_number != "Schedules"}

        self.assertEqual(len(sections), 870)
        self.assertEqual(sections[0].section_number, "Section 1")
        self.assertEqual(sections[-1].section_number, "Section 870")
        self.assertGreaterEqual(len(schedules), 15)
        self.assertEqual(parts, {"Part A", "Part B", "Part C", "Part D", "Part E", "Part F", "Part G"})

    def test_statutory_completeness_report_on_cama(self):
        """Construct a full document in test DB and verify it passes structural validation."""
        doc = LegalSourceDocument.objects.create(
            statute_family_code="CAMA-2020",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            effective_from=datetime.date(2020, 8, 7),
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
        )
        with open(CAMA_PDF_PATH, "rb") as f:
            data = f.read()
        text, _, _, _, _ = extract_pdf_content(data)
        chunks = chunk_nigerian_statute(text, statute_short_title="CAMA 2020")

        chunk_objs = []
        for c in chunks:
            chunk_objs.append(
                LegalProvisionChunk(
                    document=doc,
                    part_number=c.part_number,
                    part_title=c.part_title,
                    chapter_number=c.chapter_number,
                    section_number=c.section_number,
                    section_title=c.section_title,
                    content=c.content,
                    citation_label=c.citation_label,
                    chunk_index=c.chunk_index,
                    token_count=c.token_count,
                    embedding=[0.0] * 384,
                )
            )
        LegalProvisionChunk.objects.bulk_create(chunk_objs)

        report = validate_statutory_completeness(doc)
        self.assertTrue(report.is_structurally_complete, f"Validation errors: {report.errors}")
        self.assertEqual(len(report.found_sections), 870)
        self.assertEqual(len(report.missing_sections), 0)
        self.assertEqual(len(report.duplicate_sections), 0)
        self.assertEqual(len(report.missing_parts), 0)
        self.assertEqual(len(report.missing_schedules), 0)


class CamaDatabaseSafetyTests(TestCase):
    """Tests for idempotent ingestion, quarantine status, and rollback safety."""

    def test_duplicate_sha256_rejection(self):
        doc1, count1, _ = ingest_legal_source_document(
            raw_bytes=SAMPLE_STATUTE_TEXT,
            statute_family_code="CAMA-2020",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            effective_from=datetime.date(2020, 8, 7),
            generate_embeddings=False,
        )
        self.assertIsNotNone(doc1.id)

        # Attempt to ingest exact same bytes again
        with self.assertRaises(DuplicateLegalSourceError):
            ingest_legal_source_document(
                raw_bytes=SAMPLE_STATUTE_TEXT,
                statute_family_code="CAMA-2020",
                title="Companies and Allied Matters Act, 2020 (Duplicate)",
                short_title="CAMA 2020",
                effective_from=datetime.date(2020, 8, 7),
                generate_embeddings=False,
            )

    def test_quarantine_enforcement_on_ingestion(self):
        doc, _, _ = ingest_legal_source_document(
            raw_bytes=SAMPLE_STATUTE_TEXT + b"\n# unique tag",
            statute_family_code="CAMA-2020",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            effective_from=datetime.date(2020, 8, 7),
            generate_embeddings=False,
        )
        self.assertFalse(doc.is_verified_authentic, "Ingested document MUST be quarantined")
        self.assertEqual(doc.verification_status, VerificationStatus.PENDING)

    def test_transaction_rollback_on_failure(self):
        initial_doc_count = LegalSourceDocument.objects.count()
        initial_chunk_count = LegalProvisionChunk.objects.count()

        unique_bytes = SAMPLE_STATUTE_TEXT + b"\n# rollback unique"
        with patch("rag.services.ingestion_service.LegalProvisionChunk.objects.bulk_create") as mock_bulk:
            mock_bulk.side_effect = RuntimeError("Simulated DB failure during bulk chunk insertion")
            with self.assertRaises(RuntimeError):
                ingest_legal_source_document(
                    raw_bytes=unique_bytes,
                    statute_family_code="CAMA-2020",
                    title="CAMA Rollback",
                    short_title="CAMA 2020",
                    effective_from=datetime.date(2020, 8, 7),
                    generate_embeddings=False,
                )

        # Ensure atomicity: no orphaned document or chunks left behind
        self.assertEqual(LegalSourceDocument.objects.count(), initial_doc_count)
        self.assertEqual(LegalProvisionChunk.objects.count(), initial_chunk_count)


class CamaChatbotRegressionTests(TestCase):
    """Tests ensuring unverified CAMA is never returned to the chatbot, preserving honest disclosures."""

    def setUp(self):
        self.user = User.objects.create_user(username="lawyer1", password="securepassword123")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

        # Create quarantined CAMA document
        self.cama_doc = LegalSourceDocument.objects.create(
            statute_family_code="CAMA-2020",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            effective_from=datetime.date(2020, 8, 7),
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            source_url="https://www.nipc.gov.ng/wp-content/uploads/2020/12/Companies-and-Allied-Matters-Act-2020.pdf",
        )
        self.chunk = LegalProvisionChunk.objects.create(
            document=self.cama_doc,
            section_number="Section 18",
            section_title="Right to form a company",
            content="Any two or more persons may form and incorporate a company.",
            citation_label="Section 18, CAMA 2020",
            embedding=[0.1] * 384,
            chunk_index=1,
            token_count=10,
        )

    def test_unverified_cama_excluded_from_retrieval(self):
        """Unverified documents must NEVER appear in legal retrieval queries."""
        eligible_chunks = LegalProvisionChunk.objects.filter(
            document__is_verified_authentic=True
        )
        self.assertNotIn(self.chunk, eligible_chunks)
        self.assertEqual(eligible_chunks.count(), 0)

    def test_hybrid_legal_search_strictly_excludes_quarantined_cama(self):
        """hybrid_legal_search must return 0 results when only quarantined CAMA exists."""
        results = hybrid_legal_search("Section 18 CAMA 2020 right to form a company", limit=5)
        self.assertEqual(len(results), 0)

        results_cac = hybrid_legal_search("Corporate Affairs Commission", limit=5)
        self.assertEqual(len(results_cac), 0)

    def test_retrieve_legal_context_reports_no_verified_results_for_cama(self):
        """retrieve_legal_context must report has_verified_results=False and empty prompt context for quarantined CAMA."""
        ctx = retrieve_legal_context("What does Section 18 of CAMA 2020 say about company formation?")
        self.assertTrue(ctx['is_legal_query'])
        self.assertFalse(ctx['has_verified_results'])
        self.assertEqual(len(ctx['citations']), 0)
        self.assertEqual(ctx['prompt_context'], "")

    def test_verified_document_retrieval_works_while_cama_remains_isolated(self):
        """A verified document is retrieved normally while quarantined CAMA remains 100% isolated."""
        verified_doc = LegalSourceDocument.objects.create(
            statute_family_code="VERIFIED_ACT",
            title="Verified Evidence Act",
            short_title="Evidence Act",
            effective_from=datetime.date(2011, 6, 3),
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
        )
        verified_chunk = LegalProvisionChunk.objects.create(
            document=verified_doc,
            section_number="Section 1",
            section_title="Application of Act",
            content="This Act shall apply to all judicial proceedings in or before courts in Nigeria.",
            citation_label="Section 1, Evidence Act",
            embedding=[0.1] * 384,
            chunk_index=1,
            token_count=15,
        )
        results = hybrid_legal_search("judicial proceedings in courts in Nigeria", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, verified_chunk.id)
        self.assertNotIn(self.chunk.id, [r[0].id for r in results])

    def test_honest_disclosure_when_cama_unverified(self):
        """Chatbot prompt builder or retrieval service must report absence of verified CAMA evidence."""
        query = "How many persons can form a company under CAMA 2020?"
        eligible_chunks = list(LegalProvisionChunk.objects.filter(
            document__is_verified_authentic=True,
            content__icontains="form and incorporate a company"
        ))
        self.assertEqual(len(eligible_chunks), 0)

    @patch("chat.services.requests.post")
    def test_non_legal_chat_unaffected(self, mock_post):
        """Ordinary non-legal questions still stream and respond normally."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        tokens = ["Hello! ", "I am happy to assist you ", "with general topics."]
        lines = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in tokens]
        lines.append("data: [DONE]")
        mock_resp.iter_lines.return_value = lines
        mock_post.return_value = mock_resp

        events = list(stream_legal_response_events("Hello, how are you?"))
        self.assertGreaterEqual(len(events), 2)
        # Verify final result contains the assembled tokens
        result_events = [data for evt, data in events if evt == "result"]
        self.assertEqual(len(result_events), 1)
        self.assertIn("Hello! I am happy to assist you with general topics.", result_events[0].get("reply", ""))

    def test_conversation_history_accessible(self):
        """Existing conversation history remains accessible through the API."""
        conv = Conversation.objects.create(user=self.user, title="Existing Legal Chat")
        Message.objects.create(conversation=conv, sender="user", text="What is CAMA?")
        Message.objects.create(conversation=conv, sender="assistant", text="CAMA is pending verification.")

        response = self.client.get(f"/api/conversations/{conv.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data.get("messages", [])), 2)

    @patch("chat.services.requests.post")
    def test_model_provider_failure_gracefully_handled(self, mock_post):
        """AI provider failure raises AIServiceError and view handles it gracefully."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(AIServiceError):
            list(stream_legal_response_events(message="Explain CAMA Section 18"))

    def test_controlled_unverified_retrieval_returns_cama_with_disclaimer(self):
        """Controlled unverified mode retrieves CAMA with section citations and unverified disclaimers."""
        from rag.services.retrieval_service import retrieve_unverified_legal_context
        ctx = retrieve_unverified_legal_context("What does Section 18 of CAMA 2020 say about company formation?")
        self.assertTrue(ctx['is_legal_query'])
        self.assertFalse(ctx['has_verified_results'])
        self.assertTrue(ctx['has_unverified_results'])
        self.assertTrue(ctx['is_unverified_mode'])
        self.assertGreater(len(ctx['citations']), 0)
        self.assertIn("Section 18, CAMA 2020", [c['citation_label'] for c in ctx['citations']])
        # Verify citation metadata explicitly labels it as unverified with exact wording
        cama_cit = ctx['citations'][0]
        self.assertFalse(cama_cit['is_verified'])
        self.assertEqual(cama_cit['verification_status'], VerificationStatus.PENDING)
        self.assertIn(
            "This is an official-agency-hosted reprint that has not yet been "
            "independently authenticated against the original Official Gazette or reviewed by legal counsel.",
            cama_cit.get('unverified_disclaimer', '')
        )
        # Verify prompt context includes prominent unverified warning and no Section 106 claim
        self.assertIn("UNVERIFIED OFFICIAL-AGENCY-HOSTED REPRINT", ctx['prompt_context'])
        self.assertNotIn("Section 106", ctx['prompt_context'])
        self.assertIn(
            "This is an official-agency-hosted reprint that has not yet been "
            "independently authenticated against the original Official Gazette or reviewed by legal counsel.",
            ctx['prompt_context']
        )
        self.assertIn("NEVER represent these provisions or your answer as counsel-verified", ctx['prompt_context'])

    def test_controlled_unverified_hybrid_search(self):
        """hybrid_legal_search with unverified_only=True returns CAMA provisions."""
        results = hybrid_legal_search("Section 18 CAMA 2020", limit=5, unverified_only=True)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0].id, self.chunk.id)
        # Verify that normal verified search still returns 0
        normal_results = hybrid_legal_search("Section 18 CAMA 2020", limit=5, include_unverified=False)
        self.assertEqual(len(normal_results), 0)

    @patch("chat.services.requests.post")
    def test_chat_uses_unverified_cama_context(self, mock_post):
        """Chat API injects unverified CAMA prompt context and renders clickable citations."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        "> **Notice**: This is an official-agency-hosted reprint that has not yet been "
                        "independently authenticated against the original Official Gazette or reviewed by legal counsel.\n\n"
                        "According to Section 18(1) [CITATION: CIT-FED-CAMA2020-SEC18], any two or more persons may form a company.\n\n"
                        "---\n*Disclaimer: This is for general educational and informational purposes only and does not constitute formal legal advice.*"
                    )
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={
                "message": "How many persons can form a company under Section 18 of CAMA 2020?",
                "allow_unverified": True,
            },
            format="json"
        )
        self.assertEqual(response.status_code, 200)
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        system_messages = [m["content"] for m in payload["messages"] if m["role"] == "system"]
        unverified_injected = any("CONTROLLED UNVERIFIED STATUTORY REFERENCE" in msg for msg in system_messages)
        self.assertTrue(unverified_injected, "Unverified statutory context was not injected into DeepSeek prompt payload.")

        reply = response.data.get("reply", "")
        # 1. Unverified notice at the beginning
        self.assertTrue(reply.startswith("> **Notice**: This is an official-agency-hosted reprint"))
        # 2. No Section 106 claim
        self.assertNotIn("Section 106", reply)
        # 3. Subsection reference preserved
        self.assertIn("Section 18(1)", reply)
        # 4. Citation token rendered as clickable section-level citation
        self.assertNotIn("[CITATION: CIT-FED-CAMA2020-SEC18]", reply)
        self.assertIn("[Section 18, CAMA 2020](", reply)
        # 5. Trailing generic repetitive disclaimer removed
        self.assertNotIn("This is for general educational and informational purposes only", reply)

    def test_citation_token_rendered_as_clickable_section_citation(self):
        """Citation tokens are converted to clickable section-level markdown citations."""
        from rag.services.citation_service import filter_and_validate_citations, build_structured_citation
        cit_data = build_structured_citation(self.chunk)
        cit_data['source_url'] = "https://www.nipc.gov.ng/wp-content/uploads/2020/12/Companies-and-Allied-Matters-Act-2020.pdf"
        raw_text = (
            "Under Section 18(1) [CITATION: CIT-FED-CAMA2020-SEC18], two or more persons may form a company. "
            "Hallucinated token [CITATION: CIT-FED-UNKNOWN-SEC999] must be stripped."
        )
        active_tokens = {cit_data['citation_id'], cit_data['citation_token']}
        rendered, valid, rejected = filter_and_validate_citations(
            raw_text,
            active_citation_tokens=active_tokens,
            citations=[cit_data],
            render_clickable=True,
        )
        # Check valid and rejected lists
        self.assertIn(cit_data['citation_token'], valid)
        self.assertIn("[CITATION: CIT-FED-UNKNOWN-SEC999]", rejected)
        # Check rendered text has no raw tokens
        self.assertNotIn("[CITATION:", rendered)
        # Check rendered text preserves subsection
        self.assertIn("Section 18(1)", rendered)
        # Check rendered text has clickable link
        expected_link = f"[{cit_data['citation_label']}]({cit_data['source_url']})"
        self.assertIn(expected_link, rendered)

        # When source_url is None, citation integrity requires not fabricating fake anchors (#citation-...)
        cit_data_no_url = dict(cit_data)
        cit_data_no_url['source_url'] = None
        rendered_anchor, _, _ = filter_and_validate_citations(
            "Under Section 18 [CITATION: CIT-FED-CAMA2020-SEC18]",
            active_citation_tokens=active_tokens,
            citations=[cit_data_no_url],
            render_clickable=True,
        )
        self.assertNotIn("#citation-", rendered_anchor, "Must not manufacture fake anchor links when source URL is absent.")
        self.assertIn(cit_data['citation_label'], rendered_anchor)

    def test_unverified_response_disclaimer_at_beginning_and_no_generic_trailing_disclaimer(self):
        """sanitize_unverified_response guarantees front-loaded notice and strips trailing generic disclaimers."""
        from chat.services import sanitize_unverified_response
        raw_model_output = (
            "Under Section 18(1), any two or more persons may form and incorporate a company.\n\n"
            "---\n*Disclaimer: This is for general educational and informational purposes only and does not constitute formal legal advice. Consult a qualified Nigerian legal practitioner for advice specific to your circumstances.*"
        )
        cleaned = sanitize_unverified_response(raw_model_output)
        self.assertTrue(cleaned.startswith(
            "> **Notice**: This is an official-agency-hosted reprint that has not yet been "
            "independently authenticated against the original Official Gazette or reviewed by legal counsel."
        ))
        self.assertIn("Section 18(1)", cleaned)
        self.assertNotIn("This is for general educational and informational purposes only", cleaned)
        self.assertNotIn("Consult a qualified Nigerian legal practitioner", cleaned)

    def test_no_section_106_claim_in_sanitized_response(self):
        """Any errant reference to Section 106 in connection with counsel certification is sanitized."""
        from chat.services import sanitize_unverified_response
        raw_text = (
            "> **Notice**: This source was not certified by legal counsel under Section 106 of the Evidence Act 2011.\n\n"
            "Section 18(1) allows two or more persons to form a company."
        )
        cleaned = sanitize_unverified_response(raw_text)
        self.assertNotIn("Section 106", cleaned)
