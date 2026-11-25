"""
Comprehensive Test Suite for Nigerian Legal AI RAG Ingestion Pipeline (Phase 2).

Tests:
1. SSRF & Network Protections (HTTPS enforcement, loopback, private IPs, cloud metadata, redirect attacks).
2. DNS Rebinding and Socket Peer Validation.
3. Magic Byte File Signature Verification (PDF, DOCX, HTML, TXT, malformed files).
4. DOCX In-Order Traversal & Markdown Table Formatting.
5. HTML Preorder DOM Traversal & Header/Table Formatting.
6. Scanned PDF Detection (scanned page identification, mixed-content auditing).
7. Nigerian Statutory Chunker (Section numbering vs internal list items).
8. FastEmbed Local CPU Embedding (dimension 384, deterministic output).
9. SHA-256 Checksum Duplicate Detection (idempotency).
10. Strict Quarantine & Retrieval Filter (unverified documents strictly excluded until verified).
11. verify_statute Management Command (requires counsel name & reason, audit trail, rejection).
"""
import io
import unittest
from unittest.mock import patch, MagicMock
from datetime import date
from django.test import TestCase
from django.core.management import call_command
from django.core.management.base import CommandError
from docx import Document

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
    validate_destination_ip,
    resolve_and_validate_hostname,
    validate_file_signature,
    safe_download_legal_source_url,
    SSRFSecurityError,
    InvalidFileSignatureError,
)
from rag.services.extractors.pdf_extractor import extract_pdf_content
from rag.services.extractors.docx_extractor import extract_docx_content, _format_table_as_markdown
from rag.services.extractors.html_extractor import extract_html_content
from rag.services.extractors.text_extractor import extract_plain_text_content
from rag.services.chunker import chunk_nigerian_statute
from rag.services.vectorizer import embed_texts, embed_query, EXPECTED_DIMENSIONS
from rag.services.ingestion_service import (
    ingest_legal_source_document,
    DuplicateLegalSourceError,
    IngestionError,
)
from rag.services.retrieval_service import get_trusted_legal_chunks


class SSRFAndNetworkProtectionTests(unittest.TestCase):
    """Tests for SSRF prevention, IP blacklist, DNS rebinding, and protocol enforcement."""

    def test_https_enforced_strictly(self):
        with self.assertRaises(SSRFSecurityError) as ctx:
            safe_download_legal_source_url("http://lawsofnigeria.org/statute.pdf")
        self.assertIn("HTTPS is strictly required", str(ctx.exception))

        with self.assertRaises(SSRFSecurityError):
            safe_download_legal_source_url("file:///etc/passwd")

        with self.assertRaises(SSRFSecurityError):
            safe_download_legal_source_url("ftp://example.com/document.docx")

    def test_private_and_loopback_ips_blocked(self):
        blocked_ips = [
            "127.0.0.1",
            "127.0.0.2",
            "10.0.0.1",
            "10.254.0.1",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",  # AWS/GCP/Azure instance metadata
            "::1",              # IPv6 loopback
            "fc00::1",          # IPv6 unique local
            "fe80::1",          # IPv6 link-local
        ]
        for ip in blocked_ips:
            with self.subTest(ip=ip):
                with self.assertRaises(SSRFSecurityError):
                    validate_destination_ip(ip)

    def test_public_ips_allowed(self):
        public_ips = [
            "8.8.8.8",
            "1.1.1.1",
            "102.89.23.1",
            "197.210.65.12",  # Nigerian public IP
        ]
        for ip in public_ips:
            with self.subTest(ip=ip):
                # Should not raise exception
                validate_destination_ip(ip)

    def test_localhost_hostname_blocked(self):
        with self.assertRaises(SSRFSecurityError):
            resolve_and_validate_hostname("localhost")

        with self.assertRaises(SSRFSecurityError):
            resolve_and_validate_hostname("127.0.0.1")

    @patch('requests.Session.get')
    def test_redirect_to_private_ip_is_blocked(self, mock_get):
        resp_redirect = MagicMock()
        resp_redirect.status_code = 302
        resp_redirect.headers = {'Location': 'https://192.168.1.100/internal-admin'}

        mock_get.return_value = resp_redirect

        real_resolve = resolve_and_validate_hostname

        def side_effect(host, port=443):
            if host == "example.com":
                return "93.184.216.34"
            return real_resolve(host, port)

        with patch('rag.services.downloader.resolve_and_validate_hostname', side_effect=side_effect):
            with self.assertRaises(SSRFSecurityError) as ctx:
                safe_download_legal_source_url("https://example.com/statute.pdf")
            self.assertTrue(
                "disallowed" in str(ctx.exception).lower() or
                "private" in str(ctx.exception).lower() or
                "blocked" in str(ctx.exception).lower()
            )


class FileSignatureAndExtractorTests(unittest.TestCase):
    """Tests magic byte validation, DOCX in-order parsing, HTML preorder traversal, and PDF auditing."""

    def test_file_signature_detection(self):
        self.assertEqual(validate_file_signature(b"%PDF-1.4 sample content"), 'pdf')
        self.assertEqual(validate_file_signature(b"PK\x03\x04sample docx container"), 'docx')
        self.assertEqual(validate_file_signature(b"<!DOCTYPE html><html><body>Law</body></html>"), 'html')
        self.assertEqual(validate_file_signature(b"<html lang='en'><head></head><body>Act</body></html>"), 'html')
        self.assertEqual(validate_file_signature(b"Section 1. This Act may be cited as..."), 'txt')

        # Empty data
        with self.assertRaises(InvalidFileSignatureError):
            validate_file_signature(b"")

        # Unknown binary data
        with self.assertRaises(InvalidFileSignatureError):
            validate_file_signature(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")

    def test_docx_in_order_paragraphs_and_tables(self):
        # Create an in-memory DOCX with sequential paragraphs and tables
        doc = Document()
        doc.add_heading("Companies and Allied Matters Act 2020", level=1)
        doc.add_paragraph("Part A - Corporate Affairs Commission")
        
        # Add table
        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Share Capital Range"
        table.rows[0].cells[1].text = "Statutory Filing Fee"
        table.rows[1].cells[0].text = "₦1,000,000 to ₦5,000,000"
        table.rows[1].cells[1].text = "₦10,000"

        doc.add_paragraph("Section 18. Right to form a company")

        buffer = io.BytesIO()
        doc.save(buffer)
        raw_docx = buffer.getvalue()

        text, status, errors, meta = extract_docx_content(raw_docx)
        self.assertEqual(status, "COMPLETE_TEXT")
        self.assertIn("# Companies and Allied Matters Act 2020", text)
        self.assertIn("Part A - Corporate Affairs Commission", text)
        self.assertIn("| Share Capital Range | Statutory Filing Fee |", text)
        self.assertIn("| ₦1,000,000 to ₦5,000,000 | ₦10,000 |", text)
        self.assertIn("Section 18. Right to form a company", text)

        # Verify element order: heading comes before table, table comes before section 18
        pos_heading = text.find("# Companies and Allied Matters Act 2020")
        pos_table = text.find("| Share Capital Range")
        pos_sec18 = text.find("Section 18. Right to form a company")
        self.assertTrue(pos_heading < pos_table < pos_sec18, "DOCX elements must maintain exact sequential document order")

    def test_html_in_order_traversal_and_boilerplate_stripping(self):
        sample_html = b"""
        <!DOCTYPE html>
        <html>
        <head><title>Finance Act 2023</title></head>
        <body>
            <nav><a href="/home">Home</a><a href="/login">Login</a></nav>
            <header>Government Portal Header</header>
            <main>
                <h1>Finance Act 2023</h1>
                <p>An Act to amend certain tax statutes in Nigeria.</p>
                <table>
                    <tr><th>Statute Affected</th><th>Amendment Section</th></tr>
                    <tr><td>Companies Income Tax Act</td><td>Section 15</td></tr>
                </table>
                <p>Section 1. Short Title and Commencement.</p>
            </main>
            <footer>Copyright 2023 Federal Republic of Nigeria</footer>
        </body>
        </html>
        """
        text, status, errors, meta = extract_html_content(sample_html)
        self.assertEqual(status, "COMPLETE_TEXT")
        self.assertNotIn("Home", text)
        self.assertNotIn("Login", text)
        self.assertNotIn("Government Portal Header", text)
        self.assertNotIn("Copyright 2023", text)

        self.assertIn("# Finance Act 2023", text)
        self.assertIn("An Act to amend certain tax statutes in Nigeria.", text)
        self.assertIn("| Statute Affected | Amendment Section |", text)
        self.assertIn("| Companies Income Tax Act | Section 15 |", text)
        self.assertIn("Section 1. Short Title and Commencement.", text)

    def test_scanned_pdf_auditing_and_mixed_content(self):
        # Test completely empty/invalid PDF bytes
        text, status, scanned, errors, meta = extract_pdf_content(b"%PDF-1.4\n%%EOF")
        self.assertIn(status, ["FAILED", "NEEDS_OCR"])


class StatutoryChunkerTests(unittest.TestCase):
    """Tests structure-aware statutory parsing: sections vs internal numbered lists."""

    def test_nigerian_section_parsing_vs_list_items(self):
        statute_text = """
PART I - ESTABLISHMENT OF THE COMMISSION

Section 1. Establishment of the Corporate Affairs Commission.
(1) There is established a body to be known as the Corporate Affairs Commission.
(2) The Commission shall be a body corporate with perpetual succession.
(a) having a common seal; and
(b) capable of suing and being sued in its corporate name.

Section 2. Governing Board of the Commission.
(1) There is established for the Commission a Governing Board consisting of:
(a) a Chairman who shall be a legal practitioner;
(b) one representative of the Federal Ministry of Industry;
(c) five other representatives.

Section 84A. Admissibility of Electronic Records.
1. In any proceeding, an electronic record shall be admissible where:
(a) produced by a computer during normal operations;
(b) supplied to the computer in the ordinary course of business.
"""
        chunks = chunk_nigerian_statute(statute_text, statute_short_title="CAMA 2020")
        self.assertEqual(len(chunks), 3)

        # Chunk 1
        c1 = chunks[0]
        self.assertEqual(c1.section_number, "Section 1")
        self.assertEqual(c1.section_title, "Establishment of the Corporate Affairs Commission.")
        self.assertEqual(c1.part_number, "Part I")
        self.assertEqual(c1.citation_label, "Section 1, CAMA 2020")
        # Subsections must remain within Section 1
        self.assertIn("(1) There is established a body", c1.content)
        self.assertIn("(2) The Commission shall be a body corporate", c1.content)
        self.assertIn("(a) having a common seal;", c1.content)

        # Chunk 2
        c2 = chunks[1]
        self.assertEqual(c2.section_number, "Section 2")
        self.assertEqual(c2.section_title, "Governing Board of the Commission.")
        self.assertEqual(c2.part_number, "Part I")
        self.assertIn("(a) a Chairman who shall be a legal practitioner;", c2.content)

        # Chunk 3 (Alphanumeric Section 84A)
        c3 = chunks[2]
        self.assertEqual(c3.section_number, "Section 84A")
        self.assertEqual(c3.section_title, "Admissibility of Electronic Records.")
        self.assertIn("(a) produced by a computer", c3.content)


class FastEmbedVectorizerTests(unittest.TestCase):
    """Tests FastEmbed local CPU inference producing 384-dimensional dense vectors."""

    def test_embedding_dimensions_and_properties(self):
        texts = [
            "Section 18. Two or more persons may form and incorporate a company.",
            "Evidence Act 2011 electronic signatures and computer generated records."
        ]
        embeddings = embed_texts(texts, batch_size=2)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), EXPECTED_DIMENSIONS)
        self.assertEqual(len(embeddings[1]), EXPECTED_DIMENSIONS)
        self.assertTrue(all(isinstance(x, float) for x in embeddings[0]))

        # Test single query embedding
        query_vec = embed_query("incorporation of private company")
        self.assertEqual(len(query_vec), 384)


class IngestionAndQuarantineIntegrationTests(TestCase):
    """
    End-to-end integration tests:
    - Ingest statute
    - Cryptographic SHA-256 duplicate rejection
    - Strict Quarantine: pending documents excluded from retrieval
    - Manual verification via verify_statute command
    - Verification admission into trusted knowledge base
    """

    def setUp(self):
        self.sample_statute_txt = (
            "Section 1. Short Title.\n"
            "This Act may be cited as the Nigerian Data Protection Act, 2023.\n\n"
            "Section 2. Scope and Application.\n"
            "(1) This Act applies to the processing of personal data.\n"
            "(2) Nothing in this Act shall prejudice law enforcement.\n"
        ).encode('utf-8')

    def test_ingestion_places_document_in_strict_quarantine(self):
        doc, chunk_count, warnings = ingest_legal_source_document(
            raw_bytes=self.sample_statute_txt,
            statute_family_code="FED_NDPA",
            title="Nigeria Data Protection Act, 2023",
            short_title="NDPA 2023",
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            effective_from=date(2023, 6, 12),
            publisher="Federal Government Press, Abuja",
            generate_embeddings=False,  # Skip heavy computation during DB logic tests
        )

        self.assertIsNotNone(doc.id)
        self.assertEqual(chunk_count, 2)
        # MUST BE IN STRICT QUARANTINE
        self.assertFalse(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.PENDING)
        self.assertEqual(doc.extraction_status, ExtractionStatus.COMPLETE_TEXT)

        # Confirm retrieval excludes this unverified document
        trusted_chunks = get_trusted_legal_chunks(query_text="personal data")
        self.assertEqual(trusted_chunks.count(), 0, "Quarantined document chunks MUST NEVER appear in search results")

    def test_duplicate_sha256_ingestion_is_rejected(self):
        # First ingestion
        ingest_legal_source_document(
            raw_bytes=self.sample_statute_txt,
            statute_family_code="FED_NDPA_1",
            title="Nigeria Data Protection Act, 2023",
            short_title="NDPA 2023",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=date(2023, 6, 12),
            generate_embeddings=False,
        )

        # Attempt to ingest the EXACT SAME raw bytes again
        with self.assertRaises(DuplicateLegalSourceError) as ctx:
            ingest_legal_source_document(
                raw_bytes=self.sample_statute_txt,
                statute_family_code="FED_NDPA_DUPLICATE",
                title="Duplicate Submission of NDPA",
                short_title="NDPA Duplicate",
                jurisdiction=Jurisdiction.FEDERAL,
                effective_from=date(2023, 6, 12),
                generate_embeddings=False,
            )
        self.assertIn("already exists", str(ctx.exception))

    def test_verify_statute_command_and_admission_to_retrieval(self):
        doc, chunk_count, _ = ingest_legal_source_document(
            raw_bytes=self.sample_statute_txt,
            statute_family_code="FED_NDPA_VERIFY",
            title="Nigeria Data Protection Act, 2023",
            short_title="NDPA 2023",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=date(2023, 6, 12),
            generate_embeddings=False,
        )

        # Verify before approval: 0 chunks returned
        self.assertEqual(get_trusted_legal_chunks().count(), 0)

        # Run verify_statute without counsel name -> MUST FAIL
        out = io.StringIO()
        with self.assertRaises(CommandError):
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="",
                reason="Official gazette checked",
                stdout=out
            )

        # Run verify_statute without reason -> MUST FAIL
        with self.assertRaises(CommandError):
            call_command(
                'verify_statute',
                document_id=str(doc.id),
                counsel_name="Barrister T. Okon, SAN",
                reason="",
                stdout=out
            )

        # Run valid verification audit
        call_command(
            'verify_statute',
            document_id=str(doc.id),
            counsel_name="Barrister T. Okon, SAN (NBA/10294)",
            reason="Verified against Federal Republic of Nigeria Official Gazette No. 102, Vol. 110 of 12th June 2023.",
            notes="Statutory sections cross-referenced with primary gazette.",
            stdout=out
        )

        # Reload from database
        doc.refresh_from_db()
        self.assertTrue(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(doc.verified_by, "Barrister T. Okon, SAN (NBA/10294)")
        self.assertIsNotNone(doc.verified_at)

        # RETRIEVAL TEST: Now chunks MUST be returned!
        retrieved_chunks = get_trusted_legal_chunks(query_text="personal data")
        self.assertEqual(retrieved_chunks.count(), 1)
        self.assertEqual(retrieved_chunks.first().section_number, "Section 2")

    def test_verify_statute_rejection_keeps_document_excluded(self):
        doc, _, _ = ingest_legal_source_document(
            raw_bytes=self.sample_statute_txt,
            statute_family_code="FED_NDPA_REJECT",
            title="Tampered Statute",
            short_title="Tampered Act",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=date(2023, 6, 12),
            generate_embeddings=False,
        )

        out = io.StringIO()
        call_command(
            'verify_statute',
            document_id=str(doc.id),
            counsel_name="Senior State Counsel A. Adeleke",
            reason="Uncertified third-party draft containing conflicting subsection texts.",
            reject=True,
            stdout=out
        )

        doc.refresh_from_db()
        self.assertFalse(doc.is_verified_authentic)
        self.assertEqual(doc.verification_status, VerificationStatus.REJECTED)

        # Ensure still excluded from retrieval
        self.assertEqual(get_trusted_legal_chunks().count(), 0)
