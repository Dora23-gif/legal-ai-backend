"""
Live Neon PostgreSQL & pgvector Integration Verification Script.

Tests the full RAG pipeline against the live Neon database:
1. Document and chunk creation on Neon.
2. 384-dimensional dense vector storage in the pgvector column.
3. Cosine similarity search using pgvector.
4. PostgreSQL full-text search (automatic trigger synchronization).
5. Quarantine exclusion: unverified documents return 0 results.
6. Verification admission: verify_statute admits document into trusted retrieval.
7. Clean teardown of synthetic test records.

Uses 100% synthetic test statutes. No real legal documents are ingested.
"""
import os
import sys
import uuid
import datetime
from dotenv import load_dotenv

# Ensure backend root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import connection
from django.core.management import call_command
from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    Jurisdiction,
    DocumentCategory,
    DocumentStatus,
    VerificationStatus,
)
from rag.services.ingestion_service import ingest_legal_source_document
from rag.services.retrieval_service import get_trusted_legal_chunks
from rag.services.vectorizer import embed_query
from pgvector.django import CosineDistance


SYNTHETIC_CODE = "SYNTHETIC_TEST_PIPELINE_2026"
SYNTHETIC_TITLE = "Synthetic Nigerian Legal Innovation Act, 2026"
SYNTHETIC_SHORT = "Synthetic Act 2026"

SYNTHETIC_STATUTE_TEXT = b"""
PART I - PRELIMINARY AND ESTABLISHMENT

Section 1. Short Title and Commencement.
This Act may be cited as the Synthetic Nigerian Legal Innovation Act, 2026 and shall come into operation on the 1st day of January, 2026.

Section 2. Establishment of the Legal Artificial Intelligence Registry.
(1) There is established the Legal Artificial Intelligence Registry which shall be an autonomous regulatory registry in the Federal Republic of Nigeria.
(2) The Registry shall maintain an immutable public index of validated statutory legal provisions.
(3) Any entity deploying algorithmic legal reasoning systems shall register with the Registry.
"""


def run_pipeline_verification():
    print("=" * 75)
    print("    NEON POSTGRESQL & PGVECTOR LIVE PIPELINE VERIFICATION")
    print("=" * 75)

    vendor = connection.vendor
    print(f"Database engine: {vendor} ({connection.settings_dict.get('HOST', 'unknown')})")
    if vendor != 'postgresql':
        print(f"[FAIL] Expected PostgreSQL connection, got '{vendor}'. Check DATABASE_URL.")
        sys.exit(1)

    # Pre-clean any leftover synthetic test data
    cleanup_synthetic_data()

    doc = None
    try:
        # 1. Ingestion: Save document and chunks to Neon with 384-dim embeddings
        print("\n[1/6] Ingesting Synthetic Statute & Vectorizing with FastEmbed...")
        doc, chunk_count, warnings = ingest_legal_source_document(
            raw_bytes=SYNTHETIC_STATUTE_TEXT,
            statute_family_code=SYNTHETIC_CODE,
            title=SYNTHETIC_TITLE,
            short_title=SYNTHETIC_SHORT,
            jurisdiction=Jurisdiction.FEDERAL,
            category=DocumentCategory.PRIMARY_ACT,
            version_label="As Enacted",
            effective_from=datetime.date(2026, 1, 1),
            publisher="Federal Government Press, Abuja (Synthetic)",
            generate_embeddings=True,
        )

        assert doc is not None, "Failed to create LegalSourceDocument on Neon."
        assert chunk_count == 2, f"Expected 2 chunks, got {chunk_count}"
        print(f"  [PASS] Document created on Neon: ID={doc.id}")
        print(f"  [PASS] Chunks created on Neon: Count={chunk_count}")

        # 2. Verify 384-Dimensional Vectors in Neon pgvector Column
        print("\n[2/6] Verifying 384-dimensional Vector Storage in pgvector column...")
        chunks = list(LegalProvisionChunk.objects.filter(document=doc).order_by('chunk_index'))
        assert len(chunks) == 2, f"Expected 2 chunks in DB, got {len(chunks)}"

        for ch in chunks:
            emb = ch.embedding
            assert emb is not None, f"Chunk {ch.section_number} has null embedding!"
            emb_len = len(emb)
            assert emb_len == 384, f"Chunk {ch.section_number} dimension mismatch: {emb_len} != 384"
            print(f"  [PASS] {ch.section_number}: Vector dimensions={emb_len}, sample={emb[:3]}...")

        # 3. Similarity Search via pgvector Cosine Distance on Neon
        print("\n[3/6] Testing Semantic Vector Search using pgvector Cosine Distance...")
        search_query = "autonomous regulatory registry in Nigeria"
        query_vector = embed_query(search_query)
        assert len(query_vector) == 384, "Query vector dimension mismatch."

        ranked_chunks = list(
            LegalProvisionChunk.objects.filter(document=doc)
            .annotate(cosine_dist=CosineDistance('embedding', query_vector))
            .order_by('cosine_dist')
        )

        top_chunk = ranked_chunks[0]
        print(f"  Query: '{search_query}'")
        print(f"  Top Match: {top_chunk.section_number} ({top_chunk.section_title})")
        print(f"  Cosine Distance: {top_chunk.cosine_dist:.6f}")
        assert top_chunk.section_number == "Section 2", (
            f"Expected Section 2 to rank highest for registry query, got {top_chunk.section_number}"
        )
        assert top_chunk.cosine_dist < 0.8, f"Cosine distance unexpectedly high: {top_chunk.cosine_dist}"
        print("  [PASS] Semantic similarity ranking correctly retrieved Section 2.")

        # 4. PostgreSQL Full-Text Search (Automatic Trigger Validation)
        print("\n[4/6] Verifying PostgreSQL Full-Text Search (Trigger & tsvector)...")
        with connection.cursor() as cur:
            cur.execute("""
                SELECT section_number, search_vector
                FROM rag_legal_provision_chunk
                WHERE document_id = %s
                ORDER BY chunk_index;
            """, [str(doc.id)])
            fts_rows = cur.fetchall()

        for sec_num, s_vec in fts_rows:
            assert s_vec is not None, f"search_vector is NULL for {sec_num}!"
            print(f"  [PASS] {sec_num} tsvector: {str(s_vec)[:60]}...")

        # Run FTS query using PostgreSQL to_tsquery
        with connection.cursor() as cur:
            cur.execute("""
                SELECT section_number, citation_label, ts_rank(search_vector, to_tsquery('english', 'registry & algorithmic')) as rank
                FROM rag_legal_provision_chunk
                WHERE document_id = %s AND search_vector @@ to_tsquery('english', 'registry & algorithmic')
                ORDER BY rank DESC;
            """, [str(doc.id)])
            matching_fts = cur.fetchall()

        assert len(matching_fts) == 1, f"Expected 1 FTS match, got {len(matching_fts)}"
        print(f"  [PASS] PostgreSQL FTS match: {matching_fts[0][0]} ({matching_fts[0][1]}) - Rank={matching_fts[0][2]:.4f}")

        # 5. Quarantine Retrieval Exclusion
        print("\n[5/6] Verifying Quarantine Isolation (Unverified Excluded from Retrieval)...")
        assert doc.is_verified_authentic is False, "Doc should be unverified upon ingestion!"
        assert doc.verification_status == VerificationStatus.PENDING, "Doc status should be PENDING!"

        unverified_results = get_trusted_legal_chunks(query_text="Artificial Intelligence Registry")
        matching_unverified = [c for c in unverified_results if c.document_id == doc.id]
        assert len(matching_unverified) == 0, (
            f"SECURITY BREACH: Quarantined document chunks appeared in retrieval! Found: {len(matching_unverified)}"
        )
        print("  [PASS] Strict quarantine confirmed: 0 chunks returned while PENDING_VERIFICATION.")

        # 6. Verification Admission via verify_statute Command
        print("\n[6/6] Verifying Admission to Knowledge Base after Manual Counsel Audit...")
        call_command(
            'verify_statute',
            document_id=str(doc.id),
            counsel_name="Senior Legal Auditor C. Nwosu, SAN (NBA/29401)",
            reason="Verified against Federal Government Synthetic Gazette No. 99, Vol. 113.",
            notes="Statutory sections 1 and 2 validated for accuracy.",
        )

        doc.refresh_from_db()
        assert doc.is_verified_authentic is True, "Document should now be verified authentic!"
        assert doc.verification_status == VerificationStatus.VERIFIED, "Document status should be VERIFIED_AUTHENTIC!"
        print(f"  [PASS] Status updated: {doc.verification_status} by {doc.verified_by}")

        verified_results = get_trusted_legal_chunks(query_text="Artificial Intelligence Registry")
        matching_verified = [c for c in verified_results if c.document_id == doc.id]
        assert len(matching_verified) == 1, (
            f"Expected exactly 1 chunk returned after verification, got {len(matching_verified)}"
        )
        retrieved_chunk = matching_verified[0]
        print(f"  [PASS] Successfully retrieved verified chunk: {retrieved_chunk.citation_label}")
        print(f"         Content preview: {retrieved_chunk.content[:70]}...")

        print("\n" + "=" * 75)
        print("    ALL 6/6 NEON POSTGRESQL & PGVECTOR INTEGRATION CHECKS PASSED")
        print("=" * 75)

    finally:
        # Always clean up synthetic test data
        cleanup_synthetic_data()
        print("\n[TEARDOWN] All synthetic test records removed. Neon database remains clean.")


def cleanup_synthetic_data():
    """Removes any synthetic test records with code SYNTHETIC_CODE."""
    deleted_docs, _ = LegalSourceDocument.objects.filter(statute_family_code=SYNTHETIC_CODE).delete()
    if deleted_docs > 0:
        print(f"  Cleaned up {deleted_docs} synthetic record(s) from previous test runs.")


if __name__ == '__main__':
    run_pipeline_verification()
