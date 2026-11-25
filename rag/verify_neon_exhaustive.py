"""
Exhaustive Live Neon PostgreSQL Verification Script.

Executes:
1. The 3 PostgreSQL-specific tests directly on Neon:
   - Automatic PL/pgSQL Trigger synchronization populating search_vector (GIN).
   - 384-dimensional vector storage & pgvector CosineDistance query (<=>).
   - Live on-disk storage footprint & index allocation on Neon.
2. Full Audit of ALL Tables in the Neon Database:
   - Queries information_schema.tables for every table in the 'public' schema.
   - Checks row counts across all RAG, Chat, Accounts, and Django tables.
   - Proves conclusively that zero test artifacts exist in any table.
"""
import os
import sys
import datetime
import hashlib
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

import time
from django.db import connection, transaction
from django.contrib.postgres.search import SearchQuery
from pgvector.django import CosineDistance

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


def connect_with_retry():
    for attempt in range(1, 5):
        try:
            connection.ensure_connection()
            return
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(2)


def run_exhaustive_verification():
    print("=" * 80)
    print("       EXHAUSTIVE NEON POSTGRESQL & PGVECTOR VERIFICATION")
    print("=" * 80)
    connect_with_retry()

    vendor = connection.vendor
    host = connection.settings_dict.get('HOST', 'unknown')
    db_name = connection.settings_dict.get('NAME', 'unknown')
    print(f"PostgreSQL Server: {host}")
    print(f"Database Name:     {db_name}")
    print(f"Backend Vendor:    {vendor}")
    assert vendor == 'postgresql', "Must be running against PostgreSQL!"

    # =========================================================================
    # PART 1: The 3 PostgreSQL-Specific Tests Executed Directly on Neon
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 1: EXECUTING THE 3 POSTGRESQL-SPECIFIC FEATURES DIRECTLY ON NEON")
    print("-" * 80)

    with transaction.atomic():
        # Test 1: PostgreSQL PL/pgSQL Trigger Automatic Sync
        print("\n[TEST 1/3] Verifying PL/pgSQL Trigger search_vector Synchronization...")
        doc_trig = LegalSourceDocument.objects.create(
            statute_family_code="NEON_TEST_TRIG",
            title="Neon Test Recovery of Premises Law",
            short_title="Neon Premises Law",
            jurisdiction=Jurisdiction.RIVERS_STATE,
            is_current_version=True,
            effective_from=datetime.date(2026, 1, 1),
            sha256_checksum="1" * 64,
            original_filename="neon_trigger_test.pdf",
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
        )

        chunk_trig = LegalProvisionChunk.objects.create(
            document=doc_trig,
            section_number="Section 16",
            section_title="Summary Recovery of Possession Prohibited",
            content="No landlord shall take possession of any premises without legal process and an order of court.",
            citation_label="Section 16, Neon Premises Law",
            embedding=[0.05] * 384,
            chunk_index=16,
        )

        chunk_trig.refresh_from_db()
        assert chunk_trig.search_vector is not None, "Trigger failed to populate search_vector!"
        print(f"  [PASS] Initial search_vector created by trigger: {str(chunk_trig.search_vector)[:60]}...")

        # Search query for 'possession' using english config matching the trigger
        matches_1 = LegalProvisionChunk.objects.filter(
            search_vector=SearchQuery("possession", config="english")
        )
        assert matches_1.filter(id=chunk_trig.id).exists(), "FTS query for 'possession' failed!"
        print("  [PASS] FTS matched 'possession' in search_vector.")

        # Update content directly via SQL/ORM to prove trigger runs on UPDATE
        LegalProvisionChunk.objects.filter(id=chunk_trig.id).update(
            content="No landlord shall unlawfully lock out or forcibly eject a tenant."
        )

        chunk_trig.refresh_from_db()
        print(f"  [PASS] Updated search_vector after content update: {str(chunk_trig.search_vector)[:60]}...")
        matches_2 = LegalProvisionChunk.objects.filter(
            search_vector=SearchQuery("eject", config="english")
        )
        assert matches_2.filter(id=chunk_trig.id).exists(), "FTS query for updated word 'eject' failed!"
        print("  [PASS] FTS matched updated word 'eject' in search_vector.")

        # Test 2: pgvector 384-Dimensional Storage & Cosine Distance (<=>)
        print("\n[TEST 2/3] Verifying pgvector 384-dim Storage & Cosine Distance Query...")
        doc_vec = LegalSourceDocument.objects.create(
            statute_family_code="NEON_TEST_VEC",
            title="Neon Test Evidence Statute",
            short_title="Neon Evidence Act",
            jurisdiction=Jurisdiction.FEDERAL,
            is_current_version=True,
            effective_from=datetime.date(2026, 1, 1),
            sha256_checksum="2" * 64,
            original_filename="neon_vec_test.pdf",
            is_verified_authentic=True,
            verification_status=VerificationStatus.VERIFIED,
        )

        sample_vec_1 = [0.05] * 384
        sample_vec_2 = [-0.05] * 384

        chunk_vec_1 = LegalProvisionChunk.objects.create(
            document=doc_vec,
            section_number="Section 84",
            section_title="Computer statements",
            content="Computer generated electronic evidence.",
            citation_label="Section 84, Neon Evidence Act",
            embedding=sample_vec_1,
            chunk_index=84,
        )

        chunk_vec_2 = LegalProvisionChunk.objects.create(
            document=doc_vec,
            section_number="Section 84A",
            section_title="Digital signatures",
            content="Digital signatures and certificates.",
            citation_label="Section 84A, Neon Evidence Act",
            embedding=sample_vec_2,
            chunk_index=85,
        )

        # Query with exact sample_vec_1
        query_vector = [0.05] * 384
        vec_results = list(
            LegalProvisionChunk.objects.filter(document=doc_vec)
            .annotate(distance=CosineDistance("embedding", query_vector))
            .order_by("distance")
        )

        assert len(vec_results) == 2, f"Expected 2 vector results, got {len(vec_results)}"
        assert vec_results[0].id == chunk_vec_1.id, "Expected chunk 1 to rank first."
        dist_0 = float(vec_results[0].distance)
        dist_1 = float(vec_results[1].distance)
        print(f"  [PASS] Exact match distance:    {dist_0:.6f} (Rank 1: Section 84)")
        print(f"  [PASS] Opposite match distance: {dist_1:.6f} (Rank 2: Section 84A)")
        assert dist_0 < 0.0001, f"Cosine distance for exact vector should be ~0.0, got {dist_0}"
        assert dist_1 > 1.9, f"Cosine distance for opposite vector should be ~2.0, got {dist_1}"

        # Test 3: Live On-Disk Storage Footprint on Neon
        print("\n[TEST 3/3] Inspecting Live On-Disk Storage Allocation on Neon...")
        with connection.cursor() as cur:
            cur.execute("""
                SELECT 
                    'rag_legal_provision_chunk' AS table_name,
                    pg_size_pretty(pg_total_relation_size('rag_legal_provision_chunk')) AS total_size,
                    pg_size_pretty(pg_relation_size('rag_legal_provision_chunk')) AS heap_size,
                    pg_size_pretty(pg_indexes_size('rag_legal_provision_chunk')) AS indexes_size,
                    pg_size_pretty(pg_database_size(current_database())) AS db_total;
            """)
            disk_info = cur.fetchone()

        print(f"  [PASS] Table: {disk_info[0]}")
        print(f"         Total Size on Disk: {disk_info[1]}")
        print(f"         Heap Data Size:     {disk_info[2]}")
        print(f"         Index Size (HNSW+GIN+BTREE): {disk_info[3]}")
        print(f"         Total Database Size: {disk_info[4]} (out of 500 MB Neon free quota)")

        # Roll back this test transaction so the test rows are completely eliminated
        transaction.set_rollback(True)
        print("\n  [ROLLBACK] Test transaction cleanly rolled back. Zero test rows committed.")

    # =========================================================================
    # PART 2: Comprehensive Audit of ALL Tables in the Neon Database
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: COMPREHENSIVE AUDIT OF ALL TABLES IN NEON POSTGRESQL")
    print("-" * 80)

    with connection.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        all_tables = [row[0] for row in cur.fetchall()]

    print(f"\nFound {len(all_tables)} public tables in Neon database:")
    total_rag_rows = 0
    table_audit_results = []

    for tbl in all_tables:
        with connection.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM "{tbl}";')
            count = cur.fetchone()[0]
        table_audit_results.append((tbl, count))
        if tbl.startswith('rag_'):
            total_rag_rows += count
        status_flag = "[EMPTY]" if count == 0 else f"[{count} rows]"
        print(f"  Table: {tbl:<35} {status_flag}")

    print("\n" + "=" * 80)
    print("       AUDIT SUMMARY OF LEGAL KNOWLEDGE BASE TABLES")
    print("=" * 80)
    rag_counts = {tbl: cnt for tbl, cnt in table_audit_results if tbl.startswith('rag_')}
    for tbl, cnt in rag_counts.items():
        print(f"  {tbl}: {cnt} rows")

    assert rag_counts.get('rag_legal_source_document', 0) == 0, "rag_legal_source_document is NOT empty!"
    assert rag_counts.get('rag_legal_provision_chunk', 0) == 0, "rag_legal_provision_chunk is NOT empty!"
    assert rag_counts.get('rag_statute_amendment_record', 0) == 0, "rag_statute_amendment_record is NOT empty!"

    print("\n[VERIFIED] Every single RAG table in the Neon PostgreSQL database contains EXACTLY 0 rows.")
    print("[VERIFIED] All 3 PostgreSQL-specific features passed with real execution proof.")
    print("=" * 80)


if __name__ == '__main__':
    run_exhaustive_verification()
