"""
Live Verification Script for Neon PostgreSQL RAG Database Infrastructure.
Tests:
1. Active extensions: 'vector' and 'btree_gist'
2. Real-time PostgreSQL trigger populating search_vector
3. 384-dimensional dense vector storage & cosine distance via pgvector
4. Exclusion constraint on overlapping statute date ranges
5. Precise disk storage measurements (table, heap, and index sizes)
6. Clean transaction rollback leaving production database clean
"""
import os
import sys
import uuid
import datetime
import hashlib
from dotenv import load_dotenv

# Ensure backend root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("[ERROR] DATABASE_URL environment variable is missing.")
    sys.exit(1)


def run_neon_verification():
    print("=" * 70)
    print("      LIVE NEON POSTGRESQL RAG INFRASTRUCTURE VERIFICATION")
    print("=" * 70)

    import time
    conn = None
    for attempt in range(1, 4):
        try:
            print(f"Connecting to Neon PostgreSQL (attempt {attempt}/3)...")
            conn = psycopg.connect(DATABASE_URL)
            break
        except Exception as e:
            if attempt == 3:
                raise e
            print(f"Connection retry {attempt} in 2s ({e})...")
            time.sleep(2)

    conn.autocommit = False

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            # 1. Verify Installed Extensions
            print("\n[1/6] Checking PostgreSQL Extensions...")
            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'btree_gist');")
            extensions = {row['extname']: row['extversion'] for row in cur.fetchall()}
            assert 'vector' in extensions, "vector extension is NOT installed!"
            assert 'btree_gist' in extensions, "btree_gist extension is NOT installed!"
            print(f"  [PASS] pgvector version: {extensions['vector']}")
            print(f"  [PASS] btree_gist version: {extensions['btree_gist']}")

            # 2. Verify Database Tables
            print("\n[2/6] Checking RAG Database Tables on Neon...")
            cur.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' AND tablename LIKE 'rag_%'
                ORDER BY tablename;
            """)
            tables = [row['tablename'] for row in cur.fetchall()]
            expected_tables = [
                'rag_legal_provision_chunk',
                'rag_legal_source_document',
                'rag_statute_amendment_record',
            ]
            for t in expected_tables:
                assert t in tables, f"Expected table {t} missing from Neon!"
                print(f"  [PASS] Table verified: {t}")

            # 3. Verify Database Trigger for search_vector
            print("\n[3/6] Verifying PostgreSQL Trigger for search_vector...")
            cur.execute("""
                SELECT trigger_name, event_manipulation, action_statement
                FROM information_schema.triggers
                WHERE event_object_table = 'rag_legal_provision_chunk' 
                  AND trigger_name = 'trg_rag_chunk_search_vector';
            """)
            trigger_row = cur.fetchone()
            assert trigger_row is not None, "trg_rag_chunk_search_vector trigger not found!"
            print(f"  [PASS] Native trigger active: {trigger_row['trigger_name']}")

            # 4. Insert Test Legal Document & Chunk to test Vector + FTS Trigger
            print("\n[4/6] Testing Vector Embedding & Trigger Synchronization...")
            test_doc_id = str(uuid.uuid4())
            test_sha256 = hashlib.sha256(b"Verification Sample Gazette").hexdigest()
            cur.execute("""
                INSERT INTO rag_legal_source_document (
                    id, statute_family_code, title, short_title, jurisdiction, category,
                    version_label, is_current_version, effective_from, status, publisher,
                    source_type, official_gazette_ref, original_filename, sha256_checksum,
                    verified_by, created_at, updated_at
                ) VALUES (
                    %s, 'FED_CAMA', 'Companies and Allied Matters Act, 2020', 'CAMA 2020',
                    'FEDERAL', 'PRIMARY_ACT', 'As Enacted', true, '2020-08-07', 'ACTIVE',
                    'Federal Government Printing Press', 'OFFICIAL_GAZETTE',
                    'Official Gazette No. 124', 'cama_2020.pdf', %s, 'Legal Verification Suite',
                    NOW(), NOW()
                );
            """, (test_doc_id, test_sha256))

            # Test vector: 384 dimensions
            sample_vector_1 = "[" + ",".join(["0.05"] * 384) + "]"
            sample_vector_2 = "[" + ",".join(["-0.05"] * 384) + "]"
            test_chunk_id = str(uuid.uuid4())

            cur.execute("""
                INSERT INTO rag_legal_provision_chunk (
                    id, document_id, part_number, part_title, chapter_number,
                    section_number, section_title, content, citation_label,
                    embedding, token_count, chunk_index, created_at
                ) VALUES (
                    %s, %s, 'Part B', 'Incorporation of Companies', 'Chapter 1',
                    'Section 18', 'Right to form a company',
                    'Any two or more persons may form and incorporate a company with liability limited by shares.',
                    'Section 18, CAMA 2020', %s::vector, 20, 1, NOW()
                ) RETURNING search_vector;
            """, (test_chunk_id, test_doc_id, sample_vector_1))

            returned_tsv = cur.fetchone()['search_vector']
            assert returned_tsv is not None, "search_vector was not populated by trigger!"
            print(f"  [PASS] search_vector automatically generated by trigger: {returned_tsv[:60]}...")

            # 5. Test Vector Cosine Distance
            print("\n[5/7] Testing pgvector Cosine Distance Query...")
            query_vector = "[" + ",".join(["0.05"] * 384) + "]"
            cur.execute("""
                SELECT id, citation_label, (embedding <=> %s::vector) AS cosine_distance
                FROM rag_legal_provision_chunk
                WHERE id = %s;
            """, (query_vector, test_chunk_id))
            vec_row = cur.fetchone()
            dist = float(vec_row['cosine_distance'])
            assert dist < 0.0001, f"Cosine distance mismatch: {dist}"
            print(f"  [PASS] pgvector cosine distance calculation: {dist:.6f} (Exact match to query vector)")

            # 6. Test PostgreSQL GiST Exclusion Constraint on Overlapping Date Ranges
            print("\n[6/7] Testing Date-Range Exclusion Constraint (No Overlaps)...")
            cur.execute("SAVEPOINT test_exclusion_sp;")
            try:
                # Inserting an overlapping date range in FED_CAMA family (e.g. starting 2021 while 2020 has NULL end) must fail!
                cur.execute("""
                    INSERT INTO rag_legal_source_document (
                        id, statute_family_code, title, short_title, jurisdiction, category,
                        version_label, is_current_version, effective_from, effective_to, status, publisher,
                        source_type, official_gazette_ref, original_filename, sha256_checksum,
                        verified_by, created_at, updated_at
                    ) VALUES (
                        gen_random_uuid(), 'FED_CAMA', 'CAMA 2021 Conflicting', 'CAMA 2021',
                        'FEDERAL', 'PRIMARY_ACT', 'Conflicting Range', false, '2021-01-01', NULL, 'ACTIVE',
                        'Federal Government Press', 'OFFICIAL_GAZETTE',
                        'Gazette No. 12', 'cama_conflict.pdf', %s, 'Audit Tester',
                        NOW(), NOW()
                    );
                """, (test_sha256,))
                raise AssertionError("Exclusion constraint FAILED to block overlapping statute dates!")
            except psycopg.errors.ExclusionViolation:
                cur.execute("ROLLBACK TO SAVEPOINT test_exclusion_sp;")
                print("  [PASS] Exclusion constraint 'exclude_overlapping_statute_periods' successfully rejected overlapping dates.")

            # 7. Live Neon Disk Storage Measurement
            print("\n[7/7] Measuring Live On-Disk Storage Footprint on Neon...")
            cur.execute("""
                SELECT 
                    'rag_legal_provision_chunk' AS table_name,
                    pg_size_pretty(pg_total_relation_size('rag_legal_provision_chunk')) AS total_size,
                    pg_size_pretty(pg_relation_size('rag_legal_provision_chunk')) AS heap_size,
                    pg_size_pretty(pg_indexes_size('rag_legal_provision_chunk')) AS indexes_size,
                    pg_total_relation_size('rag_legal_provision_chunk') AS total_bytes,
                    pg_database_size(current_database()) AS db_total_bytes,
                    pg_size_pretty(pg_database_size(current_database())) AS db_total_pretty;
            """)
            disk_row = cur.fetchone()
            print(f"  [PASS] Table 'rag_legal_provision_chunk' Total On-Disk: {disk_row['total_size']}")
            print(f"    - Table Heap (Row Data): {disk_row['heap_size']}")
            print(f"    - Indexes (HNSW + GIN + B-Trees): {disk_row['indexes_size']}")
            print(f"  [PASS] Total Neon Database Storage: {disk_row['db_total_pretty']} (out of 500 MB free quota)")

        # Rollback verification transaction cleanly so test rows don't stay in DB
        conn.rollback()
        print("\n[CLEANUP] Test transaction rolled back. Database remains 100% pristine.")
        print("=" * 70)
        print("        ALL RAG INFRASTRUCTURE VERIFICATIONS PASSED (6/6)")
        print("=" * 70)
    finally:
        conn.close()


if __name__ == '__main__':
    run_neon_verification()
