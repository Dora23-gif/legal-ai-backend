"""
Exhaustive Live Neon Verification:
1. Direct physical schema comparison: Django model fields vs Neon information_schema.columns.
2. All 4 two-way verification state combinations on live Neon.
3. Complete cleanup and 0-row verification on Neon.
"""
import os
import sys
import time
import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from django.db import connection
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
from rag.services.temporal_router import format_statutory_lineage_notice
from rag.services.hybrid_retriever import hybrid_legal_search

SYNTHETIC_PREFIX = "SYN_4STATE_"


def ensure_neon_connection():
    for attempt in range(1, 6):
        try:
            print(f"Connecting to Neon PostgreSQL (attempt {attempt}/5)...")
            connection.ensure_connection()
            print("  [CONNECTED] Successfully connected to Neon PostgreSQL.")
            return
        except Exception as e:
            if attempt == 5:
                raise e
            print(f"  Attempt {attempt} failed ({e}). Retrying in 2s...")
            time.sleep(2)


def direct_schema_comparison():
    print("\n" + "=" * 80)
    print("  DIRECT PHYSICAL SCHEMA AUDIT: DJANGO MODELS VS NEON POSTGRESQL")
    print("=" * 80)

    tables_to_check = [
        ('rag_legal_source_document', LegalSourceDocument),
        ('rag_statute_amendment_record', StatuteAmendmentRecord),
        ('rag_legal_provision_chunk', LegalProvisionChunk),
    ]

    with connection.cursor() as cursor:
        for table_name, model in tables_to_check:
            print(f"\n--- Table: {table_name} (Model: {model.__name__}) ---")
            cursor.execute("""
                SELECT column_name, data_type, character_maximum_length, is_nullable, udt_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, [table_name])
            neon_cols = {row[0]: row for row in cursor.fetchall()}
            
            # Django model fields (excluding m2m or reverse relations)
            django_fields = {}
            for f in model._meta.get_fields():
                if f.concrete and not f.many_to_many:
                    col_name = f.column
                    django_fields[col_name] = f

            print(f"  Neon Physical Columns: {len(neon_cols)} | Django Concrete Fields: {len(django_fields)}")

            # Check for any missing columns
            missing_in_db = set(django_fields.keys()) - set(neon_cols.keys())
            missing_in_model = set(neon_cols.keys()) - set(django_fields.keys())

            assert not missing_in_db, f"Columns missing in Neon DB: {missing_in_db}"
            assert not missing_in_model, f"Columns in Neon DB not in Django model: {missing_in_model}"

            print("  [MATCH] All Django model fields exist as physical columns in Neon PostgreSQL.")
            print("  [MATCH] All Neon PostgreSQL physical columns map to Django model fields.")

            # Sample 5 columns with data types
            for col, f in list(django_fields.items())[:6]:
                db_info = neon_cols[col]
                print(f"    - Field '{f.name}' -> Column '{col}': DB Type={db_info[1]} ({db_info[4]}), Nullable={db_info[3]}")


def run_four_state_verification():
    print("\n" + "=" * 80)
    print("  TWO-WAY VERIFICATION TEST: ALL 4 COMBINATIONS ON LIVE NEON")
    print("=" * 80)

    # Pre-test cleanup
    LegalSourceDocument.objects.filter(statute_family_code__startswith=SYNTHETIC_PREFIX).delete()

    # Create base statute and amending Act
    base_doc = LegalSourceDocument.objects.create(
        statute_family_code=f"{SYNTHETIC_PREFIX}BASE",
        title="Synthetic Base Statute 2020",
        short_title="Synthetic Base Act",
        jurisdiction=Jurisdiction.FEDERAL,
        category=DocumentCategory.PRIMARY_ACT,
        version_label="As Enacted",
        is_current_version=True,
        effective_from=datetime.date(2020, 8, 7),
        publisher="Federal Government Printer",
        original_filename="base.pdf",
        sha256_checksum="1" * 64,
        is_verified_authentic=False,
        verification_status=VerificationStatus.PENDING,
        extraction_status=ExtractionStatus.COMPLETE_TEXT,
    )
    chunk = LegalProvisionChunk.objects.create(
        document=base_doc,
        section_number="Section 18",
        section_title="Right to incorporate",
        content="Any two or more persons may form a company under this Act.",
        citation_label="Section 18, Synthetic Base Act",
        embedding=[0.01] * 384,
        token_count=14,
        chunk_index=0,
    )
    amending_doc = LegalSourceDocument.objects.create(
        statute_family_code=f"{SYNTHETIC_PREFIX}AMEND",
        title="Synthetic Amending Act 2022",
        short_title="Synthetic Amending Act",
        jurisdiction=Jurisdiction.FEDERAL,
        category=DocumentCategory.PRIMARY_ACT,
        version_label="As Enacted",
        is_current_version=True,
        effective_from=datetime.date(2023, 2, 8),
        publisher="Federal Government Printer",
        original_filename="amending.pdf",
        sha256_checksum="2" * 64,
        is_verified_authentic=False,
        verification_status=VerificationStatus.PENDING,
        extraction_status=ExtractionStatus.COMPLETE_TEXT,
    )
    amend_rec = StatuteAmendmentRecord.objects.create(
        base_statute=base_doc,
        amending_act=amending_doc,
        amendment_title="BFA Amendments to Base Act",
        commencement_date=datetime.date(2023, 2, 8),
        affected_sections=["Section 18"],
        summary_of_changes="Permits single-member company formation.",
        official_gazette_ref="Gazette No. 32, Vol. 110",
    )
    chunk.amendment_record = amend_rec
    chunk.save()

    def reload_chunk():
        return LegalProvisionChunk.objects.select_related(
            'document', 'amendment_record__base_statute', 'amendment_record__amending_act'
        ).get(id=chunk.id)

    # -----------------------------------------------------------------
    # State 1: Base=False (Unverified), Amending=False (Unverified)
    # -----------------------------------------------------------------
    print("\n[COMBINATION 1/4] Base Statute: UNVERIFIED | Amending Act: UNVERIFIED")
    base_doc.is_verified_authentic = False
    base_doc.verification_status = VerificationStatus.PENDING
    base_doc.save()
    amending_doc.is_verified_authentic = False
    amending_doc.verification_status = VerificationStatus.PENDING
    amending_doc.save()

    ch1 = reload_chunk()
    notice1 = format_statutory_lineage_notice(ch1)
    # Retrieval search
    results1 = hybrid_legal_search(query="form a company", limit=5)
    chunk_in_retrieval1 = any(r[0].id == chunk.id for r in results1)

    print(f"  Base in Retrieval:      {chunk_in_retrieval1} (Expected: False - strictly quarantined)")
    print(f"  Amendment Active:       {notice1['is_amended']} (Expected: False)")
    print(f"  Provision Nature:       {notice1['provision_nature']} (Expected: ORIGINAL)")
    print(f"  Lineage Notice Text:    '{notice1['lineage_text']}' (Expected: empty)")
    assert not chunk_in_retrieval1, "State 1 failed: unverified base leaked into retrieval!"
    assert not notice1['is_amended'], "State 1 failed: amendment was marked active!"
    assert notice1['lineage_text'] == "", "State 1 failed: lineage notice emitted!"
    print("  [PASS] Combination 1 verified: Entirely quarantined and dormant.")

    # -----------------------------------------------------------------
    # State 2: Base=False (Unverified), Amending=True (Verified)
    # -----------------------------------------------------------------
    print("\n[COMBINATION 2/4] Base Statute: UNVERIFIED | Amending Act: VERIFIED")
    base_doc.is_verified_authentic = False
    base_doc.verification_status = VerificationStatus.PENDING
    base_doc.save()
    amending_doc.is_verified_authentic = True
    amending_doc.verification_status = VerificationStatus.VERIFIED
    amending_doc.save()

    ch2 = reload_chunk()
    notice2 = format_statutory_lineage_notice(ch2)
    results2 = hybrid_legal_search(query="form a company", limit=5)
    chunk_in_retrieval2 = any(r[0].id == chunk.id for r in results2)

    print(f"  Base in Retrieval:      {chunk_in_retrieval2} (Expected: False - base is unverified)")
    print(f"  Amendment Active:       {notice2['is_amended']} (Expected: False - base unverified)")
    print(f"  Provision Nature:       {notice2['provision_nature']} (Expected: ORIGINAL)")
    print(f"  Lineage Notice Text:    '{notice2['lineage_text']}' (Expected: empty)")
    assert not chunk_in_retrieval2, "State 2 failed: unverified base leaked into retrieval!"
    assert not notice2['is_amended'], "State 2 failed: amendment was marked active when base unverified!"
    assert notice2['lineage_text'] == "", "State 2 failed: lineage notice emitted!"
    print("  [PASS] Combination 2 verified: Base excluded; amendment relationship dormant.")

    # -----------------------------------------------------------------
    # State 3: Base=True (Verified), Amending=False (Unverified)
    # -----------------------------------------------------------------
    print("\n[COMBINATION 3/4] Base Statute: VERIFIED | Amending Act: UNVERIFIED")
    base_doc.is_verified_authentic = True
    base_doc.verification_status = VerificationStatus.VERIFIED
    base_doc.save()
    amending_doc.is_verified_authentic = False
    amending_doc.verification_status = VerificationStatus.PENDING
    amending_doc.save()

    ch3 = reload_chunk()
    notice3 = format_statutory_lineage_notice(ch3)
    results3 = hybrid_legal_search(query="Section 18 form a company", limit=5)
    chunk_in_retrieval3 = any(r[0].id == chunk.id for r in results3)

    print(f"  Base in Retrieval:      {chunk_in_retrieval3} (Expected: True - base is verified)")
    print(f"  Amendment Active:       {notice3['is_amended']} (Expected: False - amending act unverified)")
    print(f"  Provision Nature:       {notice3['provision_nature']} (Expected: ORIGINAL)")
    print(f"  Lineage Notice Text:    '{notice3['lineage_text']}' (Expected: empty)")
    assert chunk_in_retrieval3, "State 3 failed: verified base not found in retrieval!"
    assert not notice3['is_amended'], "State 3 failed: amendment active while amending act unverified!"
    assert notice3['lineage_text'] == "", "State 3 failed: unverified amendment emitted notice!"
    print("  [PASS] Combination 3 verified: Base admitted, but unverified amendment strictly suppressed.")

    # -----------------------------------------------------------------
    # State 4: Base=True (Verified), Amending=True (Verified)
    # -----------------------------------------------------------------
    print("\n[COMBINATION 4/4] Base Statute: VERIFIED | Amending Act: VERIFIED")
    base_doc.is_verified_authentic = True
    base_doc.verification_status = VerificationStatus.VERIFIED
    base_doc.save()
    amending_doc.is_verified_authentic = True
    amending_doc.verification_status = VerificationStatus.VERIFIED
    amending_doc.save()

    ch4 = reload_chunk()
    notice4 = format_statutory_lineage_notice(ch4)
    results4 = hybrid_legal_search(query="Section 18 form a company", limit=5)
    chunk_in_retrieval4 = any(r[0].id == chunk.id for r in results4)

    print(f"  Base in Retrieval:      {chunk_in_retrieval4} (Expected: True - base is verified)")
    print(f"  Amendment Active:       {notice4['is_amended']} (Expected: True - BOTH verified)")
    print(f"  Provision Nature:       {notice4['provision_nature']} (Expected: AMENDED)")
    print(f"  Amending Act Title:     {notice4['amending_act_title']} (Expected: Synthetic Amending Act)")
    print(f"  Lineage Notice Text:    {notice4['lineage_text']}")
    assert chunk_in_retrieval4, "State 4 failed: verified base not found in retrieval!"
    assert notice4['is_amended'], "State 4 failed: amendment not active when both verified!"
    assert notice4['provision_nature'] == 'AMENDED', "State 4 failed: provision nature not AMENDED!"
    assert "AMENDMENT NOTICE: Section 18 was amended by Synthetic Amending Act" in notice4['lineage_text']
    print("  [PASS] Combination 4 verified: BOTH verified -> Amendment relationship fully active.")

    # -----------------------------------------------------------------
    # Cleanup & Verification
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  CLEANUP AUDIT ON LIVE NEON POSTGRESQL")
    print("=" * 80)
    count, _ = LegalSourceDocument.objects.filter(statute_family_code__startswith=SYNTHETIC_PREFIX).delete()
    print(f"Deleted {count} synthetic test record(s) and cascading objects.")

    doc_cnt = LegalSourceDocument.objects.count()
    chunk_cnt = LegalProvisionChunk.objects.count()
    amend_cnt = StatuteAmendmentRecord.objects.count()

    print(f"Neon Live Count - LegalSourceDocument:    {doc_cnt}")
    print(f"Neon Live Count - LegalProvisionChunk:     {chunk_cnt}")
    print(f"Neon Live Count - StatuteAmendmentRecord: {amend_cnt}")

    assert doc_cnt == 0, f"LEAK: {doc_cnt} documents remain!"
    assert chunk_cnt == 0, f"LEAK: {chunk_cnt} chunks remain!"
    assert amend_cnt == 0, f"LEAK: {amend_cnt} amendments remain!"
    print("[CLEANUP AUDIT CONFIRMED] Exactly ZERO records remain in Neon PostgreSQL.")


if __name__ == '__main__':
    ensure_neon_connection()
    direct_schema_comparison()
    run_four_state_verification()
