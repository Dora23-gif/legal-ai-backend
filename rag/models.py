import uuid
from django.db import models
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex
from pgvector.django import VectorField


class Jurisdiction(models.TextChoices):
    FEDERAL = 'FEDERAL', 'Federal Republic of Nigeria'
    RIVERS_STATE = 'RIVERS_STATE', 'Rivers State'
    LAGOS_STATE = 'LAGOS_STATE', 'Lagos State'
    FCT_ABUJA = 'FCT_ABUJA', 'Federal Capital Territory (Abuja)'


class DocumentCategory(models.TextChoices):
    CONSTITUTION = 'CONSTITUTION', 'Constitution'
    PRIMARY_ACT = 'PRIMARY_ACT', 'Act of the National Assembly / State Law'
    SUBSIDIARY = 'SUBSIDIARY', 'Regulations / Orders / By-laws'
    STATUTORY_INSTRUMENT = 'STATUTORY_INSTRUMENT', 'Statutory Instrument / Rule'


class DocumentStatus(models.TextChoices):
    ACTIVE = 'ACTIVE', 'Active / In Force'
    AMENDED = 'AMENDED', 'Amended (Provisions Modified)'
    REPEALED = 'REPEALED', 'Repealed / No Longer in Force'
    PENDING = 'PENDING', 'Enacted but Awaiting Commencement'


class SourceType(models.TextChoices):
    OFFICIAL_GAZETTE = 'OFFICIAL_GAZETTE', 'Official Government Gazette'
    OFFICIAL_AGENCY_HOSTED_REPRINT = 'OFFICIAL_AGENCY_HOSTED_REPRINT', 'Official Agency-Hosted Reprint'
    NASS = 'NASS', 'National Assembly Official Bill/Act Print'
    GOVT_PRINTER = 'GOVT_PRINTER', 'Federal/State Government Printing Press'
    REGULATOR = 'REGULATOR', 'Regulatory Commission (e.g. CAC, NDPC, FCCPC)'
    OFFICIAL_REVISION = 'OFFICIAL_REVISION', 'Laws of the Federation (LFN) / State Revision'


class VerificationStatus(models.TextChoices):
    PENDING = 'PENDING_VERIFICATION', 'Pending Manual Legal Audit'
    VERIFIED = 'VERIFIED_AUTHENTIC', 'Verified Authentic by Legal Counsel'
    REJECTED = 'REJECTED', 'Rejected / Inauthentic / Tampered'


class ExtractionStatus(models.TextChoices):
    COMPLETE_TEXT = 'COMPLETE_TEXT', 'Complete Digital Text Extracted'
    MIXED_PARTIAL_TEXT = 'MIXED_PARTIAL_TEXT', 'Mixed Digital Text & Scanned Image Pages'
    NEEDS_OCR = 'NEEDS_OCR', 'Scanned Image Only (Requires OCR Pipeline)'
    FAILED = 'FAILED', 'Extraction Failed / Unreadable File'


class LegalSourceDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Statute Lineage & Family Tracking
    statute_family_code = models.CharField(
        max_length=80,
        db_index=True,
        help_text="Shared identifier across all historical versions (e.g. 'FED_CAMA', 'FED_EVIDENCE_ACT', 'RIV_RECOVERY_PREMISES')"
    )
    title = models.CharField(max_length=300, help_text="Full legal title, e.g. 'Companies and Allied Matters Act, 2020'")
    short_title = models.CharField(max_length=120, help_text="e.g. 'CAMA 2020', 'Evidence Act 2011'")
    jurisdiction = models.CharField(max_length=30, choices=Jurisdiction.choices, default=Jurisdiction.FEDERAL)
    category = models.CharField(max_length=30, choices=DocumentCategory.choices, default=DocumentCategory.PRIMARY_ACT)

    # Version Management & Timeline
    version_label = models.CharField(max_length=80, default="As Enacted", help_text="e.g. 'As Enacted', '2023 Amendment', 'Consolidated 2020'")
    is_current_version = models.BooleanField(default=True, help_text="True if this is the active in-force version of this Act/Law")
    effective_from = models.DateField(help_text="Date when this version officially took legal effect (commencement date)")
    effective_to = models.DateField(null=True, blank=True, help_text="Date when this version was amended/repealed (NULL if currently active)")
    superseded_by = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='superseded_versions',
        help_text="Points to the newer Act/Amendment that amended or repealed this document"
    )
    status = models.CharField(max_length=20, choices=DocumentStatus.choices, default=DocumentStatus.ACTIVE)

    # Provenance & Source Integrity Checksums
    publisher = models.CharField(max_length=200, help_text="e.g. Federal Government Press, Rivers State Government Printer, NDPC")
    source_type = models.CharField(max_length=30, choices=SourceType.choices, default=SourceType.OFFICIAL_GAZETTE)
    official_gazette_ref = models.CharField(max_length=255, blank=True, help_text="e.g. Official Gazette No. 124, Vol. 107 of 7th August, 2020")
    source_url = models.URLField(max_length=500, blank=True, null=True, help_text="Official download URL if available online")
    original_filename = models.CharField(max_length=255, help_text="Name of the source document stored in archive")
    sha256_checksum = models.CharField(max_length=64, help_text="SHA-256 cryptographic hash of source file to guarantee zero tampering")

    # Verification Metadata & Quarantine Audit
    is_verified_authentic = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Strict quarantine flag: must be True for document chunks to enter search/retrieval."
    )
    verification_status = models.CharField(
        max_length=30,
        choices=VerificationStatus.choices,
        default=VerificationStatus.PENDING,
        db_index=True,
        help_text="Quarantine audit status. Only VERIFIED_AUTHENTIC documents are trusted."
    )
    verified_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp when legal integrity was audited and verified")
    verified_by = models.CharField(max_length=150, blank=True, help_text="Name or role of person/team who verified source authenticity")
    verification_reason = models.TextField(
        blank=True,
        help_text="Explicit legal rationale or statutory authority recorded by reviewing counsel."
    )
    verification_notes = models.TextField(
        blank=True,
        help_text="Internal counsel notes during verification audit."
    )

    # Extraction Audit & Scanned Page Tracking
    extraction_status = models.CharField(
        max_length=30,
        choices=ExtractionStatus.choices,
        default=ExtractionStatus.COMPLETE_TEXT,
        help_text="Audit of text completeness."
    )
    scanned_pages = models.JSONField(
        default=list,
        blank=True,
        help_text="List of 1-based page numbers detected as scanned images requiring OCR."
    )
    extraction_errors = models.JSONField(
        default=list,
        blank=True,
        help_text="List of extraction warnings, parsing anomalies, or unextracted blocks."
    )
    extractor_version = models.CharField(
        max_length=20,
        default="2.0.0",
        help_text="Version of ingestion pipeline extractor used."
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'rag_legal_source_document'
        ordering = ['-effective_from', 'short_title']
        constraints = [
            # Exactly one current version per statute family code within a jurisdiction
            models.UniqueConstraint(
                fields=['statute_family_code', 'jurisdiction'],
                condition=models.Q(is_current_version=True),
                name='unique_current_version_per_statute_family'
            ),
            # Effective date range validity: effective_to >= effective_from (or NULL)
            models.CheckConstraint(
                condition=models.Q(effective_to__isnull=True) | models.Q(effective_to__gte=models.F('effective_from')),
                name='check_effective_dates_valid'
            ),
        ]
        indexes = [
            models.Index(fields=['is_verified_authentic', 'status', 'is_current_version'], name='idx_doc_ver_status_curr'),
            models.Index(fields=['jurisdiction', 'status', 'is_current_version'], name='idx_doc_jur_status_curr'),
            models.Index(fields=['effective_from', 'effective_to'], name='idx_doc_effective_range'),
            models.Index(fields=['statute_family_code', 'is_current_version'], name='idx_doc_fam_current'),
            models.Index(fields=['sha256_checksum'], name='idx_doc_checksum'),
        ]

    def __str__(self):
        status_tag = "ACTIVE" if self.is_current_version else f"HISTORICAL (until {self.effective_to or 'repealed'})"
        return f"{self.short_title} ({self.version_label}) [{self.jurisdiction}] - {status_tag}"


class StatuteAmendmentRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    base_statute = models.ForeignKey(
        LegalSourceDocument,
        on_delete=models.CASCADE,
        related_name='amendments_received',
        help_text="The parent statute whose provisions were altered"
    )
    amending_act = models.ForeignKey(
        LegalSourceDocument,
        on_delete=models.CASCADE,
        related_name='amendments_made',
        help_text="The legislative Act that enacted the amendments"
    )
    amendment_title = models.CharField(
        max_length=255,
        help_text="e.g. 'Finance Act 2023 Amendments to CAMA 2020'"
    )
    commencement_date = models.DateField(
        help_text="Date when this amendment officially took legal effect"
    )
    affected_sections = models.JSONField(
        default=list,
        help_text="Array of section numbers modified, e.g. ['Section 18', 'Section 84A']"
    )
    summary_of_changes = models.TextField(
        help_text="Concise legal summary of what was substituted, inserted, or repealed"
    )
    official_gazette_ref = models.CharField(
        max_length=255,
        blank=True,
        help_text="Gazette reference of the amending instrument"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'rag_statute_amendment_record'
        ordering = ['-commencement_date']
        indexes = [
            models.Index(fields=['base_statute', 'commencement_date'], name='idx_amend_base_date'),
            models.Index(fields=['amending_act'], name='idx_amend_acting_act'),
        ]

    def __str__(self):
        return f"{self.amending_act.short_title} amending {self.base_statute.short_title} ({self.commencement_date})"


class LegalProvisionChunk(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    document = models.ForeignKey(
        LegalSourceDocument,
        on_delete=models.CASCADE,
        related_name='chunks',
        help_text="Parent Act or Legal Instrument"
    )
    amendment_record = models.ForeignKey(
        StatuteAmendmentRecord,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='amended_chunks',
        help_text="Points to the specific amending Act if this section was inserted or substituted"
    )

    # Statutory Structure Hierarchy
    part_number = models.CharField(max_length=60, blank=True, help_text="e.g. 'Part II', 'Part B'")
    part_title = models.CharField(max_length=255, blank=True, help_text="e.g. 'Incorporation of Companies and Matters Incidental Thereto'")
    chapter_number = models.CharField(max_length=60, blank=True, help_text="e.g. 'Chapter 1'")

    section_number = models.CharField(max_length=50, help_text="e.g. 'Section 18', 'Section 84A', 'First Schedule, Para 3'")
    section_title = models.CharField(max_length=300, help_text="e.g. 'Right to form a company'")

    # Text Content
    content = models.TextField(help_text="Exact statutory text of the section and subsections")
    citation_label = models.CharField(max_length=255, help_text="Standard citation, e.g. 'Section 18, CAMA 2020'")

    # Vector Embedding (FastEmbed bge-small-en-v1.5 produces 384 dimensions)
    embedding = VectorField(dimensions=384, help_text="384-dimensional dense vector embedding")

    # PostgreSQL Full-Text Search Vector (automatically synchronized by DB trigger)
    search_vector = SearchVectorField(null=True, blank=True, help_text="Automatic tsvector for keyword and Boolean legal search")

    # Metadata
    token_count = models.PositiveIntegerField(default=0, help_text="Word/token count of chunk content")
    chunk_index = models.PositiveIntegerField(default=0, help_text="Sequential position of this section within the Act")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'rag_legal_provision_chunk'
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'section_number'], name='idx_chunk_doc_section'),
            models.Index(fields=['citation_label'], name='idx_chunk_citation'),
            GinIndex(fields=['search_vector'], name='idx_chunk_fts_gin'),
        ]

    def save(self, *args, **kwargs):
        # Auto-populate citation label if omitted
        if not self.citation_label and self.document_id:
            self.citation_label = f"{self.section_number}, {self.document.short_title}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.citation_label}: {self.section_title[:50]}"
