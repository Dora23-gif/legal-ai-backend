"""
Authoritative Nigerian Legal Source Registry (Phase 4).

Defines authoritative statutory metadata for priority Nigerian legislation:
1. Companies and Allied Matters Act, 2020 (CAMA 2020)
2. Business Facilitation (Miscellaneous Provisions) Act, 2022 (BFA 2022)
3. Evidence Act, 2011
4. Evidence (Amendment) Act, 2023
5. Nigeria Data Protection Act, 2023 (NDPA 2023)
6. Federal Competition and Consumer Protection Act, 2018 (FCCPA 2018)

PROVENANCE STANDARDS & VERIFICATION TAXONOMY:
- ORIGINAL_GAZETTE_AUTHENTIC_PRINT: Original physical print or authenticated electronic
  Gazette issued directly by the Federal Government Printing Press (Lagos/Abuja).
- OFFICIAL_AGENCY_HOSTED_REPRINT: Digital copy hosted by the statutory regulatory agency
  (e.g. CAC, NDPC, FCCPC, PEBEC) for administrative/compliance guidance.
- THIRD_PARTY_REFERENCE_COPY: Digital copy hosted by legal repositories, NGOs, or
  international organizations (e.g. WIPO Lex, PLAC, Judy Legal, ILO NATLEX).
- PENDING_PRINT_AUDIT: Enactment confirmed, but official printed Gazette details
  and Act number remain pending physical inspection.

LEGAL COMMENCEMENT RULES (Interpretation Act, Cap. I23, LFN 2004):
- Section 2(1): An Act is passed when the President assents to the bill or when
  the National Assembly overrides a presidential veto.
- Section 2(2)(a): Where no other commencement date is specified, an Act comes
  into operation on the day it is passed (the date of presidential assent).
"""
from datetime import date
from typing import Dict, Any, List, Optional
from enum import Enum


class MetadataVerificationLevel(str, Enum):
    ORIGINAL_GAZETTE_AUTHENTIC_PRINT = "ORIGINAL_GAZETTE_AUTHENTIC_PRINT"
    OFFICIAL_AGENCY_HOSTED_REPRINT = "OFFICIAL_AGENCY_HOSTED_REPRINT"
    THIRD_PARTY_REFERENCE_COPY = "THIRD_PARTY_REFERENCE_COPY"
    PENDING_PRINT_AUDIT = "PENDING_PRINT_AUDIT"
    # Backwards compatibility aliases for existing test harnesses
    CONFIRMED_GAZETTE_REFERENCE = "OFFICIAL_AGENCY_HOSTED_REPRINT"
    DIRECT_OFFICIAL_GAZETTE = "OFFICIAL_AGENCY_HOSTED_REPRINT"
    SECONDARY_SOURCE_CONFIRMED = "THIRD_PARTY_REFERENCE_COPY"
    AWAITING_FILE_RECEIPT = "PENDING_PRINT_AUDIT"


# Master Registry of Authoritative Nigerian Statutory Instruments
NIGERIAN_LEGAL_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "CAMA-2020": {
        "statute_family_code": "CAMA-2020",
        "official_title": "Companies and Allied Matters Act, 2020",
        "short_title": "CAMA 2020",
        "act_number": "Act No. 3 of 2020",
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "OFFICIAL_AGENCY_HOSTED_REPRINT",
        # 5 Separate Legal Dates & Years:
        "title_year": 2020,
        "enactment_year": 2020,
        "presidential_assent_date": date(2020, 8, 7),  # Assented by President Muhammadu Buhari under s. 2(1) Interpretation Act
        "gazette_publication_date": date(2020, 8, 10),
        "commencement_date": date(2020, 8, 7),         # Express commencement date printed in Act on Page A33: [7th Day of August, 2020]
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "No. 124",
            "volume": "Vol. 107",
            "publication_date": "10th August 2020",
            "government_notice_number": "Govt. Notice No. 119",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette No. 124, Vol. 107 of 10th August, 2020",
            "supplement_part": "Part A",
            "page_range": "pages A1 - A604 (Stated Gazette Citation) / pages A1 - A528 (Actual Electronic Reprint)",
            "cover_masthead_page": "Page A1 (Federal Republic of Nigeria Official Gazette No. 124, Vol. 107, Govt. Notice No. 119)",
            "arrangement_of_sections_pages": "Pages A3 - A32",
            "substantive_enactment_page": "Page A33 (Long Title, Express Commencement [7th Day of August, 2020], Enactment Formula, Part A Section 1)",
            "substantive_concluding_page": "Page A440 (Section 870: Citation as Companies and Allied Matters Act, 2020)",
            "schedules_pages": "Pages A441 - A527 (First Schedule through Fifteenth Schedule)",
            "authentication_page": "Page A527 (Clerk to the National Assembly Certificate, 9th Day of July, 2020)",
            "assent_page": "Page A528 (Presidential Assent by President Muhammadu Buhari GCFR, 7th Day of August, 2020, LS)",
            "pagination_discrepancy_note": (
                "Documented Gazette index/body pagination discrepancy: External citations and NIPC web resource "
                "metadata cite Gazette No. 124 spanning pages A1-A604. The official electronic agency reprint PDF "
                "spans exactly 528 sequential printed pages (A1 to A528), concluding with the Clerk of the National Assembly "
                "Authentication on page A527 and Presidential Assent on page A528. Whether the 76-page difference reflects "
                "physical booklet page formatting, additional Gazette notices/supplementary instruments, or index variations "
                "remains unconfirmed pending direct inspection of the original physical Federal Government Gazette hardcopy."
            ),
            "extraction_completeness_spec": {
                "expected_sections_count": 870,
                "expected_parts_list": [
                    "Part A", "Part B", "Part C", "Part D", "Part E", "Part F", "Part G"
                ],
                "expected_schedules": [
                    "First Schedule", "Second Schedule", "Third Schedule", "Fourth Schedule",
                    "Fifth Schedule", "Sixth Schedule", "Seventh Schedule", "Eighth Schedule",
                    "Ninth Schedule", "Tenth Schedule", "Eleventh Schedule", "Twelfth Schedule",
                    "Thirteenth Schedule", "Fourteenth Schedule", "Fifteenth Schedule"
                ],
            },
            "hosting_institutions": [
                {
                    "name": "Nigerian Investment Promotion Commission (NIPC)",
                    "relationship": "Federal investment promotion agency under Federal Ministry of Industry, Trade and Investment",
                    "url": "https://nipc.gov.ng/assets/resources/companies-and-allied-matters-act-2020/cama-2020-gazetted.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                    "retrieval_status": "VERIFIED_ACTIVE_HTTP_200",
                    "file_size_bytes": 1227065,
                    "sha256": "3723358a6a4d1c582921a5e626a5c215a0bf611efa997ec0fca53763b7887726",
                },
                {
                    "name": "Corporate Affairs Commission (CAC)",
                    "relationship": "Statutory regulatory custodian established under Section 1 of CAMA 2020",
                    "url": "https://www.cac.gov.ng/wp-content/uploads/2020/12/CAMA-NOTE-BOOK-FULL-VERSION.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                    "retrieval_status": "HISTORICAL_PATH_MIGRATED_HTTP_404",
                },
                {
                    "name": "Policy and Legal Advocacy Centre (PLAC)",
                    "relationship": "Independent civic legislative archive / NGO monitoring National Assembly",
                    "url": "https://placng.org/i/wp-content/uploads/2020/11/Companies-and-Allied-Matters-Act-2020.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                    "retrieval_status": "HISTORICAL_PATH_HTTP_404",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Compute SHA-256 hash of the downloaded agency PDF to establish file integrity baseline. "
                "Step 2: Page-by-page textual and visual comparison of extracted sections against the physical "
                "imprint of Federal Government Gazette No. 124 to confirm no editorial omissions or "
                "typographical errors in agency reprints before legal authenticity promotion."
            ),
        },
        # Integrity & Verification Metadata (Populated dynamically on receipt/ingestion into LegalSourceDocument):
        "sha256_checksum": None,  # Computed from verified file bytes upon receipt
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify file integrity "
            "and detect any subsequent modification; (2) Page-by-page textual and visual comparison of the "
            "extracted content against the physical Gazette or a verified scan to confirm textual fidelity. "
            "Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.OFFICIAL_AGENCY_HOSTED_REPRINT,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted (Pre-BFA 2022)",
        "is_current_version": False,  # Amended by BFA 2022
        "amendment_relationships": [
            {
                "relationship_type": "AMENDED_BY",
                "amending_act_family": "BFA-2022",
                "amending_act_title": "Business Facilitation (Miscellaneous Provisions) Act, 2022",
                "affected_sections": [
                    "Section 18", "Section 27", "Section 78", "Section 222",
                    "Section 240", "Section 275", "Section 374", "Section 572",
                ],
                "effective_date": date(2023, 2, 8),
                "summary": "Amended provisions on single-member companies, corporate capacity, electronic meetings, and insolvency thresholds.",
            }
        ],
    },

    "BFA-2022": {
        "statute_family_code": "BFA-2022",
        "official_title": "Business Facilitation (Miscellaneous Provisions) Act, 2022",
        "short_title": "BFA 2022",  # 2022 statutory short title preserved
        "act_number": "Act No. 5 of 2022",  # Designated as Act No. 5 of 2022; assented 8 Feb 2023
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "OFFICIAL_GAZETTE",
        # 5 Separate Legal Dates & Years:
        "title_year": 2022,
        "enactment_year": 2022,
        "presidential_assent_date": date(2023, 2, 8),  # Assented by President Muhammadu Buhari
        "gazette_publication_date": date(2023, 2, 17),
        "commencement_date": date(2023, 2, 8),         # Default commencement under s. 2(2)(a) Interpretation Act
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "No. 32",
            "volume": "Vol. 110",
            "publication_date": "17th February 2023",
            "government_notice_number": "Govt. Notice No. 19 (Pending Direct Cover Verification)",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette No. 32, Vol. 110 of 17th February, 2023",
            "supplement_part": "Part A",
            "page_range": "pages A155 - A188 (Pending Direct Cover Verification)",
            "title_page": "Page A155 (Long Title, Objectives, Application - Pending Direct Cover Verification)",
            "assent_commencement_page": "Page A188 (Assent Signature, Commencement Clause, Schedule of Enactments - Pending Direct Cover Verification)",
            "hosting_institutions": [
                {
                    "name": "Presidential Enabling Business Environment Council (PEBEC)",
                    "relationship": "Special presidential initiative under the Presidency that sponsored the legislation",
                    "url": "https://pebec.gov.ng/sites/default/files/2023-02/Business-Facilitation-Act-2022.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
                {
                    "name": "World Intellectual Property Organization (WIPO)",
                    "relationship": "United Nations specialized agency holding official member state legislative deposits",
                    "url": "https://www.wipo.int/edocs/lexdocs/laws/en/ng/ng080en.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
                {
                    "name": "Nigerian Investment Promotion Commission (NIPC)",
                    "relationship": "Federal investment promotion agency",
                    "url": "https://nipc.gov.ng/wp-content/uploads/2023/02/Business-Facilitation-Act-2022.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Compute SHA-256 hash of the downloaded PEBEC PDF to establish file integrity baseline. "
                "Step 2: Perform direct cover and print audit of Official Gazette No. 32 to verify Government Notice "
                "number and page range (A155-A188). Step 3: Page-by-page textual and visual comparison to verify "
                "that the digital copy faithfully incorporates the exact schedule of amendments across all 21 "
                "amended business statutes as printed in the authentic Gazette."
            ),
        },
        # Integrity & Verification Metadata (Dynamic upon receipt):
        "sha256_checksum": None,
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify "
            "file integrity and detect any subsequent modification; (2) Page-by-page textual "
            "and visual comparison of the extracted content against the physical Gazette or a "
            "verified scan to confirm textual fidelity. Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.PENDING_PRINT_AUDIT,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted",
        "is_current_version": True,
        # Multi-Statute Amending Capability:
        "amendment_relationships": [
            {
                "relationship_type": "AMENDS_BASE_STATUTE",
                "base_statute_family": "CAMA-2020",
                "base_statute_title": "Companies and Allied Matters Act, 2020",
                "affected_sections": [
                    "Section 18", "Section 27", "Section 78", "Section 222",
                    "Section 240", "Section 275", "Section 374", "Section 572",
                ],
                "effective_date": date(2023, 2, 8),
                "summary": "Amended CAMA 2020 to facilitate business operations, permit virtual general meetings for private companies, and modernize company share capital requirements.",
            },
            {
                "relationship_type": "AMENDS_BASE_STATUTE",
                "base_statute_family": "IMMIGRATION-ACT-2015",
                "base_statute_title": "Immigration Act, 2015",
                "affected_sections": ["Section 20", "Section 36"],
                "effective_date": date(2023, 2, 8),
                "summary": "Amended entry and visa processing requirements to foster ease of doing business.",
            },
        ],
    },

    "EVIDENCE-ACT-2011": {
        "statute_family_code": "EVIDENCE-ACT-2011",
        "official_title": "Evidence Act, 2011",
        "short_title": "Evidence Act 2011",
        "act_number": "Act No. 18 of 2011",
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "OFFICIAL_GAZETTE",
        # 5 Separate Legal Dates & Years:
        "title_year": 2011,
        "enactment_year": 2011,
        "presidential_assent_date": date(2011, 6, 3),  # Signed by President Goodluck E. Jonathan, GCFR on 3 June 2011
        "gazette_publication_date": date(2011, 6, 21),
        "commencement_date": date(2011, 6, 3),         # Express commencement: the Act itself prints "[3rd Day of June, 2011]"
        "assent_authentication_notes": (
            "Clerk of the National Assembly certified the bill on 2 June 2011. "
            "President Goodluck E. Jonathan, GCFR, assented on 3 June 2011. "
            "The Act itself prints '[3rd Day of June, 2011]' as its express commencement date. "
            "Section 2(2)(a) of the Interpretation Act (Cap. I23, LFN 2004) is the default "
            "commencement rule and is not required to establish this date."
        ),
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "No. 80",
            "volume": "Vol. 98",
            "publication_date": "21st June 2011",
            "government_notice_number": "Govt. Notice No. 162",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette No. 80, Vol. 98 of 21st June, 2011",
            "supplement_part": "Part A",
            "page_range": "pages A291 - A371",
            "title_page": "Page A291 (Long Title, Part I: Relevancy of Facts)",
            "assent_commencement_page": "Page A371 (Clerk Certificate 2 June 2011; Presidential Assent 3 June 2011)",
            "hosting_institutions": [
                {
                    "name": "Policy and Legal Advocacy Centre (PLAC)",
                    "relationship": "Civic legislative advocacy body; hosts scanned digital copy of Gazette No. 80",
                    "url": "https://placng.org/i/wp-content/uploads/2019/12/Evidence-Act-2011.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
                {
                    "name": "World Intellectual Property Organization (WIPO)",
                    "relationship": "UN specialized agency member state legal repository",
                    "url": "https://www.wipo.int/edocs/lexdocs/laws/en/ng/ng036en.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
                {
                    "name": "Refworld (UN High Commissioner for Refugees)",
                    "relationship": "UNHCR global legal database repository",
                    "url": "https://www.refworld.org/pdfid/4e242d542.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Compute SHA-256 hash of the downloaded PLAC PDF to establish file integrity baseline. "
                "Step 2: Page-by-page textual and visual comparison of electronic evidence provisions "
                "(Sections 84, 84A-E) against authenticated copy of Gazette No. 80 to ensure authentic "
                "baseline text prior to applying 2023 amendments."
            ),
        },
        # Integrity & Verification Metadata (Dynamic upon receipt):
        "sha256_checksum": None,
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify "
            "file integrity and detect any subsequent modification; (2) Page-by-page textual "
            "and visual comparison of the extracted content against the physical Gazette or a "
            "verified scan to confirm textual fidelity. Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.THIRD_PARTY_REFERENCE_COPY,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted (Pre-2023 Amendment)",
        "is_current_version": False,  # Amended by Evidence (Amendment) Act 2023
        "amendment_relationships": [
            {
                "relationship_type": "AMENDED_BY",
                "amending_act_family": "EVIDENCE-AMENDMENT-2023",
                "amending_act_title": "Evidence (Amendment) Act, 2023",
                "affected_sections": [
                    "Section 84", "Section 84A", "Section 84B", "Section 84C",
                    "Section 84D", "Section 84E", "Section 93", "Section 108",
                ],
                "effective_date": date(2023, 6, 12),
                "summary": "Admissibility of electronic records, electronic signatures, digital oaths, and audio-visual testimony.",
            }
        ],
    },

    "EVIDENCE-AMENDMENT-2023": {
        "statute_family_code": "EVIDENCE-AMENDMENT-2023",
        "official_title": "Evidence (Amendment) Act, 2023",
        "short_title": "Evidence (Amendment) Act 2023",
        "act_number": None,  # Left pending: requires direct verification from physical/authenticated Gazette print
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "OFFICIAL_GAZETTE",
        # 5 Separate Legal Dates & Years:
        "title_year": 2023,
        "enactment_year": 2023,
        "presidential_assent_date": date(2023, 6, 12),  # Assented by President Bola Ahmed Tinubu on Democracy Day under s. 2(1) Interpretation Act
        "gazette_publication_date": None,               # Pending official print inspection
        "commencement_date": date(2023, 6, 12),         # Default commencement under s. 2(2)(a) Interpretation Act
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "Pending Print Audit",
            "volume": "Pending Print Audit",
            "publication_date": "Pending Print Audit",
            "government_notice_number": "Pending Print Audit",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette (Pending Official Print Audit)",
            "supplement_part": "Part A",
            "page_range": "Pending Print Audit",
            "title_page": "Pending Print Audit",
            "assent_commencement_page": "Pending Print Audit",
            "hosting_institutions": [
                {
                    "name": "State House of the Federal Republic of Nigeria",
                    "relationship": "Official Executive Office of the President (Announced presidential assent on 12 June 2023)",
                    "url": "https://statehouse.gov.ng",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
                {
                    "name": "Policy and Legal Advocacy Centre (PLAC)",
                    "relationship": "Civic legislative tracking body hosting unofficial enacted enrolled bill copy",
                    "url": "https://placng.org/i/wp-content/uploads/2023/07/Evidence-Amendment-Act-2023.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Obtain authentic Federal Government Gazette issue from Federal Government Printer "
                "(Lagos/Abuja). Step 2: Confirm final Act number and verified gazette page boundaries by "
                "visual and textual comparison against the physical print before legal admission."
            ),
        },
        # Integrity & Verification Metadata (Dynamic upon receipt):
        "sha256_checksum": None,
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify "
            "file integrity and detect any subsequent modification; (2) Page-by-page textual "
            "and visual comparison of the extracted content against the physical Gazette or a "
            "verified scan to confirm textual fidelity. Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.PENDING_PRINT_AUDIT,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted",
        "is_current_version": True,
        "amendment_relationships": [
            {
                "relationship_type": "AMENDS_BASE_STATUTE",
                "base_statute_family": "EVIDENCE-ACT-2011",
                "base_statute_title": "Evidence Act, 2011",
                "affected_sections": [
                    "Section 84", "Section 84A", "Section 84B", "Section 84C",
                    "Section 84D", "Section 84E", "Section 93", "Section 108",
                ],
                "effective_date": date(2023, 6, 12),
                "summary": "Introduced Sections 84A-84E for electronic signatures, amended Section 84 for digital records, and allowed remote oath administration.",
            }
        ],
    },

    "NDPA-2023": {
        "statute_family_code": "NDPA-2023",
        "official_title": "Nigeria Data Protection Act, 2023",
        "short_title": "NDPA 2023",
        "act_number": "Act No. 37 of 2023",  # Enacted as Act No. 37 of 2023
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "OFFICIAL_GAZETTE",
        # 5 Separate Legal Dates & Years:
        "title_year": 2023,
        "enactment_year": 2023,
        "presidential_assent_date": date(2023, 6, 12),  # Assented by President Bola Ahmed Tinubu on 12 June 2023
        "gazette_publication_date": date(2023, 7, 1),
        "commencement_date": date(2023, 6, 12),         # Default commencement under s. 2(2)(a) Interpretation Act
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "No. 119",
            "volume": "Vol. 110",
            "publication_date": "1st July 2023",
            "government_notice_number": "Govt. Notice No. 82",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette No. 119, Vol. 110 of 1st July, 2023",
            "supplement_part": "Part A",
            "page_range": "pages A719 - A758",
            "title_page": "Page A719 (Long Title, Part I: Objectives and Application)",
            "assent_commencement_page": "Page A758 (Assent Signature of President Bola Tinubu and Schedule)",
            "hosting_institutions": [
                {
                    "name": "Nigeria Data Protection Commission (NDPC)",
                    "relationship": "Independent statutory regulatory commission established under Section 4 of NDPA 2023",
                    "url": "https://ndpc.gov.ng/Files/Nigeria_Data_Protection_Act_2023.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
                {
                    "name": "Nigeria Computer Emergency Response Team (ngCERT / ONSA)",
                    "relationship": "National cybersecurity operations centre under Office of the National Security Adviser",
                    "url": "https://cert.gov.ng/ngcert/resources/Nigeria_Data_Protection_Act_2023.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
                {
                    "name": "Nigeria Governors' Forum (NGF) Repository",
                    "relationship": "Non-partisan platform for state chief executives; legal repository",
                    "url": "https://ngfrepository.org.ng/bitstream/handle/123456789/3790/NIGERIA-DATA-PROTECTION-ACT-2023.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Compute SHA-256 hash of the NDPC official PDF upon download to establish file integrity baseline. "
                "Step 2: Page-by-page textual and visual comparison of extracted content against physical Gazette No. 119 "
                "to verify zero omission of statutory penalty schedules and data protection obligations."
            ),
        },
        # Integrity & Verification Metadata (Dynamic upon receipt):
        "sha256_checksum": None,
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify "
            "file integrity and detect any subsequent modification; (2) Page-by-page textual "
            "and visual comparison of the extracted content against the physical Gazette or a "
            "verified scan to confirm textual fidelity. Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.OFFICIAL_AGENCY_HOSTED_REPRINT,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted",
        "is_current_version": True,
        "amendment_relationships": [],
    },

    "FCCPA-2018": {
        "statute_family_code": "FCCPA-2018",
        "official_title": "Federal Competition and Consumer Protection Act, 2018",
        "short_title": "FCCPA 2018",
        "act_number": "Act No. 1 of 2018",  # Passed by National Assembly as Act No. 1 of 2018; assented 30 Jan 2019
        "jurisdiction": "FEDERAL",
        "category": "PRIMARY_ACT",
        "issuing_authority": "National Assembly of the Federal Republic of Nigeria",
        "publisher": "Federal Government Printer, Lagos / Abuja",
        "source_type": "REGULATOR",  # Digital source document is hosted reprint from statutory regulator (FCCPC); enactment vehicle is Official Gazette
        # 5 Separate Legal Dates & Years:
        "title_year": 2018,
        "enactment_year": 2018,
        "presidential_assent_date": date(2019, 1, 30),  # Assented by President Muhammadu Buhari on 30 January 2019
        "gazette_publication_date": date(2019, 2, 1),
        "commencement_date": date(2019, 1, 30),         # Express commencement: the Act itself prints "[30th Day of January, 2019]"
        "commencement_provision_notes": (
            "The FCCPA explicitly prints '[30th Day of January, 2019]' as its express statutory commencement date. "
            "This is an express enactment provision printed on the face of the Act and is not derived from "
            "the default rule under section 2(2)(a) of the Interpretation Act (Cap. I23, LFN 2004)."
        ),
        # Gazette Provenance & Traceability:
        "official_gazette_details": {
            "gazette_number": "No. 18",
            "volume": "Vol. 106",
            "publication_date": "1st February 2019",
            "government_notice_number": "Govt. Notice No. 5",
            "full_gazette_ref": "Federal Republic of Nigeria Official Gazette No. 18, Vol. 106 of 1st February, 2019",
            "supplement_part": "Part A",
            "cover_stated_page_range": "pages A1 - A95",
            "actual_printed_page_range": "pages A1 - A96",
            "page_range": "pages A1 - A96 (Cover/Index: A1-A95; Scanned Body: A1-A96)",
            "pagination_discrepancy_notes": (
                "Documented Gazette index/body pagination discrepancy: The Official Gazette cover index and table "
                "of contents record the Act as spanning pages A1 - A95, whereas the actual printed page labels in "
                "the scanned supplement body run from Page A1 through Page A96. The presidential assent signature "
                "of President Muhammadu Buhari and the National Assembly authentication schedule appear on Page A96."
            ),
            "title_page": "Page A1 (Long Title, Part I: Establishment of the Commission, Sections 1-2)",
            "assent_commencement_page": "Page A96 (Presidential Assent Signature of President Muhammadu Buhari, Schedules, and Legislative Authentication)",
            "hosting_institutions": [
                {
                    "name": "Federal Competition and Consumer Protection Commission (FCCPC)",
                    "relationship": "Statutory regulatory commission established under Section 3 of FCCPA 2018",
                    "url": "https://fccpc.gov.ng/wp-content/uploads/2020/12/Federal-Competition-and-Consumer-Protection-Act-2018.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
                {
                    "name": "World Intellectual Property Organization (WIPO)",
                    "relationship": "UN specialized agency member state legal repository",
                    "url": "https://www.wipo.int/edocs/lexdocs/laws/en/ng/ng064en.pdf",
                    "provenance": "THIRD_PARTY_REFERENCE_COPY",
                },
                {
                    "name": "Securities and Exchange Commission (SEC Nigeria)",
                    "relationship": "Apex regulatory authority for Nigerian capital markets (mergers and acquisitions partner)",
                    "url": "https://sec.gov.ng/wp-content/uploads/2019/04/Federal-Competition-and-Consumer-Protection-Act-2018.pdf",
                    "provenance": "OFFICIAL_AGENCY_HOSTED_REPRINT",
                },
            ],
            "remaining_verification_requirement": (
                "Step 1: Compute SHA-256 hash of the downloaded FCCPC PDF to establish file integrity baseline. "
                "Step 2: Page-by-page textual and visual comparison across all 18 Parts (Parts I-XVIII), all 168 sections, "
                "the First Schedule (Professional Associations under s. 167(14)), and the Second Schedule (Transitional "
                "and Supplementary Provisions under s. 166) across pages A1-A96 against physical Gazette No. 18 to ensure "
                "complete extraction without omission before legal counsel verification."
            ),
            "extraction_completeness_spec": {
                "expected_parts": "Parts I - XVIII (18 Parts total; Part XVIII: Miscellaneous Provisions, ss. 156-168)",
                "expected_parts_list": [
                    "Part I", "Part II", "Part III", "Part IV", "Part V", "Part VI",
                    "Part VII", "Part VIII", "Part IX", "Part X", "Part XI", "Part XII",
                    "Part XIII", "Part XIV", "Part XV", "Part XVI", "Part XVII", "Part XVIII",
                ],
                "expected_sections_count": 168,
                "first_section": "Section 1",
                "last_section": "Section 168",
                "expected_schedules": [
                    "First Schedule",
                    "Second Schedule",
                ],
                "assent_page": "Page A96",
            },
        },
        # Integrity & Verification Metadata (Dynamic upon receipt):
        "sha256_checksum": None,
        "integrity_verification_method": (
            "Two-step verification: (1) SHA-256 hash of the exact downloaded file to verify "
            "file integrity and detect any subsequent modification; (2) Page-by-page textual "
            "and visual comparison of the extracted content against the physical Gazette or a "
            "verified scan to confirm textual fidelity. Results recorded independently."
        ),
        "metadata_status": MetadataVerificationLevel.OFFICIAL_AGENCY_HOSTED_REPRINT,
        "verification_status": "PENDING_VERIFICATION",
        "is_verified_authentic": False,
        "version_label": "As Enacted",
        "is_current_version": True,
        "amendment_relationships": [],
    },
}


def get_statute_registry_entry(family_code: str) -> Optional[Dict[str, Any]]:
    """Retrieves an authoritative legal registry entry by its statute family code."""
    return NIGERIAN_LEGAL_SOURCE_REGISTRY.get(family_code.upper())


def list_all_registry_entries() -> List[Dict[str, Any]]:
    """Returns a list of all 6 priority legal instruments in the registry."""
    return list(NIGERIAN_LEGAL_SOURCE_REGISTRY.values())


def get_expected_amendments(family_code: str) -> List[Dict[str, Any]]:
    """Retrieves expected amendment relationships for a given statute family."""
    entry = get_statute_registry_entry(family_code)
    if not entry:
        return []
    return entry.get("amendment_relationships", [])
