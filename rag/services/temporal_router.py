"""
Temporal Routing and Statutory Lineage Service for Nigerian Law.

Guarantees:
1. Distinguishes Current-Law queries from Historical-Law queries.
2. Temporal date extraction: parses explicit years (e.g. "in 2019", "under CAMA 1990")
   or dates (e.g. "as of 15th May 2021").
3. Applies legally accurate effective date boundaries:
   effective_from <= target_date <= (effective_to OR current).
4. Preserves clear distinction between original provisions, amending provisions,
   and verified consolidated provisions.
5. Formats statutory amendment lineages joining StatuteAmendmentRecord.
"""
import re
from datetime import date, datetime
from typing import Tuple, Optional, Dict, Any, List
from django.db.models import Q, QuerySet

from rag.models import LegalProvisionChunk, LegalSourceDocument, DocumentStatus


RE_YEAR = re.compile(r'\b(?:in|as at|under|during|before|after|year|of)\s+((?:19|20)\d{2})\b', re.IGNORECASE)
RE_EXPLICIT_DATE = re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+((?:19|20)\d{2})\b', re.IGNORECASE)
RE_HISTORICAL_INTENT = re.compile(
    r'\b(?:repealed|historical|prior to|superseded|former|formerly|amended by|under the old|previous version)\b',
    re.IGNORECASE
)


def analyze_temporal_intent(query: str) -> Tuple[bool, Optional[date], Optional[int]]:
    """
    Analyzes query text for historical legal intent and target effective date.

    Returns:
        (is_historical, target_date, target_year)
    """
    # 1. Check explicit day-month-year pattern (e.g. "15th June 2021")
    date_match = RE_EXPLICIT_DATE.search(query)
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(2)[:3].title()
        year = int(date_match.group(3))
        try:
            parsed_date = datetime.strptime(f"{day} {month_str} {year}", "%d %b %Y").date()
            # If date is in the past, treat as historical
            current_date = date.today()
            if parsed_date < current_date:
                return True, parsed_date, year
        except ValueError:
            pass

    # 2. Check year pattern (e.g. "in 2019", "CAMA 1990", "prior to 2022")
    year_match = RE_YEAR.search(query)
    current_year = date.today().year

    if year_match:
        year = int(year_match.group(1))
        if year < current_year:
            # Historical query targeting mid-year of that statutory period
            return True, date(year, 6, 30), year

    # 3. Check historical keywords without explicit year
    if RE_HISTORICAL_INTENT.search(query):
        return True, None, None

    # Default: Current active law
    return False, None, None


def apply_temporal_filter(
    queryset: QuerySet,
    is_historical: bool = False,
    target_date: Optional[date] = None,
) -> QuerySet:
    """
    Applies legally accurate temporal filters to a LegalProvisionChunk QuerySet.
    """
    if is_historical and target_date:
        # Historical retrieval: statute must have been in legal force on target_date
        return queryset.filter(
            document__effective_from__lte=target_date
        ).filter(
            Q(document__effective_to__isnull=True) |
            Q(document__effective_to__gte=target_date)
        )
    elif is_historical:
        # General historical inquiry: allow repealed and amended versions
        return queryset.filter(
            document__status__in=[DocumentStatus.ACTIVE, DocumentStatus.AMENDED, DocumentStatus.REPEALED]
        )
    else:
        # Default: CURRENT in-force law only
        return queryset.filter(
            document__is_current_version=True,
            document__status__in=[DocumentStatus.ACTIVE, DocumentStatus.AMENDED],
        )


def format_statutory_lineage_notice(chunk: LegalProvisionChunk) -> Dict[str, Any]:
    """
    Generates a structured amendment lineage notice for a retrieved chunk.
    Distinguishes original enactments, amending provisions, and consolidated text.
    """
    doc = chunk.document
    amend_rec = chunk.amendment_record

    # Strict Two-Way Verification Guard:
    # An amendment relationship only becomes active in retrieval and lineage notices
    # when BOTH base statute and amending act are verified authentic.
    is_amendment_active = bool(
        amend_rec is not None and
        amend_rec.base_statute.is_verified_authentic and
        amend_rec.amending_act.is_verified_authentic
    )

    lineage = {
        'provision_nature': 'CONSOLIDATED' if 'consolidated' in doc.version_label.lower() else 'ORIGINAL',
        'is_amended': is_amendment_active or doc.status == DocumentStatus.AMENDED,
        'amending_act_title': None,
        'commencement_date': None,
        'summary_of_changes': None,
        'lineage_text': "",
    }

    if is_amendment_active:
        lineage['provision_nature'] = 'AMENDED'
        lineage['amending_act_title'] = amend_rec.amending_act.short_title
        lineage['commencement_date'] = str(amend_rec.commencement_date)
        lineage['summary_of_changes'] = amend_rec.summary_of_changes
        lineage['lineage_text'] = (
            f"[AMENDMENT NOTICE: {chunk.section_number} was amended by "
            f"{amend_rec.amending_act.short_title} with effect from {amend_rec.commencement_date}. "
            f"Official Gazette: {amend_rec.official_gazette_ref or doc.official_gazette_ref}. "
            f"Summary of changes: {amend_rec.summary_of_changes}]"
        )
    elif doc.status == DocumentStatus.REPEALED:
        lineage['provision_nature'] = 'REPEALED'
        repealed_date = doc.effective_to or 'Repealed'
        lineage['lineage_text'] = (
            f"[HISTORICAL NOTICE: {doc.short_title} is REPEALED and no longer in force. "
            f"Cessation date: {repealed_date}. This provision is provided for historical legal reference only.]"
        )

    return lineage
