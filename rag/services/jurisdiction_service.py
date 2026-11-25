"""
Jurisdiction Scoping and Resolution Service for Nigerian Law.

Nigerian Legal Structure:
1. Federal Jurisdiction: Applies nationwide (e.g. Constitution, CAMA, Evidence Act, NDPA).
2. State Jurisdiction: Residual legislative matters (e.g. Tenancy, Landlord/Tenant recovery of premises, local taxes).
   Priority: When a user query explicitly concerns a state (e.g. "Lagos tenancy", "Rivers State recovery of premises"),
   that State's statutes take precedence, supplemented by relevant Federal framework legislation.
"""
import re
from typing import Tuple, List, Optional
from rag.models import Jurisdiction


STATE_PATTERNS = {
    Jurisdiction.LAGOS_STATE: [
        r'\blagos\b',
        r'\blagosian\b',
        r'\bikeja\b',
        r'\blekki\b',
        r'\bvictoria island\b',
    ],
    Jurisdiction.RIVERS_STATE: [
        r'\brivers\b',
        r'\brivers state\b',
        r'\bport harcourt\b',
        r'\bpitakwa\b',
    ],
    Jurisdiction.FCT_ABUJA: [
        r'\babuja\b',
        r'\bfct\b',
        r'\bfederal capital territory\b',
    ],
}


def detect_jurisdiction(query: str) -> Tuple[Optional[str], List[str]]:
    """
    Detects explicitly mentioned Nigerian jurisdictions from query text.

    Returns:
        (primary_jurisdiction, prioritized_jurisdiction_order)
        e.g., ("LAGOS_STATE", ["LAGOS_STATE", "FEDERAL"])
    """
    query_lower = query.lower()

    for jur, patterns in STATE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, query_lower):
                # State explicitly mentioned: State takes primary priority, followed by Federal
                return jur, [jur, Jurisdiction.FEDERAL]

    # Explicit Federal keywords
    if re.search(r'\bfederal\b|\bnigeria\b|\bnigerian\b|\bnational assembly\b', query_lower):
        return Jurisdiction.FEDERAL, [Jurisdiction.FEDERAL]

    # Default fallback: Federal law applies across Nigeria
    return Jurisdiction.FEDERAL, [Jurisdiction.FEDERAL]
