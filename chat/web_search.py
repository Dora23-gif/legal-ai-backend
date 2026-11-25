"""
Live Web Search & Intelligent Legal Research Fallback for Nigerian Legal AI Assistant.

Provides:
1. Resilient multi-endpoint web search (DuckDuckGo Lite/HTML) without external API keys or cost.
2. 6-Tier Nigerian Legal Source Hierarchy classification and ranking:
   - Tier 1: Official Government Publications & Gazette sources (nass.gov.ng, gazettes.africa, fedgov.ng)
   - Tier 2: Federal and State Ministries of Justice, State Houses of Assembly (lagosministryofjustice.org, lagoshouseofassembly.gov.ng)
   - Tier 3: Nigerian Courts and Judicial Publications (supremecourt.gov.ng, courtofappeal.gov.ng, nicn.gov.ng)
   - Tier 4: Official Regulatory Agencies (CAC, NDPC, FCCPC, CBN, NIPC, FIRS, SEC)
   - Tier 5: Reputable Legal Research Platforms & Repositories (PLAC, LawNigeria, NigerianLawGuru, Legal500)
   - Tier 6: Reputable Secondary Legal Commentary & Public Sources
3. Grounded legal research prompt formatting with clickable genuine links.
4. Strict safeguards: no fabricated links, no automatic document authentication.
"""

import html
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse
import requests

logger = logging.getLogger(__name__)

HISTORICAL_EXCLUSION_PATTERN = re.compile(
    r'\b(who was|who were|who had been|former|first|previous)\b.*?\b(19\d\d|200\d|201\d|202[0-3]|history|originally)\b|'
    r'\b(in (?:19\d\d|200\d|201\d|202[0-3]))\b.*?\b(who was|who were|who served)\b',
    re.IGNORECASE
)

SEARCH_TRIGGER_PATTERNS = [
    # Explicit present-tense inquiries about current leaders, officeholders, and executives
    r"\bwho(?:\x27s|'s|\s+is|\s+are)\b.*?\b(president|prime minister|governor|deputy governor|vice president|vp|ceo|chief executive|managing director|md|chairman|minister|attorney general|chief justice|cjn|senate president|speaker|head|leader|director general|dg)\b",
    # Temporal indicators for current or real-time information
    r'\b(latest|newest|recent|recently|current|currently|today|now|right now|at present|this year|this month|this week)\b',
    # Technology versions and software releases
    r"\b(?:what(?:\s+is|\x27s|'s)?\s+(?:the\s+)?(?:latest\s+|current\s+|newest\s+)?version\s+of)\b",
    r'\b(?:latest\s+version|new\s+release|release\s+date)\b',
    # Prices, financial metrics, exchange rates
    r'\b(price|cost|rate|exchange rate|fuel price|petrol price|inflation|cbn rate)\b',
    # News, breaking events, election outcomes
    r'\b(news|headline|headlines|breaking news|election|updates?|trends?|trending)\b',
    r'\b(?:what\s+happened\s+(?:today|recently|this week|now))\b',
    # Explicit search commands
    r'\b(search the web|search online|browse the web|google this|check online|look up online)\b',
    # Recent years (current / upcoming)
    r'\b(2025|2026)\b',
    # Fast-changing consumer tech / currencies
    r'\b(iphone|samsung|pixel|naira|dollar)\b',
    # Current legal developments
    r'\b(latest developments|recent changes|new amendments?|current status of|is .* still in force|has .* been amended)\b',
]

COMPILED_TRIGGERS = [re.compile(p, re.IGNORECASE) for p in SEARCH_TRIGGER_PATTERNS]

# ==============================================================================
# Nigerian Legal Source Hierarchy (Strict Hostname Allowlists & Anti-Poisoning)
# ==============================================================================

# Tier 1: Primary National Gazette & Legislature
# Strictly authenticated organs of the Federal Government and National Assembly.
# (Aggregators such as gazettes.africa and nigerialawguru.com are strictly NOT Tier 1).
TRUSTED_TIER_1_HOSTNAMES = {
    'nass.gov.ng',
    'officialgazette.gov.ng',
    'fedgov.ng',
}

# Tier 2: Federal and State Ministries of Justice, State Houses of Assembly
TRUSTED_TIER_2_HOSTNAMES = {
    'justice.gov.ng',
    'lagosministryofjustice.org',
    'lagoshouseofassembly.gov.ng',
}

# Tier 3: Superior Courts of Record (Supreme Court, Court of Appeal, FHC, NICN, NJC)
TRUSTED_TIER_3_HOSTNAMES = {
    'supremecourt.gov.ng',
    'courtofappeal.gov.ng',
    'nicn.gov.ng',
    'fhc.gov.ng',
    'fhc-ng.com',
    'njc.gov.ng',
    'judiciary.gov.ng',
    'laws.court.gov.ng',
}

# Tier 4: Official Statutory Regulatory Agencies
TRUSTED_TIER_4_HOSTNAMES = {
    'cac.gov.ng',
    'ndpc.gov.ng',
    'fccpc.gov.ng',
    'cbn.gov.ng',
    'nipc.gov.ng',
    'firs.gov.ng',
    'sec.gov.ng',
    'naicom.gov.ng',
    'ncc.gov.ng',
    'pencom.gov.ng',
    'nitda.gov.ng',
    'nafdac.gov.ng',
}

# Tier 5: Reputable Legal Research Platforms & Established Repositories
# Independent repositories, NGO policy institutes, and legal databases.
TRUSTED_TIER_5_HOSTNAMES = {
    'placng.org',
    'placbillstrack.org',
    'lawnigeria.com',
    'lawpavilion.com',
    'legalblitz.com',
    'gazettes.africa',        # Third-party digital gazette repository (Tier 5, not Tier 1)
    'nigerialawguru.com',     # Private digital law repository (Tier 5, not Tier 1)
    'legal500.com',
    'chambers.com',
    'mondaq.com',
    'ibanet.org',
}

# Tier 6: Recognized Practitioner Analysis, Law Firm Publications & Bar Commentary
RECOGNIZED_TIER_6_HOSTNAMES = {
    'dnllegalandstyle.com',
    'barristerng.com',
    'lawpadi.com',
    'thenigerialawyer.com',
    'loyalnigerianlawyer.com',
    'lawglobalhub.com',
}


def extract_clean_hostname(url: str) -> str:
    """
    Extracts and normalizes the lowercase hostname from a URL.
    Safely handles port numbers, trailing dots, and user credentials.
    Returns empty string if invalid or unparseable.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc or parsed.path.split('/')[0]
        if '@' in netloc:
            netloc = netloc.split('@')[-1]
        if ':' in netloc:
            netloc = netloc.split(':')[0]
        return netloc.strip().strip('.').lower()
    except Exception:
        return ""


def _matches_domain_allowlist(hostname: str, allowlist: set) -> bool:
    """
    Exact hostname validation against a trusted publisher allowlist.
    Returns True if hostname exactly equals an allowed domain or is a valid subdomain of it.
    Prevents substring poisoning (e.g. 'fake-site.com/nass.gov.ng' or 'evilsite.gov.ng.attacker.com').
    """
    if not hostname:
        return False
    for allowed in allowlist:
        if hostname == allowed or hostname.endswith('.' + allowed):
            return True
    return False


def classify_legal_source_tier(url: str) -> Tuple[int, str]:
    """
    Evaluates a URL against the Nigerian Legal Source Hierarchy using exact hostname validation.
    
    Returns (tier_number, tier_label).
    Tier ranking (1=Highest authority to 6=Unverified/General):
      - Tier 1: Official Government Publication & Gazette (NASS, Federal Gazette)
      - Tier 2: Federal/State Ministry or Legislature (Ministry of Justice, State Assembly)
      - Tier 3: Nigerian Judicial & Court Record (Superior Courts of Record)
      - Tier 4: Official Regulatory Agency (CAC, CBN, NDPC, etc.)
      - Tier 5: Reputable Legal Research Platform (PLAC, LawPavilion, LawNigeria, Gazettes.Africa)
      - Tier 6: Legal Commentary / Practitioner Analysis (for recognized legal blogs) OR
                Unverified Web Source (Caution: Unauthenticated) (for unknown websites).
    """
    hostname = extract_clean_hostname(url)
    if not hostname:
        return 6, "Unverified Web Source (Caution: Unauthenticated)"

    # 1. Tier 1: Primary National Gazette & Legislature
    if _matches_domain_allowlist(hostname, TRUSTED_TIER_1_HOSTNAMES):
        return 1, "Official Government Publication & Gazette"

    # 2. Tier 3: Superior Courts of Record
    if _matches_domain_allowlist(hostname, TRUSTED_TIER_3_HOSTNAMES):
        return 3, "Nigerian Judicial & Court Record"

    # 3. Tier 4: Official Regulatory Agencies
    if _matches_domain_allowlist(hostname, TRUSTED_TIER_4_HOSTNAMES):
        return 4, "Official Regulatory Agency"

    # 4. Tier 2: Federal and State Ministries / State Houses of Assembly
    if _matches_domain_allowlist(hostname, TRUSTED_TIER_2_HOSTNAMES):
        return 2, "Federal/State Ministry or Legislature"

    # Check for authenticated Nigerian state government portals ending strictly in .gov.ng
    # e.g. lagosstate.gov.ng, riversstate.gov.ng, kano.gov.ng
    if hostname == 'gov.ng' or hostname.endswith('.gov.ng'):
        return 2, "Federal/State Ministry or Legislature"

    # 5. Tier 5: Reputable Legal Research Platforms & Established Repositories
    if _matches_domain_allowlist(hostname, TRUSTED_TIER_5_HOSTNAMES):
        return 5, "Reputable Legal Research Platform"

    # 6. Tier 6: Recognized Commentary
    if _matches_domain_allowlist(hostname, RECOGNIZED_TIER_6_HOSTNAMES):
        return 6, "Legal Commentary / Practitioner Analysis"

    # Unknown website: NEVER classify an unknown website as an authoritative source!
    return 6, "Unverified Web Source (Caution: Unauthenticated)"


def should_search_web(
    query: str,
    has_document: bool = False,
    web_enabled: Optional[bool] = None
) -> bool:
    """
    Determines whether a live web search should be executed for the user query.
    """
    if has_document and web_enabled is not True:
        return False

    if web_enabled is True:
        return True
    if web_enabled is False:
        return False

    clean_query = query.strip()
    if not clean_query:
        return False

    # Historical exclusion: if explicitly asking about a past officeholder in a past historical year
    if HISTORICAL_EXCLUSION_PATTERN.search(clean_query):
        return False

    for pattern in COMPILED_TRIGGERS:
        if pattern.search(clean_query):
            return True

    return False


# ==============================================================================
# Search Execution Functions (Resilient ₦0 Cost Engine)
# ==============================================================================

def _execute_http_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Executes resilient DuckDuckGo search via Lite and HTML endpoints.
    Extracts authentic titles, target URLs, and snippets.
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # 1. Primary: DuckDuckGo Lite endpoint
    try:
        response = requests.post(
            'https://lite.duckduckgo.com/lite/',
            data={'q': query},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200 and 'result-link' in response.text:
            links = re.findall(
                r'<a[^>]+href=[\'"]([^\'"]+)[\'"][^>]+class=[\'"]result-link[\'"][^>]*>(.*?)</a>',
                response.text
            )
            snippets = re.findall(
                r'<td[^>]+class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>',
                response.text,
                re.DOTALL
            )
            results = []
            for i, (href, raw_title) in enumerate(links[:max_results]):
                if 'uddg=' in href:
                    actual_url = unquote(href.split('uddg=')[1].split('&')[0])
                elif href.startswith('//'):
                    actual_url = 'https:' + href
                else:
                    actual_url = href

                clean_title = html.unescape(re.sub(r'<[^>]+>', '', raw_title).strip())
                snippet_text = ""
                if i < len(snippets):
                    snippet_text = html.unescape(re.sub(r'<[^>]+>', '', snippets[i]).strip())

                if clean_title and (actual_url.startswith('http://') or actual_url.startswith('https://')):
                    results.append({
                        'title': clean_title,
                        'url': actual_url,
                        'snippet': snippet_text[:500] if len(snippet_text) > 500 else snippet_text,
                    })
            if results:
                return results
    except Exception as exc:
        logger.warning("Lite search failed: %s", exc)

    # 2. Secondary: DuckDuckGo HTML endpoint
    try:
        response = requests.post(
            'https://html.duckduckgo.com/html/',
            data={'q': query},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            items = re.findall(
                r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                response.text,
                re.DOTALL
            )
            results = []
            for href, title_html, snippet_html in items[:max_results]:
                if 'uddg=' in href:
                    actual_url = unquote(href.split('uddg=')[1].split('&')[0])
                elif href.startswith('//'):
                    actual_url = 'https:' + href
                else:
                    actual_url = href

                clean_title = html.unescape(re.sub(r'<[^>]+>', '', title_html).strip())
                clean_snippet = html.unescape(re.sub(r'<[^>]+>', '', snippet_html).strip())

                if clean_title and (actual_url.startswith('http://') or actual_url.startswith('https://')):
                    results.append({
                        'title': clean_title,
                        'url': actual_url,
                        'snippet': clean_snippet[:500] if len(clean_snippet) > 500 else clean_snippet,
                    })
            if results:
                return results
    except Exception as exc:
        logger.warning("HTML search failed: %s", exc)

    return []


def perform_web_search(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Executes a general real-time web search with fallback.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    logger.info("Executing web search for: %s", clean_query[:80])

    # 1. Primary: Resilient HTTP Search
    results = _execute_http_search(clean_query, max_results=max_results)
    if results:
        return results

    # 2. Fallback: ddgs library
    try:
        from ddgs import DDGS
        with DDGS(timeout=10) as ddgs:
            raw_results = list(ddgs.text(clean_query, max_results=max_results))
        for item in raw_results:
            title = item.get('title', '').strip()
            body = item.get('body', '').strip()
            href = item.get('href', '').strip()
            if title and (body or href):
                results.append({
                    'title': title,
                    'snippet': body[:500],
                    'url': href,
                })
        return results
    except Exception as exc:
        logger.warning("ddgs fallback failed: %s", exc)
        return []


def extract_publication_date(title: str, snippet: str) -> str:
    """
    Attempts to extract publication date or statutory enactment year from title or snippet.
    Returns formatted date string or 'Undated / Check Supporting Document'.
    """
    text = f"{title} {snippet}"

    # 1. Look for full date like '15 February 2024' or 'Feb 15, 2024'
    full_date_match = re.search(
        r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2})\b',
        text,
        re.IGNORECASE
    )
    if full_date_match:
        return full_date_match.group(1)

    us_date_match = re.search(
        r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2})\b',
        text,
        re.IGNORECASE
    )
    if us_date_match:
        return us_date_match.group(1)

    # 2. Look for explicit Act / Law / Gazette year like 'Act 2024' or 'Law, 2020' or 'Gazette No. 12 ... 2021'
    act_year_match = re.search(
        r'\b(?:Act|Law|Gazette|Decree|Order|Notice|Regulations?)\s*(?:of\s*)?,?\s*((?:19|20)\d{2})\b',
        text,
        re.IGNORECASE
    )
    if act_year_match:
        return f"Enactment Year: {act_year_match.group(1)}"

    # 3. Look for standalone 4-digit recent year
    year_match = re.search(r'\b(20[12]\d)\b', text)
    if year_match:
        return f"Year: {year_match.group(1)}"

    return "Undated / Check Supporting Document"


def perform_legal_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Executes intelligent legal web research focused strictly on Nigerian law.
    Applies the Nigerian Legal Source Hierarchy and sorts results by source authority.

    Returns:
        List of dicts: [
            {
                'title': str,
                'url': str,
                'snippet': str,
                'tier_number': int (1-6),
                'tier_label': str,
                'domain': str,
                'publication_date': str,
            }
        ]
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    search_terms = clean_query
    lower_query = clean_query.lower()
    if 'amendment' not in lower_query and any(w in lower_query for w in ['act', 'law', 'statute', 'cybercrime', 'cama', 'tax', 'electoral', 'police', 'penal', 'criminal']):
        search_terms = f"{clean_query} Nigeria amendment"
    elif 'nigeria' not in lower_query and 'lagos' not in lower_query and 'cama' not in lower_query:
        search_terms = f"{clean_query} Nigeria"

    raw_results = _execute_http_search(search_terms, max_results=max_results * 2)
    if not raw_results:
        raw_results = perform_web_search(search_terms, max_results=max_results * 2)

    if not raw_results:
        return []

    enriched_results: List[Dict[str, Any]] = []
    seen_urls = set()

    for item in raw_results:
        url = item['url']
        if url in seen_urls:
            continue
        seen_urls.add(url)

        tier_num, tier_lbl = classify_legal_source_tier(url)
        domain = extract_clean_hostname(url) or urlparse(url).netloc
        pub_date = extract_publication_date(item['title'], item['snippet'])

        enriched_results.append({
            'title': item['title'],
            'url': url,
            'snippet': item['snippet'],
            'tier_number': tier_num,
            'tier_label': tier_lbl,
            'domain': domain,
            'publication_date': pub_date,
        })

    # Sort primarily by source hierarchy (Tier 1 first, down to Tier 6), then by snippet length
    enriched_results.sort(key=lambda x: (x['tier_number'], -len(x['snippet'])))

    return enriched_results[:max_results]


# ==============================================================================
# Prompt Formatting
# ==============================================================================

def format_web_search_prompt(user_question: str, results: List[Dict[str, str]]) -> str:
    """
    Injects general live search results into the prompt with strict grounding instructions.
    If search results could not be retrieved, injects an honest transparent notice.
    """
    if not results:
        return (
            "====================================================================\n"
            "REAL-TIME SEARCH NOTICE\n"
            "====================================================================\n"
            f"A live web search was initiated for: \"{user_question}\", but real-time search results could not be retrieved at this moment.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Do NOT guess or state outdated pre-cutoff information as current settled fact for time-sensitive questions.\n"
            "- Transparently inform the user that live web search results could not be retrieved right now to verify the current information, and state what was known up to your knowledge cutoff.\n\n"
            f"User Query: {user_question}"
        )

    formatted_sources = []
    for idx, r in enumerate(results, start=1):
        formatted_sources.append(
            f"[{idx}] Title: {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    Excerpt: {r['snippet']}"
        )

    sources_block = "\n\n".join(formatted_sources)

    return (
        "====================================================================\n"
        "REAL-TIME WEB SEARCH RESULTS (GROUND TRUTH FOR CURRENT INFORMATION)\n"
        "====================================================================\n"
        f"The following live web search results were retrieved for the query: \"{user_question}\"\n\n"
        f"{sources_block}\n\n"
        "CRITICAL GROUNDING DIRECTIVES:\n"
        "1. GROUND TRUTH OVERRIDE: You MUST base your answer on the live web search results above. Do NOT rely on or substitute outdated pretraining knowledge (such as past officeholders, older software versions, or outdated statistics).\n"
        "2. ACCURACY FOR CURRENT EVENTS: For questions about current officeholders, leaders, prices, or versions, treat the search results as the authoritative reality.\n"
        "3. GENUINE CLICKABLE CITATIONS: Naturally cite or reference relevant sources with clickable markdown links (e.g. [Source Title](URL)) using the exact URLs from the search results above.\n"
        "4. ANTI-FABRICATION: Never fabricate URLs or names not present in the results above.\n\n"
        f"User Query: {user_question}"
    )


def format_legal_research_prompt(user_question: str, results: List[Dict[str, Any]]) -> str:
    """
    Formats external Nigerian legal research results according to the preferred source hierarchy.
    Instructs the assistant to synthesize findings, cite genuine clickable links,
    and maintain a helpful, professional, authoritative, and natural tone without
    leaking internal database status or quarantine metadata.
    """
    if not results:
        return (
            "====================================================================\n"
            "NIGERIAN LEGAL RESEARCH CONTEXT\n"
            "====================================================================\n"
            "INSTRUCTIONS FOR YOUR ANSWER:\n"
            "1. Helpfully and authoritatively explain the governing principles of Nigerian law applicable to this subject.\n"
            "2. Clarify jurisdiction (e.g. whether the issue is governed by Federal legislation or State law).\n"
            "3. Only disclose uncertainty when it materially affects the legal answer (e.g. if the current wording of an amendment cannot be established from authoritative records). "
            "Never expose internal system, database, or ingestion status to the user.\n"
            "4. NEVER use repetitive, robotic rejections like 'This law is not in my database', 'I cannot provide legal advice', or 'This statute is unverified'. "
            "Communicate naturally, professionally, and constructively.\n"
            "====================================================================\n\n"
            f"User Query: {user_question}"
        )

    formatted_sources = []
    for idx, r in enumerate(results, start=1):
        formatted_sources.append(
            f"[{idx}] Source Title: {r['title']}\n"
            f"    Hierarchy Tier: Tier {r['tier_number']} ({r['tier_label']})\n"
            f"    Domain: {r['domain']}\n"
            f"    Publication Date / Period: {r.get('publication_date', 'Undated')}\n"
            f"    URL: {r['url']}\n"
            f"    Excerpt: {r['snippet']}"
        )

    sources_block = "\n\n".join(formatted_sources)

    return (
        "====================================================================\n"
        "AUTHORITATIVE NIGERIAN LEGAL RESEARCH CONTEXT\n"
        "====================================================================\n"
        "The following legal research findings were retrieved from official and reputable Nigerian legal sources:\n\n"
        f"{sources_block}\n\n"
        "MANDATORY INSTRUCTIONS FOR ANSWERING WITH LEGAL RESEARCH:\n"
        "1. ACCURATE, NATURAL LEGAL SYNTHESIS: Answer the user's question directly, accurately, and authoritatively "
        "based on the governing Nigerian legislation, regulatory guidelines, and judicial authorities.\n"
        "2. GENUINE CLICKABLE CITATIONS: You MUST cite the sources using clickable Markdown links "
        "with their exact URLs from the research above (e.g. [CAC](https://cac.gov.ng/...), [PLAC](https://...)) "
        "so the user can read the supporting documents. NEVER invent URLs or link to pages not in the retrieved sources.\n"
        "3. NO INTERNAL METADATA LEAKS: NEVER state or imply that any law is 'unverified in our database', 'pending verification', "
        "or 'missing from our internal database'. Ordinary users must receive natural, authoritative legal information without exposure "
        "to internal development status or quarantine safeguards.\n"
        "4. STATUTORY CURRENCY & AMENDMENTS: Note applicable commencement dates, enactment years, and whether the provision has been "
        "amended (e.g. Cybercrimes Act 2024 Amendment Act altering Section 24). Only disclose uncertainty when it materially affects "
        "the legal answer (e.g., if recent amendments cannot be confirmed from available records).\n"
        "5. JURISDICTION & COMMENCEMENT DETAILS: Note applicable commencement dates, enactment years, and whether the rule is "
        "governed by Federal legislation or State-specific law (e.g. tenancy in Nigeria is governed by State laws "
        "such as the Lagos State Tenancy Law 2011).\n"
        "6. PRIMARY VS SECONDARY: Give greatest weight to Tier 1-4 official government records and superior court judgments. "
        "Treat Tier 5 repositories as reliable reference copies, and Tier 6 as secondary commentary.\n"
        "7. CONSTRUCTIVE & PROFESSIONAL TONE: Avoid robotic, negative statements (e.g. 'This law is not in my database' or "
        "'This response is only for general informational purposes'). Answer directly, helpfully, and professionally.\n"
        "====================================================================\n\n"
        f"User Query: {user_question}"
    )
