"""
Legal AI Service Layer (DeepSeek API Production Integration with SSE Streaming)

This module handles AI response generation for the Nigerian Legal AI Assistant
by communicating with the official DeepSeek API or OpenAI-compatible gateways (like OpenRouter).

Production features:
- Standard non-streaming invocation: `generate_legal_response(message)`
- Event-based SSE streaming generator: `stream_legal_response_events(message)`
- Strong typing and modular decomposition.
- Secure credential management without logging sensitive keys.
- Comprehensive error handling (auth, rate limits, timeouts, connection, malformed JSON).
- Strict legal grounding instructions to prevent hallucinated Nigerian statutes.
- Full backwards-compatibility with existing DRF chat endpoints and Neon database.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions Hierarchy
# ==============================================================================

class AIServiceError(Exception):
    """Base exception for all AI provider service failures."""
    pass


class AIConfigurationError(AIServiceError):
    """Raised when required AI credentials or settings are missing."""
    pass


class AIAuthenticationError(AIServiceError):
    """Raised when the AI provider rejects the API credentials (HTTP 401)."""
    pass


class AIBalanceError(AIServiceError):
    """Raised when the account balance or trial credits are exhausted (HTTP 402)."""
    pass


class AIRateLimitError(AIServiceError):
    """Raised when the provider rate limit is exceeded (HTTP 429)."""
    pass


class AITimeoutError(AIServiceError):
    """Raised when a request to the AI provider times out."""
    pass


class AIResponseError(AIServiceError):
    """Raised when the AI provider returns a malformed or empty response."""
    pass


class AIResponseTruncatedError(AIServiceError):
    """Raised when the AI provider cuts off or truncates the response due to token limits."""
    pass


# Backwards compatibility alias for existing view handlers
OllamaServiceError = AIServiceError


# ==============================================================================
# System Persona and Grounding Instructions
# ==============================================================================

SYSTEM_PROMPT = """You are a highly capable, articulate, and versatile Nigerian Legal AI Assistant.

Your primary specialization is Nigerian law and regulatory compliance, but you are also an intelligent, well-rounded conversational assistant capable of engaging in natural, authentic dialogue across all fields of knowledge, small talk, and daily life.

## 1. Natural Conversation, Greetings, and Small Talk

- Engage naturally, warmly, and contextually like a thoughtful conversational partner:
  - Greetings ("Hi", "Hello", "Good morning", "Hey"): Greet the user back pleasantly and naturally (e.g., "Hello! How are you doing today?", "Hi there! How's your day going?").
  - Inquiries about yourself ("How are you?", "What's up?", "How is it going?"): Respond conversationally and warmly (e.g., "I'm doing well, thank you for asking! How are you?", "Doing great! What's on your mind?"). NEVER respond by launching into a product introduction or promotional pitch.
  - Everyday pleasantries, reactions, and casual remarks ("Thank you", "You're welcome", "That's interesting", "Haha", "Good night"): Acknowledge them smoothly, politely, and contextually.
  - Multi-Turn Conversational Awareness: Maintain context across turns. If the user asks how you are, you respond, and they reply with "I'm good" or "Doing well", acknowledge it warmly (e.g., "Glad to hear that! What would you like to explore or discuss today?") rather than resetting or repeating greetings.
- Rules for Conversational Small Talk:
  - Do NOT repeatedly introduce yourself or your title ("I am the Nigerian Legal AI Assistant..."). Only state who you are if explicitly asked "who are you?".
  - Do NOT turn greetings or small talk into a pitch or forced invitation to ask legal questions.
  - Use natural, varied, and human-like phrasing. Avoid robotic, repetitive, or templated scripts.
  - NEVER mention your creator or founder's name in greetings or small talk (only disclose it if explicitly asked "who created you?" or "who is the founder?").

## 2. General Knowledge and Everyday Inquiries

- When users ask questions about general knowledge, science, mathematics, technology, literature, history, or daily life:
  - Answer directly, naturally, and authoritatively.
  - NEVER append a legal disclaimer, informational notice, or caveat to non-legal answers.
  - NEVER remind the user that you are primarily a legal AI or that the inquiry is outside the law.
  - Treat all general questions with rigor, depth, and helpfulness.
  - Explain concepts in accessible language and provide illustrative examples where helpful.

## 3. Nigerian Legal Specialization and Seamless Transition to Law

- Your specialized domain is the Nigerian legal, constitutional, and regulatory environment.
- When the user transitions from small talk or casual conversation to asking a legal question, transition seamlessly into your role as an authoritative Nigerian legal expert.
- Help users understand:
  - Nigerian statutes, statutory instruments, and judicial principles.
  - Individual rights, duties, and corporate compliance obligations (CAMA 2020, FCCPA 2018, Constitution 1999).
  - Business registration, commercial transactions, and agreements.
  - Data protection (NDPR), taxation, labor relations, tenancy, and property law.
- Rules for Legal Grounding:
  - Never fabricate statutory clauses, section numbers, judicial authorities, or URLs.
  - Always distinguish primary authorities (Acts of the National Assembly, State Laws, Court judgments) from secondary commentary.
  - When provided with verified statutory context from the internal knowledge base, base your answer strictly on that context and cite the exact Act and sections.
  - When provided with external legal research findings, synthesize them helpfully, cite the genuine source names with clickable markdown links, and clarify that the findings are based on authoritative external public research.
  - Always observe Nigerian federal versus state jurisdiction (e.g. state-level tenancy and premises recovery vs federal corporate law).

## 3. Document Analysis, Summarization, and Review

When reviewing, summarizing, or analyzing an attached document or agreement (contracts, leases, NDAs, etc.):
- Prioritize clarity, simplicity, and accessibility: explain what the document means in plain, easy-to-understand terms.
- Provide a concise summary and highlight key rights, obligations, or notable legal concerns/risks in clean bullet points.
- Avoid overwhelming walls of text or exhaustive, tedious clause dissection up front ("long talk").
- If the user asks to "shorten it" or asks for a brief summary, provide a tight, highly distilled response.
- Inform the user that you can provide an in-depth, clause-by-clause breakdown if they would like a deeper explanation.
- NEVER regurgitate template sample disclaimers (such as "Note: This is a fictional sample for educational purposes" or "For educational purposes only").

## 4. Communication Style, Tone, and Limitations

Your responses must be:
- Coherent, well-structured, friendly, and professional.
- Direct and concise: avoid rambling, unnecessary throat-clearing, repeating long prompts, or generating overwhelming walls of text ("long talk").
- Direct Answer First (Bottom-Line-Up-Front): Always give the clear, direct answer to the user's question in the very first 1–2 sentences before elaborating.
- Handling Multi-Part Questions (Multiple Questions in One Prompt):
  - When the user asks multiple questions or a long multi-part query, DO NOT lump everything into one messy paragraph.
  - Simplify and clearly label each question with distinct numbered headings or bold labels (e.g., **1. [Simplified Question 1]**, **2. [Simplified Question 2]**).
  - Provide a simplified, easy-to-understand answer directly under each corresponding question label, so the user immediately sees which answer belongs to which question.
- Focused and Digestible: Break down key legal points or statutory steps into clean, scannable bullet points rather than dense, unbroken paragraphs.
- Proportional Depth: Address exactly what the user asked in plain English, avoiding irrelevant historical background or tangential legal trivia.
- Offer Deeper Expansion: Conclude by letting the user know they can ask for deeper elaboration or step-by-step guidance on any specific point if desired.

Communicating Limitations Constructively:
- NEVER use repetitive, negative rejections such as:
  - "This is only for general informational purposes and should not be relied upon."
  - "This law is not included in my database."
  - "This law is unverified or pending counsel verification."
  - "I cannot provide legal advice."
  - "Note: This is a fictional sample for educational purposes."
  - "For educational purposes only."
- NEVER expose internal database status, ingestion status, quarantine safeguards, or pending counsel verification to users.
- Instead, communicate naturally, transparently, and context-sensitively:
  - Explain the governing legal framework helpfully.
  - If a specific provision cannot be conclusively established from available authoritative records, explain that specific uncertainty constructively without exposing internal database or development status.
  - Only when a user is seeking advice for an active dispute, personal litigation, or contentious dispute resolution, advise consulting a qualified Nigerian legal practitioner naturally within your explanation.

## 5. Creator and Founder Identity

- In normal greetings, introductions, or general conversation, NEVER introduce yourself with your creator or founder's name.
- ONLY when a user explicitly asks who created, built, founded, or developed you (e.g., "who created you?", "who is the founder?", "who built this?"):
  Respond using this exact wording:

"I was created by Dike Mmesoma, a passionate 400-level Software Engineering student in Nigeria. Her vision was simple but powerful: to bridge the legal knowledge gap across Nigeria, making legal understanding accessible to law students, lawyers, business owners, and everyday citizens. She built me to be a friendly, reliable companion in navigating Nigerian law—so you don’t have to feel overwhelmed by complex statutes or jargon. I'm proud to be part of that mission, and I'm here to help whenever you need."

Do not add any legal disclaimer or caveat to this response.

## 6. Accuracy, Honesty, and Research Provenance

- Never knowingly present unverified claims as settled law.
- Distinguish settled law, judicial interpretations, and pending legislative bills.
- Answer user queries authoritatively and helpfully using provided statutory and legal research context, citing primary enactments and reputable sources with clickable links.
- Real-Time Search Grounding: When live web search results are provided in the prompt, you MUST prioritize and ground your answer in those live search results over outdated pretraining knowledge.
- For questions about current officeholders, leaders, latest software versions, and real-time events, never assert outdated pre-cutoff data as current fact.
- If real-time search was needed but results could not be retrieved, communicate that limitation transparently rather than guessing or asserting outdated pre-cutoff knowledge as current fact.
- Never claim to have searched the web or retrieved documents unless an actual search or extraction tool was executed in that turn."""


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_deepseek_config() -> Tuple[str, str, str, int, int]:
    """
    Retrieves and validates DeepSeek settings from Django configuration.

    Returns:
        Tuple of (api_key, base_url, model, timeout_seconds, max_tokens)

    Raises:
        AIConfigurationError: When DEEPSEEK_API_KEY is missing or empty.
    """
    api_key = getattr(settings, 'DEEPSEEK_API_KEY', '')
    if not isinstance(api_key, str) or not api_key.strip():
        logger.error("DEEPSEEK_API_KEY is not configured or empty.")
        raise AIConfigurationError(
            "DeepSeek API key is not configured. Please set DEEPSEEK_API_KEY in backend/.env."
        )

    base_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    model = getattr(settings, 'DEEPSEEK_MODEL', 'deepseek-chat')
    timeout = int(getattr(settings, 'DEEPSEEK_TIMEOUT', 120))
    max_tokens = int(getattr(settings, 'DEEPSEEK_MAX_TOKENS', 3500))

    return api_key.strip(), base_url, model, timeout, max_tokens


def _build_payload(
    user_message: str,
    model: str,
    stream: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 3500,
    legal_prompt_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the standard OpenAI-compatible payload for DeepSeek API.
    Injects conversation history turns and optional verified statutory context
    between system prompt and the current user message.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject verified legal knowledge context if present
    if legal_prompt_context and legal_prompt_context.strip():
        messages.append({"role": "system", "content": legal_prompt_context.strip()})

    if history and isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": user_message})

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": stream
    }


def _execute_deepseek_request(
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    stream: bool = False
) -> requests.Response:
    """
    Executes the HTTP POST request to DeepSeek with status, retry, and network error verification.
    Returns the raw response object.
    """
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout, stream=stream)
            break
        except requests.exceptions.Timeout as exc:
            logger.error("DeepSeek request timed out after %s seconds.", timeout)
            raise AITimeoutError(
                "DeepSeek API request timed out. Please try again shortly."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            if attempt < max_retries:
                logger.warning(
                    "Connection error to %s on attempt %d/%d: %s. Retrying in 1s...",
                    endpoint, attempt + 1, max_retries + 1, exc
                )
                time.sleep(1)
                continue
            logger.error("Failed to connect to DeepSeek at %s after %d attempts: %s", endpoint, max_retries + 1, exc)
            raise AIServiceError(
                "Cannot connect to DeepSeek API. Please verify network connectivity."
            ) from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Network error during DeepSeek request: %s", exc)
            raise AIServiceError(f"Network error communicating with AI provider: {exc}") from exc

    # Status-specific error resolution
    if response.status_code == 401:
        logger.error("DeepSeek authentication failed (HTTP 401). Invalid API key.")
        raise AIAuthenticationError(
            "Authentication failed: Invalid DeepSeek API key. Check backend/.env."
        )
    elif response.status_code == 402:
        logger.error("DeepSeek account insufficient balance (HTTP 402).")
        raise AIBalanceError(
            "DeepSeek account balance is insufficient. Please check credits on platform.deepseek.com."
        )
    elif response.status_code == 429:
        logger.warning("DeepSeek rate limit reached (HTTP 429).")
        raise AIRateLimitError(
            "DeepSeek API rate limit reached. Please retry in a few moments."
        )
    elif response.status_code >= 500:
        logger.error("DeepSeek server error (HTTP %s).", response.status_code)
        raise AIServiceError(
            f"DeepSeek service encountered an internal error (HTTP {response.status_code})."
        )
    elif response.status_code >= 400:
        logger.error("DeepSeek client error (HTTP %s): %s", response.status_code, response.text[:200])
        raise AIServiceError(f"DeepSeek returned client error (HTTP {response.status_code}).")

    # Enforce UTF-8 decoding on all response streams to prevent emoji/symbol corruption
    response.encoding = "utf-8"
    return response


def _extract_reply_content(data: Dict[str, Any]) -> str:
    """Safely extracts the generated assistant text from the DeepSeek response payload."""
    if "error" in data:
        err_msg = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        logger.error("AI provider returned error in response: %s", err_msg)
        raise AIServiceError(f"AI Provider Error: {err_msg}")

    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        logger.error("DeepSeek response missing 'choices' list.")
        raise AIResponseError("DeepSeek returned an empty choices list.")

    first_choice = choices[0]
    finish_reason = first_choice.get("finish_reason")
    if finish_reason == "length":
        logger.warning("DeepSeek output token limit reached (finish_reason='length').")
        raise AIResponseTruncatedError(
            "The AI model reached its maximum output token limit (finish_reason='length') and could not complete the response. Please ask a more specific question or request a shorter answer."
        )

    message_obj = first_choice.get("message")
    if not isinstance(message_obj, dict):
        logger.error("DeepSeek choice missing 'message' object.")
        raise AIResponseError("DeepSeek response format invalid.")

    content = message_obj.get("content")
    if not isinstance(content, str) or not content.strip():
        # Fallback to reasoning_content if emitted solely in reasoning channel
        reasoning = message_obj.get("reasoning_content") or message_obj.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        logger.error("DeepSeek message content is empty.")
        raise AIResponseError("DeepSeek returned empty message content.")

    return content.strip()


# ==============================================================================
# Response Sanitization & Disclaimer Cleanup
# ==============================================================================

EXACT_UNVERIFIED_DISCLOSURE = (
    "This is an official-agency-hosted reprint that has not yet been "
    "independently authenticated against the original Official Gazette or reviewed by legal counsel."
)


def sanitize_chat_response(
    text: str,
    is_unverified_mode: bool = False,
    is_external_research: bool = False,
    is_legal_query: bool = False,
) -> str:
    """
    Cleans and refines the AI response across all conversation modes:
    1. If in unverified mode (e.g. CAMA 2020 reprint):
       - Ensures the front-loaded unverified notice is at the very beginning.
       - Strips any claims asserting Section 106 of the Evidence Act requires counsel certification.
       - Strips duplicate trailing reprint notices.
    2. Strips repetitive generic boilerplate disclaimers across all responses:
       - 'This is for general educational and informational purposes only and does not constitute formal legal advice...'
       - 'Consult a qualified Nigerian legal practitioner for advice specific to your circumstances.'
       - 'This response is provided for general knowledge only.'
       - 'This response is for informational purposes only.'
    3. Preserves all accurate subsection references (e.g. Section 18(1)) and genuine markdown links.
    """
    if not text:
        return text

    cleaned = text

    # 1. Remove/replace any claims asserting Section 106 of the Evidence Act requires counsel certification
    if "section 106" in cleaned.lower():
        cleaned = re.sub(
            r'(?:certified\s+by\s+legal\s+counsel\s+)?under\s+Section\s+106\s+(?:of\s+)?(?:the\s+)?Evidence\s+Act(?:\s+2011)?',
            'independently authenticated against the original Official Gazette or reviewed by legal counsel',
            cleaned,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r'Section\s+106\s+(?:of\s+)?(?:the\s+)?Evidence\s+Act(?:\s+2011)?',
            'independent official gazette authentication',
            cleaned,
            flags=re.IGNORECASE
        )

    # 2. Remove repetitive generic disclaimers or duplicate reprint notices at the end
    # Targets strictly generic boilerplate; preserves substantive legal qualifications, amendment notes, and warnings.
    trailing_disclaimer_patterns = [
        # Full boilerplate block starting with '--- \n Disclaimer:' or 'Disclaimer: This is for general educational...'
        r'(?:\n\s*---\s*)?\n\s*(?:\*{1,2})?(?:Disclaimer|Legal Disclaimer)(?:\*{1,2})?:?\s*(?:This (?:is|response is|information is) for general (?:educational and )?informational purposes only|Consult a qualified Nigerian legal practitioner)[^\n]*(?:\n\s*(?:Consult a qualified Nigerian legal practitioner|This is for general educational)[^\n]*)*[*\s]*$',
        r'(?:\n\s*---\s*)?\n\s*This is for general educational and informational purposes only and does not constitute formal legal advice\.[^\n]*[*\s]*$',
        # Trailing generic knowledge caveat (strictly targeting boilerplate disclaimer, not substantive notes)
        r'(?:\n\s*---\s*)?\n\s*(?:\*{1,2})?(?:Notice|Note|Important Note)(?:\*{1,2})?:?\s*(?:This (?:response|information) is (?:provided )?for general knowledge(?: only)?|This response is for informational purposes only)\.?[*\s]*$',
        # Trailing duplicate reprint disclaimer or note
        r'(?:\n\s*---\s*)?\n\s*(?:\*{1,2})?(?:Notice|Note|Important Note|Disclaimer)(?:\*{1,2})?:?\s*(?:This (?:response|information|document|Act|provision) (?:is|was) (?:based on|retrieved from) an official-agency-hosted reprint|This is an official-agency-hosted reprint that has not yet been independently authenticated)[^\n]*[*\s]*$',
        # Trailing fictional sample or educational boilerplate disclaimer
        r'(?:\n\s*---\s*)?\n\s*(?:\*{1,2})?(?:Note|Notice|Disclaimer)(?:\*{1,2})?:?\s*(?:This\s+is\s+a\s+(?:fictional\s+)?sample\s+(?:for\s+educational\s+purposes(?:\s+only)?)|(?:This is |For )educational purposes only)\.?[*\s]*$',
    ]
    for pattern in trailing_disclaimer_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).rstrip()

    # 3. Ensure the exact unverified-source disclosure is at the beginning for unverified mode
    if is_unverified_mode:
        disclosure_prefix = f"> **Notice**: {EXACT_UNVERIFIED_DISCLOSURE}\n\n"
        if EXACT_UNVERIFIED_DISCLOSURE.lower() not in cleaned[:350].lower():
            cleaned = disclosure_prefix + cleaned.lstrip()
    else:
        # 4. Strip any internal quarantine/database metadata leaks from ordinary user-facing responses
        internal_leak_patterns = [
            r'(?:Note:\s*)?(?:The\s+)?(?:internal\s+)?(?:verified\s+)?statutory\s+database\s+does\s+not\s+(?:currently\s+)?have\s+this\s+statute\s+(?:ingested|in\s+its\s+database)[^\n]*\n*',
            r'(?:Note:\s*)?This\s+law\s+is\s+(?:currently\s+)?not\s+(?:included\s+in|in)\s+(?:our|my)\s+(?:internal\s+)?database[^\n]*\n*',
            r'(?:Note:\s*)?This\s+(?:statute|Act|provision|document)\s+is\s+(?:currently\s+)?(?:quarantined|pending\s+(?:counsel\s+)?verification|unverified)[^\n]*\n*',
            r'(?:Note:\s*)?This\s+is\s+an\s+official-agency-hosted\s+reprint\s+that\s+has\s+not\s+yet\s+been\s+independently\s+authenticated[^\n]*\n*',
        ]
        for p in internal_leak_patterns:
            cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()

    return cleaned


# Backwards-compatibility alias for unverified-mode callers and existing tests
def sanitize_unverified_response(text: str) -> str:
    return sanitize_chat_response(text, is_unverified_mode=True)


# ==============================================================================
# Public API Functions
# ==============================================================================

def generate_legal_response(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    legal_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synchronous entrypoint for generating structured legal AI responses via DeepSeek API.

    Args:
        message: The user's input text query.
        history: Optional list of previous conversation turns.
        legal_context: Optional dictionary containing verified legal prompt context and citations.

    Returns:
        dict: {"reply": "<markdown formatted response>", "mode": "legal", "citations": [...]}
    """
    clean_message = message.strip() if isinstance(message, str) else ""
    if not clean_message:
        return {"reply": "", "mode": "legal"}

    api_key, base_url, model, timeout, max_tokens = _get_deepseek_config()
    endpoint = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Nigerian Legal AI Assistant",
    }

    prompt_ctx = legal_context.get('prompt_context', '') if legal_context else None
    payload = _build_payload(
        clean_message,
        model,
        stream=False,
        history=history,
        max_tokens=max_tokens,
        legal_prompt_context=prompt_ctx,
    )
    response = _execute_deepseek_request(endpoint, headers, payload, timeout, stream=False)

    try:
        response_data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("DeepSeek returned invalid JSON.")
        raise AIResponseError("DeepSeek returned a malformed response.") from exc

    reply_text = _extract_reply_content(response_data)

    # Sanitize and validate any citation tokens
    citations_list = []
    if legal_context and legal_context.get('active_citation_tokens'):
        from rag.services.citation_service import filter_and_validate_citations
        reply_text, valid_cits, rejected_cits = filter_and_validate_citations(
            reply_text,
            legal_context['active_citation_tokens'],
            citations=legal_context.get('citations'),
            render_clickable=True,
        )
        citations_list = legal_context.get('citations', [])

    # Sanitize response and strip repetitive generic boilerplate disclaimers
    is_unverified = bool(legal_context and legal_context.get('is_unverified_mode'))
    is_external = bool(legal_context and legal_context.get('is_external_research'))
    is_legal = bool(legal_context and legal_context.get('is_legal_query'))
    reply_text = sanitize_chat_response(
        reply_text,
        is_unverified_mode=is_unverified,
        is_external_research=is_external,
        is_legal_query=is_legal,
    )

    result_dict = {"reply": reply_text, "mode": "legal"}
    if citations_list:
        result_dict["citations"] = citations_list
    return result_dict


def stream_legal_response_events(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    legal_context: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Event-driven streaming generator for DeepSeek API.
    Emits structured status transitions and the final verified response.
    """
    clean_message = message.strip() if isinstance(message, str) else ""
    if not clean_message:
        yield ("result", {"reply": "", "mode": "legal"})
        return

    # 1. Immediate initial status
    yield ("status", {"status": "Thinking..."})

    api_key, base_url, model, timeout, max_tokens = _get_deepseek_config()
    endpoint = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "Nigerian Legal AI Assistant",
    }

    prompt_ctx = legal_context.get('prompt_context', '') if legal_context else None
    payload = _build_payload(
        clean_message,
        model,
        stream=True,
        history=history,
        max_tokens=max_tokens,
        legal_prompt_context=prompt_ctx,
    )
    response = _execute_deepseek_request(endpoint, headers, payload, timeout, stream=True)

    accumulated_content = []
    reasoning_content = []
    has_transitioned_status = False
    last_finish_reason = None
    saw_done = False
    in_stream_error = None

    try:
        # Read the SSE lines emitted by DeepSeek/OpenRouter
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode('utf-8', errors='replace')
            line = line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                saw_done = True
                break

            try:
                chunk_obj = json.loads(data_str)
            except (ValueError, json.JSONDecodeError):
                continue

            if not isinstance(chunk_obj, dict):
                continue

            # Catch in-stream errors emitted by OpenRouter or provider
            if "error" in chunk_obj:
                err_dict = chunk_obj["error"]
                err_msg = err_dict.get("message") if isinstance(err_dict, dict) else str(err_dict)
                logger.error("In-stream error from AI provider: %s", err_msg)
                in_stream_error = err_msg
                break

            choices = chunk_obj.get("choices", [])
            if not choices:
                continue

            choice_0 = choices[0]
            finish_reason = choice_0.get("finish_reason")
            if finish_reason:
                last_finish_reason = finish_reason

            delta = choice_0.get("delta", {})
            token = delta.get("content") or delta.get("text") or ""
            reasoning_chunk = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("thought")
                or ""
            )

            # Handle reasoning_details list format if string is empty
            if not reasoning_chunk and isinstance(delta.get("reasoning_details"), list):
                detail_texts = [
                    d.get("text", "") for d in delta["reasoning_details"]
                    if isinstance(d, dict) and d.get("text")
                ]
                reasoning_chunk = "".join(detail_texts)

            if reasoning_chunk:
                reasoning_content.append(reasoning_chunk)

            if token:
                accumulated_content.append(token)

                # Transition to "Preparing your answer..." on first tokens
                if not has_transitioned_status and len(accumulated_content) >= 1:
                    has_transitioned_status = True
                    yield ("status", {"status": "Preparing your answer..."})

    except requests.exceptions.RequestException as exc:
        logger.error("Stream interrupted during read: %s", exc)
        raise AIServiceError(f"Stream interrupted: {exc}") from exc
    finally:
        response.close()

    # 1. Explicit in-stream error from provider
    if in_stream_error:
        raise AIServiceError(f"AI Provider Error: {in_stream_error}")

    # 2. Reached token limit before completing
    if last_finish_reason == "length":
        logger.warning("DeepSeek output token limit reached (finish_reason='length').")
        raise AIResponseTruncatedError(
            "The AI model reached its maximum output token limit (finish_reason='length') and could not complete the response. Please ask a more specific question or request a shorter answer."
        )

    # 3. Stream disconnected prematurely without terminal finish reason or [DONE]
    if not saw_done and last_finish_reason not in ("stop", "end_turn", "stop_sequence"):
        logger.error(
            "AI stream closed prematurely without [DONE] (last_finish_reason=%s, accumulated_tokens=%d)",
            last_finish_reason,
            len(accumulated_content)
        )
        raise AIServiceError("The AI stream connection was interrupted before the response could be completed.")

    full_reply = "".join(accumulated_content).strip()
    if not full_reply and reasoning_content:
        # Fallback to synthesized reasoning if the provider emitted all text in reasoning channel
        full_reply = "".join(reasoning_content).strip()

    if not full_reply:
        logger.error("DeepSeek stream finished without emitting text content.")
        raise AIResponseError("DeepSeek returned an empty response.")

    # Sanitize and validate any citation tokens
    citations_list = []
    if legal_context and legal_context.get('active_citation_tokens'):
        from rag.services.citation_service import filter_and_validate_citations
        full_reply, valid_cits, rejected_cits = filter_and_validate_citations(
            full_reply,
            legal_context['active_citation_tokens'],
            citations=legal_context.get('citations'),
            render_clickable=True,
        )
        citations_list = legal_context.get('citations', [])

    # Sanitize response and strip repetitive generic boilerplate disclaimers
    is_unverified = bool(legal_context and legal_context.get('is_unverified_mode'))
    is_external = bool(legal_context and legal_context.get('is_external_research'))
    is_legal = bool(legal_context and legal_context.get('is_legal_query'))
    full_reply = sanitize_chat_response(
        full_reply,
        is_unverified_mode=is_unverified,
        is_external_research=is_external,
        is_legal_query=is_legal,
    )

    result_dict = {"reply": full_reply, "mode": "legal"}
    if citations_list:
        result_dict["citations"] = citations_list
    yield ("result", result_dict)
