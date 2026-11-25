"""
Automated Test Suite for Chatbot Response Intelligence, System Prompt Audit & Web Research Fallback.

Tests the 10 Realistic Scenarios mandated by Phase 4/5:
1. Ordinary non-legal questions (no legal disclaimer, no meta-announcement).
2. Nigerian legal questions supported by verified/quarantined CAMA provisions (clickable citations, accurate subsections).
3. Questions outside internal statutory coverage (triggers intelligent legal research, no "law not in database" rejection).
4. Questions requiring external legal research with genuine clickable links.
5. Questions where online sources disagree or vary across jurisdictions.
6. Questions involving outdated/repealed laws (distinguishing historical vs current legislation).
7. Questions involving state-specific legislation (identifying state jurisdiction vs federal).
8. Questions with no reliable available references (transparent, constructive uncertainty without fabrication).
9. Questions where the model cannot establish an answer (transparent communication of legislative status).
10. Provider timeout and web-search failure (resilient, graceful handling).
"""

import json
import datetime
from unittest.mock import patch, MagicMock
import requests
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from chat.models import Conversation, Message
from chat.services import (
    generate_legal_response,
    stream_legal_response_events,
    sanitize_chat_response,
    AIServiceError,
    AITimeoutError,
)
from chat.web_search import (
    classify_legal_source_tier,
    perform_legal_web_search,
    format_legal_research_prompt,
    should_search_web,
)
from rag.models import (
    LegalSourceDocument,
    LegalProvisionChunk,
    Jurisdiction,
    VerificationStatus,
)

User = get_user_model()


class ResponseIntelligenceTests(APITestCase):
    """
    Exhaustive unit and integration tests for Chatbot Response Intelligence & Web Research Fallback.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="intelligence_user",
            email="intel@example.com",
            password="TestPassword123!"
        )
        self.client.force_authenticate(user=self.user)

        # Setup CAMA 2020 in test DB
        self.cama_doc = LegalSourceDocument.objects.create(
            statute_family_code="CAMA-2020",
            title="Companies and Allied Matters Act, 2020",
            short_title="CAMA 2020",
            jurisdiction=Jurisdiction.FEDERAL,
            effective_from=datetime.date(2020, 8, 7),
            is_verified_authentic=False,
            verification_status=VerificationStatus.PENDING,
            source_url="https://www.nipc.gov.ng/wp-content/uploads/2020/12/Companies-and-Allied-Matters-Act-2020.pdf",
            sha256_checksum="a" * 64,
        )
        self.cama_chunk_18 = LegalProvisionChunk.objects.create(
            document=self.cama_doc,
            section_number="Section 18",
            section_title="Right to form a company",
            content="18.(1) As from the commencement of this Act, any two or more persons may form and incorporate a company by complying with the requirements of this Act in respect of registration of company.\n(2) Notwithstanding the provisions of subsection (1), one person may form and incorporate a private company by complying with the requirements of this Act in respect of private company.",
            citation_label="Section 18, CAMA 2020",
            chunk_index=18,
            embedding=[0.05] * 384,
        )

    # --------------------------------------------------------------------------
    # Scenario 1: Ordinary Non-Legal Question
    # --------------------------------------------------------------------------
    @patch("chat.services.requests.post")
    def test_scenario_1_ordinary_non_legal_question(self, mock_post):
        """Ordinary non-legal question receives helpful answer with zero legal disclaimers."""
        raw_ai_reply = (
            "Machine learning is a field of artificial intelligence. Supervised learning uses labeled "
            "data to train models, while unsupervised learning discovers patterns in unlabeled data."
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": raw_ai_reply}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Explain the difference between supervised and unsupervised learning"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")

        # Must answer the question
        self.assertIn("Supervised learning", reply)
        # Must NOT append boilerplate legal disclaimer
        self.assertNotIn("This is for general educational and informational purposes only", reply)
        self.assertNotIn("Consult a qualified Nigerian legal practitioner", reply)
        # Must NOT unnecessarily remind the user it is a legal AI
        self.assertNotIn("While I specialize in Nigerian law", reply)

    # --------------------------------------------------------------------------
    # Scenario 2: Ordinary User Inquiring About CAMA 2020 (External Research Triggered)
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_2_cama_receives_external_research_without_quarantine_leak(self, mock_post, mock_search):
        """CAMA questions receive external legal research rather than an internal quarantine notice, while staying quarantined in RAG."""
        mock_search.return_value = [
            {
                "title": "Corporate Affairs Commission - CAMA 2020 Incorporation",
                "url": "https://cac.gov.ng/cama2020",
                "snippet": "Under Section 18(2) of CAMA 2020, one person may form and incorporate a private company.",
                "tier_number": 4,
                "tier_label": "Official Regulatory Agency",
                "domain": "cac.gov.ng",
                "publication_date": "2020",
            }
        ]
        simulated_reply = (
            "Yes, under Section 18(2) of the [Companies and Allied Matters Act, 2020 (CAMA 2020)](https://cac.gov.ng/cama2020), "
            "one person may form and incorporate a private company by complying with the requirements of the Act."
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": simulated_reply}}]
        }
        mock_post.return_value = mock_resp

        # Normal ordinary user chat request without allow_unverified
        response = self.client.post(
            "/api/chat/",
            data={"message": "Can one person form a company under CAMA 2020?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # 1. External research must be automatically triggered
        self.assertTrue(mock_search.called, "CAMA inquiry must trigger external research fallback.")
        # 2. Quarantined CAMA content must not be injected into model prompt payload
        call_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        system_content = call_payload["messages"][0]["content"]
        self.assertNotIn("UNVERIFIED PROVISION", system_content)
        # 3. Model response contains helpful answer with genuine clickable link
        reply = response.data.get("reply", "")
        self.assertIn("Section 18(2)", reply)
        self.assertIn("https://cac.gov.ng/cama2020", reply)
        # 4. Zero internal quarantine metadata leaks
        self.assertNotIn("official-agency-hosted reprint", reply)
        self.assertNotIn("pending verification", reply.lower())
        self.assertNotIn("not in my database", reply.lower())
        self.assertNotIn("quarantined", reply.lower())
        # 5. Database document remains strictly quarantined
        cama_db = LegalSourceDocument.objects.get(statute_family_code="CAMA-2020")
        self.assertFalse(cama_db.is_verified_authentic)
        self.assertEqual(cama_db.verification_status, VerificationStatus.PENDING)

    @patch("chat.services.requests.post")
    def test_developer_explicit_unverified_mode_preview(self, mock_post):
        """Developer/admin workflow: explicit allow_unverified=True retrieves reprint preview with audit disclosure."""
        simulated_reply = (
            "Under Section 18(1) [CITATION: CIT-FED-CAMA2020-SEC18], any two or more persons may form a company. "
            "Furthermore, Section 18(2) permits one person to form a private company."
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": simulated_reply}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={
                "message": "Can one person form a company under Section 18 of CAMA 2020?",
                "allow_unverified": True,
            },
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")
        # In explicit developer mode only: audit reprint notice is present
        self.assertTrue(reply.startswith("> **Notice**: This is an official-agency-hosted reprint"))
        self.assertIn("[Section 18, CAMA 2020](https://www.nipc.gov.ng/wp-content/uploads/2020/12/Companies-and-Allied-Matters-Act-2020.pdf)", reply)

    # --------------------------------------------------------------------------
    # Scenario 3: Questions Outside Current Internal Statutory Coverage
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_3_legal_question_outside_internal_coverage_triggers_research(self, mock_post, mock_search):
        """Question outside internal RAG triggers legal research fallback without dismissive rejections."""
        mock_search.return_value = [
            {
                "title": "Tenancy Law of Lagos State 2011",
                "url": "http://lagosministryofjustice.org/wp-content/uploads/2022/01/Tenancy-Law-2011.pdf",
                "snippet": "Law to regulate rights and obligations under tenancy agreements and recovery of premises in Lagos State.",
                "tier_number": 2,
                "tier_label": "Federal/State Ministry or Legislature",
                "domain": "lagosministryofjustice.org",
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Under the [Lagos State Tenancy Law 2011](http://lagosministryofjustice.org/wp-content/uploads/2022/01/Tenancy-Law-2011.pdf), a landlord must issue a statutory notice to quit before recovering premises."}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "What is the notice period for recovering premises in Lagos State?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(mock_search.called, "Legal web research must be triggered when statute is not in internal DB.")
        reply = response.data.get("reply", "")

        # Does NOT have the negative robotic refusal
        self.assertNotIn("This law is not included in my database", reply)
        self.assertNotIn("CRITICAL LEGAL VERIFICATION NOTICE", reply)
        # Contains the helpful synthesized answer with clickable genuine link
        self.assertIn("[Lagos State Tenancy Law 2011](http://lagosministryofjustice.org/wp-content/uploads/2022/01/Tenancy-Law-2011.pdf)", reply)

    # --------------------------------------------------------------------------
    # Scenario 4: Questions Requiring External Legal Research with Clickable Links
    # --------------------------------------------------------------------------
    def test_scenario_4_legal_source_hierarchy_ranking(self):
        """classify_legal_source_tier enforces exact hostname validation, trusted allowlists, and anti-spoofing."""
        # Tier 1: Authentic National Assembly & Official Gazette
        t1, l1 = classify_legal_source_tier("https://nass.gov.ng/documents/bill123")
        self.assertEqual(t1, 1)
        self.assertEqual(l1, "Official Government Publication & Gazette")

        # gazettes.africa is Tier 5 repository, NOT Tier 1!
        t_gaz, l_gaz = classify_legal_source_tier("https://gazettes.africa/gazettes/ng")
        self.assertEqual(t_gaz, 5)
        self.assertEqual(l_gaz, "Reputable Legal Research Platform")

        # nigerialawguru.com is Tier 5 repository, NOT Tier 1!
        t_nlg, l_nlg = classify_legal_source_tier("https://nigerialawguru.com/acts/cama.html")
        self.assertEqual(t_nlg, 5)
        self.assertEqual(l_nlg, "Reputable Legal Research Platform")

        # Substring spoofing prevention: fake-site.com/nass.gov.ng is NOT Tier 1!
        t_fake, l_fake = classify_legal_source_tier("https://fake-site.com/nass.gov.ng/laws")
        self.assertEqual(t_fake, 6)
        self.assertEqual(l_fake, "Unverified Web Source (Caution: Unauthenticated)")

        # Subdomain spoofing prevention: evilsite.gov.ng.attacker.com is NOT Tier 2!
        t_spoof, l_spoof = classify_legal_source_tier("https://evilsite.gov.ng.attacker.com/acts")
        self.assertEqual(t_spoof, 6)
        self.assertEqual(l_spoof, "Unverified Web Source (Caution: Unauthenticated)")

        # Tier 2: State Ministries of Justice & State Assemblies
        t2, l2 = classify_legal_source_tier("http://lagosministryofjustice.org/laws")
        self.assertEqual(t2, 2)
        self.assertIn("Ministry", l2)

        # Tier 3: Superior Courts of Record
        t3, l3 = classify_legal_source_tier("https://supremecourt.gov.ng/judgment/sc1")
        self.assertEqual(t3, 3)
        self.assertEqual(l3, "Nigerian Judicial & Court Record")

        # Tier 4: Regulatory Agencies
        t4, l4 = classify_legal_source_tier("https://ndpc.gov.ng/guidelines")
        self.assertEqual(t4, 4)
        self.assertEqual(l4, "Official Regulatory Agency")

        # Tier 5: Reputable Legal Research Repositories (PLAC)
        t5, l5 = classify_legal_source_tier("https://placng.org/acts/ndpa2023.pdf")
        self.assertEqual(t5, 5)
        self.assertEqual(l5, "Reputable Legal Research Platform")

        # Tier 6: Unknown website is NOT classified as authoritative!
        t6, l6 = classify_legal_source_tier("https://randomblog12345.com/legal-tips")
        self.assertEqual(t6, 6)
        self.assertEqual(l6, "Unverified Web Source (Caution: Unauthenticated)")

    # --------------------------------------------------------------------------
    # Scenario 5: Questions Where Online Sources Disagree
    # --------------------------------------------------------------------------
    def test_scenario_5_format_legal_research_prompt_instructs_objective_synthesis(self):
        """External research prompt instructs model to distinguish primary from secondary sources and resolve variances."""
        sample_results = [
            {
                "title": "Child's Rights Act 2003",
                "url": "https://placng.org/acts/cra2003.pdf",
                "snippet": "Adoption provisions under the Federal Child's Rights Act.",
                "tier_number": 5,
                "tier_label": "Reputable Legal Research Platform",
                "domain": "placng.org"
            },
            {
                "title": "Adoption Law of Lagos State",
                "url": "http://lagosministryofjustice.org/adoption",
                "snippet": "State-specific adoption regulations in Lagos State.",
                "tier_number": 2,
                "tier_label": "Federal/State Ministry or Legislature",
                "domain": "lagosministryofjustice.org"
            }
        ]
        prompt = format_legal_research_prompt("Can a single parent adopt in Nigeria?", sample_results)
        self.assertIn("Tier 2", prompt)
        self.assertIn("Tier 5", prompt)
        self.assertIn("PRIMARY VS SECONDARY", prompt)
        self.assertIn("JURISDICTION & COMMENCEMENT DETAILS", prompt)
        self.assertIn("GENUINE CLICKABLE CITATIONS", prompt)

    # --------------------------------------------------------------------------
    # Scenario 6: Questions Involving Outdated/Repealed Laws
    # --------------------------------------------------------------------------
    @patch("chat.services.requests.post")
    def test_scenario_6_outdated_laws_distinguished(self, mock_post):
        """Model explains legislative transition when asked about repealed laws."""
        simulated_reply = (
            "Under the repealed CAMA 1990, a minimum of two persons was required to incorporate a company. "
            "However, under CAMA 2020, Section 18(2) permits a single person to form a private company."
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": simulated_reply}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Did CAMA 1990 allow a single shareholder?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")
        self.assertIn("repealed CAMA 1990", reply)
        self.assertIn("Section 18(2)", reply)

    # --------------------------------------------------------------------------
    # Scenario 7: Questions Involving State-Specific Legislation
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_7_state_specific_legislation_handling(self, mock_post, mock_search):
        """State-specific legal inquiry identifies state jurisdiction and provides state sources."""
        mock_search.return_value = [
            {
                "title": "Tenancy Law 2011 - Lagos State",
                "url": "http://lagosministryofjustice.org/wp-content/uploads/2022/01/Tenancy-Law-2011.pdf",
                "snippet": "Section 13 requires 6 months notice for a yearly tenant in Lagos.",
                "tier_number": 2,
                "tier_label": "Federal/State Ministry or Legislature",
                "domain": "lagosministryofjustice.org",
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "In Lagos State, tenancy is governed by the [Lagos State Tenancy Law 2011](http://lagosministryofjustice.org/wp-content/uploads/2022/01/Tenancy-Law-2011.pdf). For a yearly tenant, Section 13 stipulates a 6-month notice period."}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "How many months notice for a yearly tenant in Lagos?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")
        self.assertIn("Lagos State", reply)
        self.assertIn("6-month notice", reply)
        self.assertIn("lagosministryofjustice.org", reply)

    # --------------------------------------------------------------------------
    # Scenario 8: Questions With No Reliable Available References
    # --------------------------------------------------------------------------
    def test_scenario_8_no_reliable_references_constructive_prompt(self):
        """When research returns zero hits, prompt guides constructive explanation without dismissive refusal or metadata leaks."""
        prompt = format_legal_research_prompt("What does the 2026 Interplanetary Mining Act of Nigeria state?", [])
        self.assertIn("NIGERIAN LEGAL RESEARCH CONTEXT", prompt)
        self.assertIn("NEVER use repetitive, robotic rejections", prompt)
        self.assertIn("Never expose internal system, database, or ingestion status to the user", prompt)

    # --------------------------------------------------------------------------
    # Scenario 9: Transparent Uncertainty Handling Without Dismissiveness
    # --------------------------------------------------------------------------
    def test_scenario_9_sanitizer_removes_repetitive_disclaimers_across_modes(self):
        """sanitize_chat_response strips boilerplate footers while preserving substantive explanations."""
        model_output_general = (
            "The capital of Australia is Canberra, which was chosen as a compromise between Sydney and Melbourne.\n\n"
            "---\n*Disclaimer: This is for general educational and informational purposes only and does not constitute formal legal advice. Consult a qualified Nigerian legal practitioner for advice specific to your circumstances.*"
        )
        cleaned_general = sanitize_chat_response(model_output_general, is_legal_query=False)
        self.assertIn("Canberra", cleaned_general)
        self.assertNotIn("This is for general educational and informational purposes only", cleaned_general)
        self.assertNotIn("Consult a qualified Nigerian legal practitioner", cleaned_general)

        model_output_knowledge_caveat = (
            "Binary search has a logarithmic time complexity of O(log n).\n\n"
            "*Note: This response is provided for general knowledge only.*"
        )
        cleaned_knowledge = sanitize_chat_response(model_output_knowledge_caveat, is_legal_query=False)
        self.assertIn("O(log n)", cleaned_knowledge)
        self.assertNotIn("provided for general knowledge only", cleaned_knowledge)

        model_output_sample_disclaimer = (
            "Here is the summary of the tenancy agreement.\n\n"
            "Note: This is a fictional sample for educational purposes."
        )
        cleaned_sample = sanitize_chat_response(model_output_sample_disclaimer, is_legal_query=True)
        self.assertIn("Here is the summary", cleaned_sample)
        self.assertNotIn("This is a fictional sample for educational purposes", cleaned_sample)

    # --------------------------------------------------------------------------
    # Scenario 10: Provider Timeout and Web-Search Failure
    # --------------------------------------------------------------------------
    @patch("chat.services.requests.post")
    def test_scenario_10_provider_timeout_handled_gracefully(self, mock_post):
        """AI provider timeout raises AIServiceError and view returns HTTP 503 without crashing."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out")
        response = self.client.post(
            "/api/chat/",
            data={"message": "Explain Nigerian patent law"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
        self.assertIn("error", response.data)

    # --------------------------------------------------------------------------
    # Regression Tests: CAMA Quarantine Protection Across Chat Flows
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_quarantined_cama_excluded_from_normal_single_turn_chat(self, mock_post, mock_search):
        """In normal single-turn chat, quarantined CAMA is excluded and web research fallback is invoked."""
        mock_search.return_value = [
            {
                "title": "Corporate Affairs Commission Guidelines",
                "url": "https://cac.gov.ng/guidelines",
                "snippet": "Information on incorporating single-member companies.",
                "tier_number": 4,
                "tier_label": "Official Regulatory Agency",
                "domain": "cac.gov.ng",
                "publication_date": "2024",
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "According to [CAC](https://cac.gov.ng/guidelines), single member companies are permitted."}}]
        }
        mock_post.return_value = mock_resp

        # Normal single-turn request without allow_unverified
        response = self.client.post(
            "/api/chat/",
            data={"message": "Can one person form a company under CAMA 2020?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Quarantined CAMA must not be used as internal verified context; fallback research must trigger
        self.assertTrue(mock_search.called, "Research fallback must trigger because CAMA is quarantined.")
        # Ensure payload sent to DeepSeek does not contain unverified CAMA chunk text
        call_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        system_content = call_payload["messages"][0]["content"]
        self.assertNotIn("UNVERIFIED PROVISION 1", system_content)
        # Ensure reply does not leak quarantine metadata to the user
        reply = response.data.get("reply", "")
        self.assertNotIn("official-agency-hosted reprint", reply)
        self.assertNotIn("pending verification", reply.lower())
        self.assertNotIn("not in my database", reply.lower())

    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_quarantined_cama_excluded_from_normal_sse_chat(self, mock_post, mock_search):
        """In normal SSE streaming chat, quarantined CAMA is excluded and research status is emitted."""
        mock_search.return_value = []
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "Under Nigerian law..."}}]}',
            'data: [DONE]'
        ]
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/?stream=true",
            data={"message": "Can one person form a company under CAMA 2020?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        content = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("Researching Nigerian legal sources...", content)
        self.assertTrue(mock_search.called)

    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_quarantined_cama_excluded_from_normal_multiturn_chat(self, mock_post, mock_search):
        """In multi-turn chat, quarantined CAMA is excluded from normal model context."""
        conv = Conversation.objects.create(user=self.user, title="CAMA Test")
        mock_search.return_value = []
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Under corporate law principles..."}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            f"/api/conversations/{conv.id}/messages/",
            data={"message": "Can a single person incorporate a company under CAMA 2020?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(mock_search.called, "Multi-turn chat must exclude quarantined CAMA and trigger research.")
        reply = response.data.get("reply", "") or response.data.get("text", "")
        self.assertNotIn("official-agency-hosted reprint", reply)
        self.assertNotIn("pending verification", reply.lower())
        self.assertNotIn("not in my database", reply.lower())

    # --------------------------------------------------------------------------
    # Regression Test: Cybercrimes Act 2024 Amendment Act Section 24
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_legal_web_search")
    @patch("chat.services.requests.post")
    def test_cybercrimes_act_section_24_reflects_2024_amendment_act(self, mock_post, mock_search):
        """Assistant accounts for the 2024 Amendment Act amending Section 24 of the Cybercrimes Act."""
        mock_search.return_value = [
            {
                "title": "Cybercrimes (Prohibition, Prevention, etc.) (Amendment) Act, 2024",
                "url": "https://placng.org/acts/cybercrimes-amendment-2024.pdf",
                "snippet": "Section 24 amended to delete vague expressions of annoyance and insult, narrowing offence to cyberstalking involving pornographic bullying and threats to life.",
                "tier_number": 5,
                "tier_label": "Reputable Legal Research Platform",
                "domain": "placng.org",
                "publication_date": "2024",
            }
        ]
        simulated_reply = (
            "Under Section 24 of the Cybercrimes (Prohibition, Prevention, etc.) Act 2015, as amended by the "
            "[Cybercrimes (Prohibition, Prevention, etc.) (Amendment) Act, 2024](https://placng.org/acts/cybercrimes-amendment-2024.pdf), "
            "the former vague provisions penalizing messages causing 'annoyance, insult, or inconvenience' were excised. "
            "Section 24 now specifically penalizes knowingly sending messages that are pornographic, threaten life, or cause a breakdown of public order."
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": simulated_reply}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "What is the penalty for cyberstalking under Section 24 of the Cybercrimes Act?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")
        self.assertIn("2024", reply)
        self.assertIn("Amendment", reply)
        self.assertIn("placng.org", reply)

    # --------------------------------------------------------------------------
    # Regression Test: Citation Integrity & No Fabricated Hyperlink Anchors
    # --------------------------------------------------------------------------
    def test_citation_integrity_no_fake_hyperlinks_and_genuine_links_preserved(self):
        """Citation tokens without genuine URLs are not rendered as fake anchors (#citation-...), but as plain labels."""
        from rag.services.citation_service import filter_and_validate_citations

        # Case 1: Token with genuine URL
        tok1 = "[CITATION: CIT-FED-CAMA2020-SEC18]"
        cit_with_url = {
            "citation_id": "CIT-FED-CAMA2020-SEC18",
            "citation_label": "Section 18, CAMA 2020",
            "source_url": "https://cac.gov.ng/cama2020.pdf",
        }
        res1, valid1, _ = filter_and_validate_citations(
            f"See {tok1} for details.",
            active_citation_tokens={"CIT-FED-CAMA2020-SEC18", tok1},
            citations=[cit_with_url],
            render_clickable=True
        )
        self.assertIn("[Section 18, CAMA 2020](https://cac.gov.ng/cama2020.pdf)", res1)

        # Case 2: Token WITHOUT genuine external URL must NOT create fake '#citation-...' anchor
        tok2 = "[CITATION: CIT-FED-EVIDENCE-SEC84]"
        cit_no_url = {
            "citation_id": "CIT-FED-EVIDENCE-SEC84",
            "citation_label": "Section 84, Evidence Act 2011",
            "source_url": "",  # No external URL
        }
        res2, valid2, _ = filter_and_validate_citations(
            f"According to {tok2}, computer records are admissible.",
            active_citation_tokens={"CIT-FED-EVIDENCE-SEC84", tok2},
            citations=[cit_no_url],
            render_clickable=True
        )
        self.assertNotIn("#citation-", res2, "Must NOT fabricate fake anchor links when source URL is empty.")
        self.assertIn("Section 84, Evidence Act 2011", res2)

    # --------------------------------------------------------------------------
    # Regression Test: Sanitizer Preserves Substantive Notes and Warnings
    # --------------------------------------------------------------------------
    def test_sanitizer_preserves_substantive_notes_and_warnings(self):
        """Sanitizer strips trailing generic boilerplate but strictly preserves substantive legal notes and warnings."""
        raw_output = (
            "Under the Public Officers Protection Act, legal action against a public officer must be instituted within three months.\n\n"
            "Warning: Instituting an action outside the 3-month limitation period deprives the court of jurisdiction.\n\n"
            "Note: The 2024 Amendment Act substantially revised penalties under Section 24.\n\n"
            "---\n*Disclaimer: This is for general educational and informational purposes only and does not constitute formal legal advice. Consult a qualified Nigerian legal practitioner.*"
        )
        cleaned = sanitize_chat_response(raw_output, is_legal_query=True)
        # Substantive warning and note must be intact
        self.assertIn("Warning: Instituting an action outside the 3-month limitation period", cleaned)
        self.assertIn("Note: The 2024 Amendment Act substantially revised penalties", cleaned)
        # Boilerplate disclaimer must be removed
        self.assertNotIn("This is for general educational and informational purposes only", cleaned)
        self.assertNotIn("Consult a qualified Nigerian legal practitioner", cleaned)

    # --------------------------------------------------------------------------
    # Scenario 11: Conversational Greetings and Natural Small Talk
    # --------------------------------------------------------------------------
    def test_scenario_11_system_prompt_enforces_natural_greetings_without_hardcoded_template(self):
        """System prompt instructs natural conversational responses to greetings and small talk without hardcoded templates."""
        from chat.services import SYSTEM_PROMPT

        # Must instruct natural conversational handling
        self.assertIn("Natural Conversation, Greetings, and Small Talk", SYSTEM_PROMPT)
        self.assertIn("Engage naturally, warmly, and contextually", SYSTEM_PROMPT)
        self.assertIn("Inquiries about yourself", SYSTEM_PROMPT)
        # Must explicitly forbid forcing product introductions or sales pitches
        self.assertIn("Do NOT repeatedly introduce yourself or your title", SYSTEM_PROMPT)
        self.assertIn("Do NOT turn greetings or small talk into a pitch or forced invitation", SYSTEM_PROMPT)
        # Must explicitly instruct multi-turn awareness
        self.assertIn("Multi-Turn Conversational Awareness", SYSTEM_PROMPT)
        # Must NOT contain the old hardcoded canned greeting template
        self.assertNotIn("Use this exact concise greeting:", SYSTEM_PROMPT)

    @patch("chat.services.requests.post")
    def test_scenario_11_how_are_you_returns_natural_conversation_without_disclaimers(self, mock_post):
        """When user asks 'Hi chat, how are you?', model responds conversationally without robotic introductions or disclaimers."""
        mock_ai_reply = "I'm doing great, thank you for asking! How is your day going so far?"
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": mock_ai_reply}}]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Hi chat, how are you?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        reply = response.data.get("reply", "")

        self.assertIn("I'm doing great", reply)
        # Must NOT have boilerplate disclaimers
        self.assertNotIn("Disclaimer", reply)
        self.assertNotIn("formal legal advice", reply)
        # Verify payload sent to provider has system prompt with conversational instructions
        sent_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        messages = sent_payload.get("messages", [])
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        self.assertIn("Natural Conversation, Greetings, and Small Talk", system_msg)

    @patch("chat.services.requests.post")
    def test_scenario_11_transition_from_small_talk_to_legal_query(self, mock_post):
        """Conversational flow transitions smoothly from small talk to legal advice with context preserved."""
        conv = Conversation.objects.create(user=self.user, title="Natural Chat")
        url = f"/api/conversations/{conv.id}/messages/"

        # Turn 1: Small talk
        mock_resp1 = MagicMock(spec=requests.Response)
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "I'm doing well, thank you! What's on your mind?"}}]
        }
        mock_post.return_value = mock_resp1

        res1 = self.client.post(url, data={"message": "Hello, how are you?"}, format="json")
        self.assertEqual(res1.status_code, status.HTTP_201_CREATED)

        # Turn 2: User transitions to a legal question
        mock_resp2 = MagicMock(spec=requests.Response)
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Under Nigerian law, a contract requires offer, acceptance, and consideration."}}]
        }
        mock_post.return_value = mock_resp2

        res2 = self.client.post(url, data={"message": "What is a contract under Nigerian law?"}, format="json")
        self.assertEqual(res2.status_code, status.HTTP_201_CREATED)

        # Verify history sent to Turn 2 includes Turn 1
        sent_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        history_roles = [m["role"] for m in sent_payload.get("messages", [])]
        self.assertIn("assistant", history_roles)
        self.assertEqual(conv.messages.count(), 4)

    # --------------------------------------------------------------------------
    # Scenario 12: Real-Time Current Information & Grounded Web Search Verification
    # --------------------------------------------------------------------------
    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_president_of_usa_triggers_search_and_grounds_reply(self, mock_post, mock_search):
        """'Who's the president of USA?' triggers live search and grounds reply in current search results."""
        mock_search.return_value = [
            {
                "title": "Presidents of the United States | The White House",
                "url": "https://www.whitehouse.gov/about-the-white-house/presidents/",
                "snippet": "Donald J. Trump is the 47th President of the United States, inaugurated on January 20, 2025."
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The current President of the United States is Donald J. Trump, who took office as the 47th president on January 20, 2025. See [The White House](https://www.whitehouse.gov/about-the-white-house/presidents/) for official records."
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Who's the president of USA?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        # Search MUST be called
        mock_search.assert_called_once_with("Who's the president of USA?")

        # Prompt sent to model MUST contain the live search results
        sent_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        user_msg = next((m["content"] for m in sent_payload.get("messages", []) if m["role"] == "user"), "")
        self.assertIn("REAL-TIME WEB SEARCH RESULTS", user_msg)
        self.assertIn("Donald J. Trump", user_msg)
        self.assertIn("https://www.whitehouse.gov", user_msg)

        reply = response.data.get("reply", "")
        self.assertIn("Donald J. Trump", reply)
        self.assertNotIn("Joe Biden", reply)

    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_president_of_nigeria_triggers_search(self, mock_post, mock_search):
        """'Who is the president of Nigeria?' triggers live web search."""
        mock_search.return_value = [
            {
                "title": "State House, Abuja | The Official Website of the State House",
                "url": "https://statehouse.gov.ng",
                "snippet": "Asiwaju Bola Ahmed Tinubu, GCFR, is the President of the Federal Republic of Nigeria."
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The current President of Nigeria is [Bola Ahmed Tinubu](https://statehouse.gov.ng), who assumed office on May 29, 2023."
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Who is the president of Nigeria?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_search.assert_called_once_with("Who is the president of Nigeria?")
        self.assertIn("Bola Ahmed Tinubu", response.data.get("reply", ""))

    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_historical_president_query_skips_search(self, mock_post, mock_search):
        """'Who was the president of the USA in 2020?' is historical and does NOT trigger web search."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "In 2020, Donald Trump was the President of the United States, serving his first term from 2017 to 2021."
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Who was the president of the USA in 2020?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_search.assert_not_called()
        self.assertIn("Donald Trump", response.data.get("reply", ""))

    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_python_version_triggers_search(self, mock_post, mock_search):
        """'What is the latest version of Python?' triggers live search."""
        mock_search.return_value = [
            {
                "title": "Python Releases for Windows | Python.org",
                "url": "https://www.python.org/downloads/windows/",
                "snippet": "Latest Python 3 Release - Python 3.13.0."
            }
        ]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The latest stable version of Python is Python 3.13. See [Python.org](https://www.python.org/downloads/windows/)."
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "What is the latest version of Python?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_search.assert_called_once_with("What is the latest version of Python?")
        self.assertIn("Python 3.13", response.data.get("reply", ""))

    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_greeting_skips_search(self, mock_post, mock_search):
        """'Hi, how are you?' skips web search and responds naturally."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you! How are you doing today?"
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Hi, how are you?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_search.assert_not_called()
        self.assertIn("doing well", response.data.get("reply", ""))

    @patch("chat.views.perform_web_search")
    @patch("chat.services.requests.post")
    def test_scenario_12_search_failure_injects_transparent_notice(self, mock_post, mock_search):
        """When web search fails to return results, prompt transparently notifies model rather than guessing."""
        mock_search.return_value = []  # Search returned empty or failed
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Live search results could not be retrieved at this moment to verify the current information."
                }
            }]
        }
        mock_post.return_value = mock_resp

        response = self.client.post(
            "/api/chat/",
            data={"message": "Who's the president of USA?"},
            format="json"
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        sent_payload = mock_post.call_args[1].get("json") or json.loads(mock_post.call_args[1].get("data", "{}"))
        user_msg = next((m["content"] for m in sent_payload.get("messages", []) if m["role"] == "user"), "")
        self.assertIn("REAL-TIME SEARCH NOTICE", user_msg)
        self.assertIn("Do NOT guess or state outdated pre-cutoff information", user_msg)
