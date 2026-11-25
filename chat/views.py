import json
import logging
from django.shortcuts import get_object_or_404
from django.db.models import Count
from django.db import transaction, connection, OperationalError
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer, BaseRenderer
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser

from .models import Conversation, Message
from .serializers import (
    ChatRequestSerializer,
    ConversationListSerializer,
    ConversationDetailSerializer,
    ConversationCreateUpdateSerializer,
    SendMessageSerializer,
    MessageSerializer,
)
from .services import (
    generate_legal_response,
    stream_legal_response_events,
    AIServiceError,
    OllamaServiceError,
)
from .document_utils import (
    validate_document_file,
    extract_text_from_file,
    build_document_analysis_prompt,
    DocumentValidationError,
    DocumentExtractionError,
)
from .web_search import (
    should_search_web,
    perform_web_search,
    format_web_search_prompt,
    perform_legal_web_search,
    format_legal_research_prompt,
)
from rag.services.retrieval_service import retrieve_legal_context

logger = logging.getLogger(__name__)


class EventStreamRenderer(BaseRenderer):
    """Custom DRF renderer allowing Server-Sent Events content negotiation."""
    media_type = 'text/event-stream'
    format = 'event-stream'
    charset = 'utf-8'

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data


from django.core.serializers.json import DjangoJSONEncoder


def sse_event(event_name: str, data: dict) -> str:
    """Formats payload into standard Server-Sent Event block."""
    return f"event: {event_name}\ndata: {json.dumps(data, cls=DjangoJSONEncoder)}\n\n"


def get_user_conversations_queryset(request):
    """
    Enforces conversation ownership.
    Strictly returns conversations owned by the authenticated user.
    Unauthenticated callers receive an empty queryset.
    """
    if request.user and request.user.is_authenticated:
        return Conversation.objects.filter(user=request.user)
    return Conversation.objects.none()


class ChatAPIView(APIView):
    """
    Standalone endpoint (Phase 2 backwards compatibility):
    POST /api/chat/
    Accepts: { "message": "..." }
    Supports SSE streaming when requested via Accept: text/event-stream or ?stream=true.
    """
    permission_classes = [permissions.AllowAny]
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer, EventStreamRenderer]

    def post(self, request, *args, **kwargs):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            error_msg = errors.get('message', ['Invalid request payload'])[0]
            return Response(
                {"error": str(error_msg), "details": errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        message = serializer.validated_data['message']
        is_streaming = (
            request.headers.get('Accept') == 'text/event-stream' or
            request.query_params.get('stream', '').lower() == 'true'
        )

        # CAMA Quarantine Safeguard: Ordinary legal retrieval strictly receives verified documents only.
        # Unverified-source retrieval (e.g. quarantined CAMA 2020 reprint) is isolated behind an explicit parameter.
        allow_unverified = bool(
            serializer.validated_data.get('allow_unverified') is True or
            serializer.validated_data.get('mode') == 'unverified' or
            request.data.get('allow_unverified') is True or
            request.data.get('mode') == 'unverified' or
            request.query_params.get('unverified', '').lower() in ('true', '1') or
            request.query_params.get('mode', '').lower() == 'unverified'
        )

        # RAG Legal Context Retrieval & Intelligent Research Fallback
        legal_ctx = None
        prompt_to_run = message
        needs_web_search = should_search_web(message)
        try:
            legal_ctx = retrieve_legal_context(message, allow_unverified=allow_unverified)
            if legal_ctx.get('is_legal_query'):
                if not legal_ctx.get('has_verified_results') and not legal_ctx.get('has_unverified_results'):
                    # Fallback to intelligent Nigerian legal web research
                    research_results = perform_legal_web_search(message)
                    legal_ctx['prompt_context'] = format_legal_research_prompt(message, research_results)
                    legal_ctx['is_external_research'] = True
            elif needs_web_search:
                try:
                    search_results = perform_web_search(message)
                except Exception as search_err:
                    logger.warning("Web search error in ChatAPIView: %s", search_err)
                    search_results = []
                prompt_to_run = format_web_search_prompt(message, search_results)
        except Exception as e:
            logger.warning("RAG retrieval exception in ChatAPIView: %s", e)

        if is_streaming:
            def chat_stream():
                accumulated = ""
                mode = "legal"
                citations = []
                if legal_ctx and legal_ctx.get('is_external_research'):
                    yield sse_event("status", {"status": "Researching Nigerian legal sources..."})
                elif needs_web_search:
                    yield sse_event("status", {"status": "Searching the web..."})

                try:
                    for event_type, event_data in stream_legal_response_events(prompt_to_run, legal_context=legal_ctx):
                        if event_type == "status":
                            yield sse_event("status", event_data)
                        elif event_type == "result":
                            accumulated = event_data.get("reply", "")
                            mode = event_data.get("mode", "legal")
                            citations = event_data.get("citations", [])

                    yield sse_event("complete", {"reply": accumulated, "mode": mode, "citations": citations})
                except (AIServiceError, OllamaServiceError) as err:
                    yield sse_event("error", {
                        "error": str(err),
                        "details": str(err)
                    })
                except Exception as err:
                    logger.exception("Unexpected error in standalone chat stream:")
                    yield sse_event("error", {
                        "error": "An error occurred while generating the AI response.",
                        "details": str(err)
                    })

            response = StreamingHttpResponse(chat_stream(), content_type='text/event-stream; charset=utf-8')
            response['Cache-Control'] = 'no-cache, no-transform'
            response['X-Accel-Buffering'] = 'no'
            return response

        try:
            response_data = generate_legal_response(prompt_to_run, legal_context=legal_ctx)
            return Response(response_data, status=status.HTTP_200_OK)
        except (AIServiceError, OllamaServiceError) as err:
            return Response(
                {
                    "error": "The AI assistant service is currently unavailable.",
                    "details": str(err)
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception as err:
            return Response(
                {
                    "error": "An error occurred while generating the AI response.",
                    "details": str(err)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ConversationListCreateAPIView(APIView):
    """
    GET  /api/conversations/ - List all conversations belonging to the authenticated user
    POST /api/conversations/ - Create a new conversation session owned by the authenticated user
    DELETE /api/conversations/ - Clear all conversations belonging to the authenticated user
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        queryset = get_user_conversations_queryset(request).annotate(
            message_count=Count('messages')
        ).order_by('-pinned', '-updated_at')
        serializer = ConversationListSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = ConversationCreateUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            error_msg = next(iter(errors.values()))[0] if errors else "Invalid request data"
            return Response({"error": str(error_msg), "details": errors}, status=status.HTTP_400_BAD_REQUEST)

        title = serializer.validated_data.get('title', 'New Chat')
        pinned = serializer.validated_data.get('pinned', False)

        conversation = Conversation.objects.create(
            user=request.user,
            title=title,
            pinned=pinned
        )
        return Response(
            ConversationDetailSerializer(conversation).data,
            status=status.HTTP_201_CREATED
        )

    def delete(self, request, *args, **kwargs):
        queryset = get_user_conversations_queryset(request)
        count = queryset.count()
        queryset.delete()
        return Response(
            {"detail": f"Successfully cleared {count} conversations."},
            status=status.HTTP_204_NO_CONTENT
        )


class ConversationDetailAPIView(APIView):
    """
    GET    /api/conversations/<uuid:pk>/ - Retrieve conversation with full message history (owner only)
    PATCH  /api/conversations/<uuid:pk>/ - Rename or update pinned status (owner only)
    DELETE /api/conversations/<uuid:pk>/ - Delete conversation and its messages (owner only)
    """
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self, request, pk):
        queryset = get_user_conversations_queryset(request).prefetch_related('messages')
        return get_object_or_404(queryset, pk=pk)

    def get(self, request, pk, *args, **kwargs):
        conversation = self.get_object(request, pk)
        serializer = ConversationDetailSerializer(conversation)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, pk, *args, **kwargs):
        conversation = self.get_object(request, pk)
        serializer = ConversationCreateUpdateSerializer(conversation, data=request.data, partial=True)
        if not serializer.is_valid():
            errors = serializer.errors
            error_msg = next(iter(errors.values()))[0] if errors else "Invalid request data"
            return Response({"error": str(error_msg), "details": errors}, status=status.HTTP_400_BAD_REQUEST)

        serializer.save()
        return Response(
            ConversationDetailSerializer(conversation).data,
            status=status.HTTP_200_OK
        )

    def delete(self, request, pk, *args, **kwargs):
        conversation = self.get_object(request, pk)
        conversation.delete()
        return Response(
            {"detail": "Conversation deleted successfully."},
            status=status.HTTP_204_NO_CONTENT
        )


class ConversationMessagesAPIView(APIView):
    """
    POST /api/conversations/<uuid:pk>/messages/
    Appends user message (with optional document attachment), computes assistant legal response,
    and saves both atomically to owner's conversation.
    Supports SSE streaming when requested via Accept: text/event-stream or ?stream=true.
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer, EventStreamRenderer]

    def post(self, request, pk, *args, **kwargs):
        try:
            queryset = get_user_conversations_queryset(request)
            conversation = get_object_or_404(queryset, pk=pk)
        except OperationalError:
            connection.close()
            queryset = get_user_conversations_queryset(request)
            conversation = get_object_or_404(queryset, pk=pk)

        serializer = SendMessageSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            first_err = next(iter(errors.values())) if errors else None
            error_msg = first_err[0] if (first_err and isinstance(first_err, list)) else "Invalid request payload"
            return Response(
                {"error": str(error_msg), "details": errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        user_text = serializer.validated_data['message']
        uploaded_file = serializer.validated_data.get('file') or request.FILES.get('file')
        web_search_flag = serializer.validated_data.get('web_search')

        ai_prompt_text = user_text
        db_user_text = user_text

        # ----------------------------------------------------------------------
        # Document Attachment Processing (In-Memory Text Extraction & Guardrails)
        # ----------------------------------------------------------------------
        if uploaded_file:
            try:
                extracted_text = extract_text_from_file(uploaded_file)
                ai_prompt_text = build_document_analysis_prompt(
                    user_question=user_text,
                    filename=uploaded_file.name,
                    extracted_text=extracted_text
                )
                db_user_text = f"📎 [Attached: {uploaded_file.name}]\n\n{user_text}"
            except (DocumentValidationError, DocumentExtractionError) as doc_err:
                return Response(
                    {"error": str(doc_err)},
                    status=status.HTTP_400_BAD_REQUEST
                )

        needs_web_search = should_search_web(
            user_text,
            has_document=bool(uploaded_file),
            web_enabled=web_search_flag
        )

        is_streaming = (
            request.headers.get('Accept') == 'text/event-stream' or
            request.query_params.get('stream', '').lower() == 'true'
        )

        # ----------------------------------------------------------------------
        # Multi-Turn Context Window (Last 8 messages preserved in chronological order)
        # ----------------------------------------------------------------------
        try:
            past_messages = list(conversation.messages.order_by('created_at'))[-8:]
        except OperationalError:
            connection.close()
            past_messages = list(conversation.messages.order_by('created_at'))[-8:]

        history = [
            {
                "role": "user" if m.sender == "user" else "assistant",
                "content": m.text
            }
            for m in past_messages
            if m.text and m.text.strip()
        ]

        # CAMA Quarantine Safeguard: Ordinary legal retrieval strictly receives verified documents only.
        allow_unverified = bool(
            serializer.validated_data.get('allow_unverified') is True or
            serializer.validated_data.get('mode') == 'unverified' or
            request.data.get('allow_unverified') is True or
            request.data.get('mode') == 'unverified' or
            request.query_params.get('unverified', '').lower() in ('true', '1') or
            request.query_params.get('mode', '').lower() == 'unverified'
        )

        # ----------------------------------------------------------------------
        # RAG Statutory Legal Context Retrieval & Intelligent Research Fallback
        # ----------------------------------------------------------------------
        legal_ctx = None
        if not uploaded_file:  # If document uploaded, document text takes prompt priority
            try:
                legal_ctx = retrieve_legal_context(user_text, allow_unverified=allow_unverified)
                if legal_ctx.get('is_legal_query'):
                    if not legal_ctx.get('has_verified_results') and not legal_ctx.get('has_unverified_results'):
                        # Trigger intelligent Nigerian legal web research fallback
                        research_results = perform_legal_web_search(user_text)
                        legal_ctx['prompt_context'] = format_legal_research_prompt(user_text, research_results)
                        legal_ctx['is_external_research'] = True
            except Exception as e:
                logger.warning("RAG retrieval exception in ConversationMessagesAPIView: %s", e)

        # ----------------------------------------------------------------------
        # SSE Streaming Execution Flow
        # ----------------------------------------------------------------------
        if is_streaming:
            def event_stream():
                accumulated_reply = ""
                mode = "legal"
                citations = []
                prompt_to_stream = ai_prompt_text

                # Honest real-time status update for research / web search
                if legal_ctx and legal_ctx.get('is_external_research'):
                    yield sse_event("status", {"status": "Researching Nigerian legal sources..."})
                elif needs_web_search:
                    yield sse_event("status", {"status": "Searching the web..."})
                    try:
                        search_results = perform_web_search(user_text)
                    except Exception as search_err:
                        logger.warning("Web search error: %s", search_err)
                        search_results = []
                    prompt_to_stream = format_web_search_prompt(user_text, search_results)

                try:
                    for event_type, event_data in stream_legal_response_events(
                        prompt_to_stream,
                        history=history,
                        legal_context=legal_ctx
                    ):
                        if event_type == "status":
                            yield sse_event("status", event_data)
                        elif event_type == "result":
                            accumulated_reply = event_data.get("reply", "")
                            mode = event_data.get("mode", "legal")
                            citations = event_data.get("citations", [])
                except (AIServiceError, OllamaServiceError) as err:
                    yield sse_event("error", {
                        "error": str(err),
                        "details": str(err)
                    })
                    return
                except Exception as err:
                    logger.exception("Unexpected error during legal AI streaming:")
                    yield sse_event("error", {
                        "error": "An error occurred while generating the AI response.",
                        "details": str(err)
                    })
                    return

                # Atomic persistence: Save only once when generation succeeds completely
                try:
                    with transaction.atomic():
                        user_message = Message.objects.create(
                            conversation=conversation,
                            sender='user',
                            text=db_user_text
                        )
                        assistant_message = Message.objects.create(
                            conversation=conversation,
                            sender='assistant',
                            text=accumulated_reply
                        )

                        if conversation.title == "New Chat":
                            clean_title = user_text.strip().replace('\n', ' ')
                            if len(clean_title) > 36:
                                clean_title = clean_title[:33] + "..."
                            conversation.title = clean_title

                        conversation.save()

                    yield sse_event("complete", {
                        "reply": accumulated_reply,
                        "mode": mode,
                        "citations": citations,
                        "conversation_id": str(conversation.id),
                        "conversation_title": conversation.title,
                        "user_message": MessageSerializer(user_message).data,
                        "assistant_message": MessageSerializer(assistant_message).data,
                    })
                except Exception as err:
                    logger.exception("Database error persisting streamed messages:")
                    yield sse_event("error", {
                        "error": "Failed to save message to database.",
                        "details": str(err)
                    })

            response = StreamingHttpResponse(
                event_stream(),
                content_type='text/event-stream; charset=utf-8'
            )
            response['Cache-Control'] = 'no-cache, no-transform'
            response['X-Accel-Buffering'] = 'no'
            return response

        # ----------------------------------------------------------------------
        # Standard Synchronous Flow (Backwards Compatibility)
        # ----------------------------------------------------------------------
        prompt_to_run = ai_prompt_text
        if not (legal_ctx and legal_ctx.get('is_external_research')) and needs_web_search:
            try:
                search_results = perform_web_search(user_text)
            except Exception as search_err:
                logger.warning("Web search error: %s", search_err)
                search_results = []
            prompt_to_run = format_web_search_prompt(user_text, search_results)

        try:
            legal_response = generate_legal_response(
                prompt_to_run,
                history=history,
                legal_context=legal_ctx
            )
        except (AIServiceError, OllamaServiceError) as err:
            return Response(
                {
                    "error": str(err),
                    "details": str(err)
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        except Exception as err:
            return Response(
                {
                    "error": "An error occurred while generating the AI response.",
                    "details": str(err)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        assistant_text = legal_response.get('reply', '')
        mode = legal_response.get('mode', 'legal')
        citations = legal_response.get('citations', [])

        with transaction.atomic():
            user_message = Message.objects.create(
                conversation=conversation,
                sender='user',
                text=db_user_text
            )
            assistant_message = Message.objects.create(
                conversation=conversation,
                sender='assistant',
                text=assistant_text
            )

            if conversation.title == "New Chat":
                clean_title = user_text.strip().replace('\n', ' ')
                if len(clean_title) > 36:
                    clean_title = clean_title[:33] + "..."
                conversation.title = clean_title
            conversation.save()

        return Response(
            {
                "reply": assistant_text,
                "mode": mode,
                "citations": citations,
                "conversation_id": str(conversation.id),
                "conversation_title": conversation.title,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(assistant_message).data,
            },
            status=status.HTTP_201_CREATED
        )
