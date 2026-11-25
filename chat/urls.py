from django.urls import path
from .views import (
    ChatAPIView,
    ConversationListCreateAPIView,
    ConversationDetailAPIView,
    ConversationMessagesAPIView,
)

urlpatterns = [
    # Standalone chat endpoint (Phase 2 compatibility)
    path('chat/', ChatAPIView.as_view(), name='chat-endpoint'),

    # Conversation endpoints (Phase 3 persistence)
    path('conversations/', ConversationListCreateAPIView.as_view(), name='conversation-list-create'),
    path('conversations/<uuid:pk>/', ConversationDetailAPIView.as_view(), name='conversation-detail'),
    path('conversations/<uuid:pk>/messages/', ConversationMessagesAPIView.as_view(), name='conversation-messages'),
]
      