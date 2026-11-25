from rest_framework import serializers
from .models import Conversation, Message

class ChatRequestSerializer(serializers.Serializer):
    """
    Validates incoming chat request payload for the standalone /api/chat/ endpoint.
    """
    message = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        error_messages={
            'required': 'Message field is required.',
            'blank': 'Message cannot be empty or contain only whitespace.'
        }
    )
    allow_unverified = serializers.BooleanField(required=False, default=False)
    mode = serializers.CharField(required=False, allow_blank=True, default="verified")


class MessageSerializer(serializers.ModelSerializer):
    """
    Serializes a message exchanged in a conversation.
    """
    timestamp = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = ['id', 'conversation_id', 'sender', 'text', 'created_at', 'timestamp']
        read_only_fields = ['id', 'conversation_id', 'created_at', 'timestamp']

    def get_timestamp(self, obj):
        if obj.created_at:
            return obj.created_at.strftime("%I:%M %p")
        return ""


class ConversationListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for listing conversations in the sidebar.
    """
    message_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'pinned', 'created_at', 'updated_at', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at', 'message_count']


class ConversationDetailSerializer(serializers.ModelSerializer):
    """
    Detailed serializer for a conversation including all its messages.
    """
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'pinned', 'created_at', 'updated_at', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at', 'messages']


class ConversationCreateUpdateSerializer(serializers.ModelSerializer):
    """
    Validates creation, renaming, and pinning of conversations.
    """
    title = serializers.CharField(
        required=False,
        allow_blank=False,
        max_length=255,
        trim_whitespace=True,
        error_messages={
            'blank': 'Title cannot be empty.'
        }
    )
    pinned = serializers.BooleanField(required=False)

    class Meta:
        model = Conversation
        fields = ['title', 'pinned']


class SendMessageSerializer(serializers.Serializer):
    """
    Validates sending a message within an existing conversation.
    Supports optional document attachment (PDF, DOCX, TXT).
    """
    message = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        error_messages={
            'required': 'Message field is required.',
            'blank': 'Message cannot be empty or contain only whitespace.'
        }
    )
    file = serializers.FileField(required=False, allow_null=True)
    web_search = serializers.BooleanField(required=False, default=None, allow_null=True)
    allow_unverified = serializers.BooleanField(required=False, default=False)
    mode = serializers.CharField(required=False, allow_blank=True, default="verified")
