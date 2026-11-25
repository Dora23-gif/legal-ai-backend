from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """
    Serializes basic user profile information.
    """
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class RegisterSerializer(serializers.Serializer):
    """
    Validates and registers a new user with secure password hashing.
    """
    username = serializers.CharField(
        required=True,
        min_length=3,
        max_length=150,
        trim_whitespace=True,
        error_messages={
            'required': 'Username is required.',
            'blank': 'Username cannot be empty.',
            'min_length': 'Username must be at least 3 characters long.'
        }
    )
    email = serializers.EmailField(
        required=True,
        error_messages={
            'required': 'Email is required.',
            'blank': 'Email cannot be empty.',
            'invalid': 'Please provide a valid email address.'
        }
    )
    password = serializers.CharField(
        required=True,
        write_only=True,
        min_length=8,
        error_messages={
            'required': 'Password is required.',
            'blank': 'Password cannot be empty.',
            'min_length': 'Password must be at least 8 characters long.'
        }
    )

    def validate_username(self, value):
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value

    def validate_email(self, value):
        clean_email = value.strip().lower()
        if User.objects.filter(email__iexact=clean_email).exists():
            raise serializers.ValidationError("An account with this email address already exists.")
        return clean_email

    def validate_password(self, value):
        try:
            validate_password(value)
        except DjangoValidationError as e:
            raise serializers.ValidationError(list(e.messages))
        return value

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password']
        )


class LoginSerializer(serializers.Serializer):
    """
    Validates user credentials (supports username or email).
    """
    username = serializers.CharField(
        required=True,
        trim_whitespace=True,
        error_messages={'required': 'Username or email is required.'}
    )
    password = serializers.CharField(
        required=True,
        write_only=True,
        error_messages={'required': 'Password is required.'}
    )

    def validate(self, attrs):
        username_or_email = attrs.get('username')
        password = attrs.get('password')

        # Check if login identifier is an email
        user_obj = None
        if '@' in username_or_email:
            user_obj = User.objects.filter(email__iexact=username_or_email).first()
            if user_obj:
                username_to_auth = user_obj.username
            else:
                username_to_auth = username_or_email
        else:
            username_to_auth = username_or_email

        user = authenticate(username=username_to_auth, password=password)

        if not user:
            raise serializers.ValidationError("Invalid username or password. Please verify your credentials.")

        if not user.is_active:
            raise serializers.ValidationError("This account has been deactivated. Please contact support.")

        attrs['user'] = user
        return attrs
