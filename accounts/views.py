from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.authtoken.models import Token
from .serializers import RegisterSerializer, LoginSerializer, UserSerializer


class RegisterAPIView(APIView):
    """
    POST /api/auth/register/
    Registers a new user, hashes password, and returns auth token.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            # Extract first user-friendly error message
            first_key = next(iter(errors))
            raw_err = errors[first_key]
            first_err = raw_err[0] if isinstance(raw_err, list) else raw_err
            return Response(
                {"error": str(first_err), "details": errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)

        return Response(
            {
                "message": "Registration successful.",
                "token": token.key,
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_201_CREATED
        )


class LoginAPIView(APIView):
    """
    POST /api/auth/login/
    Authenticates user and returns auth token.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            errors = serializer.errors
            first_key = next(iter(errors))
            raw_err = errors[first_key]
            first_err = raw_err[0] if isinstance(raw_err, list) else raw_err
            return Response(
                {"error": str(first_err), "details": errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = serializer.validated_data['user']
        token, _ = Token.objects.get_or_create(user=user)

        return Response(
            {
                "message": "Login successful.",
                "token": token.key,
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_200_OK
        )


class LogoutAPIView(APIView):
    """
    POST /api/auth/logout/
    Revokes the current user's auth token.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            # Delete token to invalidate server session
            request.user.auth_token.delete()
        except Exception:
            pass

        return Response(
            {"detail": "Successfully logged out."},
            status=status.HTTP_200_OK
        )


class CurrentUserAPIView(APIView):
    """
    GET /api/auth/me/
    Returns the currently authenticated user's profile information.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        serializer = UserSerializer(request.user)
        return Response({"user": serializer.data}, status=status.HTTP_200_OK)
