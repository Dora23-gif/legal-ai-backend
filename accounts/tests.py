from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework.authtoken.models import Token

User = get_user_model()


class AuthenticationAPITests(APITestCase):
    def setUp(self):
        self.register_url = reverse('auth-register')
        self.login_url = reverse('auth-login')
        self.logout_url = reverse('auth-logout')
        self.me_url = reverse('auth-me')

        self.user_data = {
            "username": "adeola_law",
            "email": "adeola@nigerianlaw.ng",
            "password": "StrongPassword123!"
        }

    def test_register_new_user_success(self):
        """POST /api/auth/register/ should create user and return auth token."""
        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("token", response.data)
        self.assertIn("user", response.data)
        self.assertEqual(response.data["user"]["username"], "adeola_law")
        self.assertEqual(response.data["user"]["email"], "adeola@nigerianlaw.ng")

        # Verify password is hashed, not plain text
        user = User.objects.get(username="adeola_law")
        self.assertNotEqual(user.password, "StrongPassword123!")
        self.assertTrue(user.check_password("StrongPassword123!"))

    def test_register_duplicate_username_rejected(self):
        """POST /api/auth/register/ should reject existing username with 400."""
        User.objects.create_user(username="adeola_law", email="other@law.ng", password="Password123!")
        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_register_duplicate_email_rejected(self):
        """POST /api/auth/register/ should reject existing email with 400."""
        User.objects.create_user(username="different_user", email="adeola@nigerianlaw.ng", password="Password123!")
        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_register_short_password_rejected(self):
        """POST /api/auth/register/ should reject password shorter than 8 characters."""
        data = {
            "username": "new_barrister",
            "email": "barrister@law.ng",
            "password": "123"
        }
        response = self.client.post(self.register_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_login_with_username_success(self):
        """POST /api/auth/login/ with valid username returns auth token."""
        User.objects.create_user(username="adeola_law", email="adeola@nigerianlaw.ng", password="StrongPassword123!")
        login_data = {"username": "adeola_law", "password": "StrongPassword123!"}
        response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("token", response.data)
        self.assertEqual(response.data["user"]["username"], "adeola_law")

    def test_login_with_email_success(self):
        """POST /api/auth/login/ with valid email returns auth token."""
        User.objects.create_user(username="adeola_law", email="adeola@nigerianlaw.ng", password="StrongPassword123!")
        login_data = {"username": "adeola@nigerianlaw.ng", "password": "StrongPassword123!"}
        response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("token", response.data)

    def test_login_with_invalid_credentials_rejected(self):
        """POST /api/auth/login/ with wrong password returns 400."""
        User.objects.create_user(username="adeola_law", email="adeola@nigerianlaw.ng", password="StrongPassword123!")
        login_data = {"username": "adeola_law", "password": "WrongPassword!"}
        response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_logout_revokes_token(self):
        """POST /api/auth/logout/ deletes token and revokes access."""
        user = User.objects.create_user(username="adeola_law", email="adeola@nigerianlaw.ng", password="StrongPassword123!")
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

        # Verify access before logout
        me_res = self.client.get(self.me_url)
        self.assertEqual(me_res.status_code, status.HTTP_200_OK)

        # Logout
        logout_res = self.client.post(self.logout_url)
        self.assertEqual(logout_res.status_code, status.HTTP_200_OK)

        # Subsequent request with old token should fail
        me_after = self.client.get(self.me_url)
        self.assertEqual(me_after.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_current_user_unauthenticated_rejected(self):
        """GET /api/auth/me/ without token returns 401."""
        response = self.client.get(self.me_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
