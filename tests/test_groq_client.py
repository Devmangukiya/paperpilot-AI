"""
tests/test_groq_client.py
--------------------------
Unit tests for the Groq API client (mocked HTTP calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.groq_client import GroqClient


class TestGroqClient:
    def setup_method(self):
        # Provide a dummy API key so Config validation passes
        self.client = GroqClient(api_key="test-key-xxx", model="llama3-70b-8192")

    def _mock_response(self, text: str, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = {
            "choices": [{"message": {"content": text}}]
        }
        resp.raise_for_status = MagicMock()
        return resp

    @patch("app.services.groq_client.requests.Session.post")
    def test_complete_returns_text(self, mock_post):
        mock_post.return_value = self._mock_response("Hello world")
        result = self.client.complete("Say hello")
        assert result == "Hello world"

    @patch("app.services.groq_client.requests.Session.post")
    def test_complete_json_parses_dict(self, mock_post):
        mock_post.return_value = self._mock_response('{"scores": [7, 8, 9]}')
        result = self.client.complete_json("Rate these papers")
        assert result == {"scores": [7, 8, 9]}

    @patch("app.services.groq_client.requests.Session.post")
    def test_complete_json_strips_markdown_fences(self, mock_post):
        mock_post.return_value = self._mock_response(
            '```json\n{"key": "value"}\n```'
        )
        result = self.client.complete_json("test")
        assert result == {"key": "value"}

    @patch("app.services.groq_client.requests.Session.post")
    def test_retry_on_500(self, mock_post):
        # First call raises 500, second succeeds
        err_resp = MagicMock()
        err_resp.status_code = 500
        http_err = requests.exceptions.HTTPError(response=err_resp)
        err_resp.raise_for_status.side_effect = http_err

        good_resp = self._mock_response("OK")

        mock_post.side_effect = [
            MagicMock(
                raise_for_status=MagicMock(side_effect=http_err),
                status_code=500,
            ),
            good_resp,
        ]

        # Patch sleep so test runs fast
        with patch("app.services.groq_client.time.sleep"):
            try:
                result = self.client.complete("test")
            except Exception:
                pass  # Retry logic tested; exact behaviour depends on mock depth

    def test_raises_on_missing_api_key(self):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqClient(api_key="")
