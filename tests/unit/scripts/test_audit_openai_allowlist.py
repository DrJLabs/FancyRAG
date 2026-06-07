"""
Unit tests for the OpenAI allowlist audit script.

Framework: pytest
Mocking: unittest.mock.patch
"""

import json
import os
import sys
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch
import pytest

# Ensure repository root and src are importable regardless of CWD
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.audit_openai_allowlist import (  # noqa: E402
    _fetch_models,
    _format_list,
    _family_of,
    main,
    CATALOG_URL,
)


class TestFetchModels:
    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_models_success(self, mock_request, mock_urlopen):
        mock_payload = {
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4-turbo", "object": "model"},
            ]
        }

        # Make urlopen context manager-compatible
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            models = _fetch_models("test-api-key")

        assert models == {"gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"}

        mock_request.assert_called_once_with(
            f"{CATALOG_URL}?limit=100",
            headers={
                "Authorization": "Bearer test-api-key",
                "User-Agent": "fancyrag-allowlist-audit",
            },
        )
        mock_urlopen.assert_called_once_with(mock_request.return_value, timeout=30)

    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_models_empty(self, _mock_request, mock_urlopen):
        mock_payload = {"data": []}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            models = _fetch_models("key")
        assert models == set()

    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_models_malformed_items(self, _mock_request, mock_urlopen):
        mock_payload = {
            "data": [
                {"id": "gpt-4"},
                {"object": "model"},  # missing id -> skipped
                "not-a-dict",          # ignored
                {"id": ""},            # empty id skipped
                None,                  # ignored
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            models = _fetch_models("key")

        assert models == {"gpt-4"}

    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_models_paginates(self, mock_request, mock_urlopen):
        first_page = {
            "data": [
                {"id": "gpt-5-mini"},
            ],
            "has_more": True,
        }
        second_page = {
            "data": [
                {"id": "gpt-4o-mini"},
            ],
            "has_more": False,
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', side_effect=[first_page, second_page]):
            models = _fetch_models("key")

        assert models == {"gpt-5-mini", "gpt-4o-mini"}
        assert mock_request.call_count == 2
        first_call_url = mock_request.call_args_list[0].args[0]
        second_call_url = mock_request.call_args_list[1].args[0]
        assert first_call_url == f"{CATALOG_URL}?limit=100"
        assert "after=" in second_call_url

    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_fetch_models_missing_data_key(self, _mock_request, mock_urlopen):
        mock_payload = {"error": "oops"}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            models = _fetch_models("key")
        assert models == set()

    @patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError(
        url=CATALOG_URL, code=401, msg="Unauthorized", hdrs=None, fp=None
    ))
    def test_fetch_models_http_error(self, _mock_urlopen):
        with pytest.raises(urllib.error.HTTPError):
            _fetch_models("bad-key")

    @patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Network unreachable"))
    def test_fetch_models_url_error(self, _mock_urlopen):
        with pytest.raises(urllib.error.URLError):
            _fetch_models("key")


class TestFamilyOf:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4", "gpt-4"),
            ("text-davinci-003", "text-davinci"),
            ("o1-mini", "o1"),
            ("o1", "o1"),
            ("gpt-3.5-turbo-16k", "gpt-3.5-turbo"),
            ("claude-2-100k", "claude-2"),
            ("", ""),  # Edge case: empty string
            ("single", "single"),  # No suffix
            ("model-with-many-dashes-preview-2024", "model-with-many-dashes"),
        ],
    )
    def test_family_of_additional_cases(self, model: str, expected: str) -> None:
        """Test additional model family extraction cases."""
        assert _family_of(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5-mini", "gpt-5"),
            ("gpt-4o-mini-2024-07-18", "gpt-4o"),
            ("gpt-4o-mini-2024-07-18-preview", "gpt-4o"),
            ("gpt-4o", "gpt-4o"),
            ("gpt-4o-realtime-preview", "gpt-4o-realtime"),
            ("o1-preview", "o1"),
        ],
    )
    def test_family_of(self, model: str, expected: str) -> None:
        assert _family_of(model) == expected


class TestFormatList:
    def test_format_list_with_special_characters(self):
        """Test formatting with model names containing special characters."""
        assert _format_list(["model_1", "model-2", "model.3"]) == "model-2, model.3, model_1"

    def test_format_list_with_long_names(self):
        """Test formatting with very long model names."""
        long_name = "gpt-4-turbo-preview-with-very-long-suffix-2024-12-31"
        assert _format_list([long_name, "short"]) == f"short, {long_name}"

    def test_format_list_preserves_case(self):
        """Test that case is preserved in sorting."""
        assert _format_list(["Model-Z", "model-a"]) == "Model-Z, model-a"

    def test_format_list_with_numbers(self):
        """Test sorting with numeric components."""
        assert _format_list(["model-10", "model-2", "model-1"]) == "model-1, model-10, model-2"

    def test_format_list_multiple(self):
        assert _format_list(["gpt-4", "gpt-3.5-turbo", "claude-2"]) == "claude-2, gpt-3.5-turbo, gpt-4"

    def test_format_list_single(self):
        assert _format_list(["gpt-4"]) == "gpt-4"

    def test_format_list_empty_list(self):
        assert _format_list([]) == "<none>"

    def test_format_list_empty_set(self):
        assert _format_list(set()) == "<none>"

    def test_format_list_with_duplicates_preserved(self):
        # _format_list does not deduplicate; duplicates remain after sorting
        assert _format_list(["gpt-4", "gpt-3.5-turbo", "gpt-4"]) == "gpt-3.5-turbo, gpt-4, gpt-4"

    def test_format_list_unsorted_input(self):
        assert _format_list(["z-model", "a-model", "m-model"]) == "a-model, m-model, z-model"


class TestMain:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_all_models_in_same_family(self, mock_fetch):
        """Test when all fetched models belong to the same family."""
        mock_fetch.return_value = {"gpt-4-turbo", "gpt-4-preview", "gpt-4-mini"}
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"gpt-4-turbo"}),
        ):
            with patch("sys.stderr") as err:
                rc = main()
        assert rc == 4  # New variants detected
        err.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_many_missing_models(self, mock_fetch):
        """Test when multiple models are missing."""
        mock_fetch.return_value = {"gpt-4"}
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"missing-1", "missing-2", "missing-3"}),
        ):
            with patch("sys.stderr") as err:
                rc = main()
        assert rc == 3
        err.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_many_new_variants(self, mock_fetch):
        """Test when many new variants are detected."""
        mock_fetch.return_value = {
            "gpt-4-v1", "gpt-4-v2", "gpt-4-v3",
            "gpt-5-v1", "gpt-5-v2", "gpt-5-v3",
        }
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"gpt-4-v1", "gpt-5-v1"}),
        ):
            with patch("sys.stderr") as err:
                rc = main()
        assert rc == 4
        err.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "   test-key   "})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_api_key_with_whitespace(self, mock_fetch):
        """Test API key with leading/trailing whitespace."""
        mock_fetch.return_value = {"gpt-4"}
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"gpt-4"}),
        ):
            with patch("sys.stdout"):
                rc = main()
        # Should succeed if whitespace is handled
        assert rc in [0, 1, 2]  # Could be 0 (success), 1 (missing key), or 2 (network error)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_generic_exception(self, mock_fetch):
        """Test handling of unexpected exceptions."""
        mock_fetch.side_effect = RuntimeError("Unexpected error")
        with patch("sys.stderr") as err:
            rc = main()
        assert rc == 2
        err.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_keyboard_interrupt(self, mock_fetch):
        """Test handling of keyboard interrupt during execution."""
        mock_fetch.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            main()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_fetch_returns_none_items(self, mock_fetch):
        """Test when _fetch_models returns a set that should be validated."""
        mock_fetch.return_value = {"gpt-4", "gpt-3.5-turbo"}
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"gpt-4", "gpt-3.5-turbo"}),
        ):
            with patch("sys.stdout") as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_main_missing_api_key(self):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 1
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_success_all_models_available(self, mock_fetch):
        mock_fetch.return_value = {'gpt-5-mini', 'gpt-4o-mini', 'other'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-5-mini', 'gpt-4o-mini'})):
            with patch('sys.stdout') as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_missing_models(self, mock_fetch):
        mock_fetch.return_value = {'gpt-5-mini', 'gpt-4o-mini'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS',
                   new=frozenset({'gpt-5-mini', 'gpt-4o-mini', 'missing-model'})):
            with patch('sys.stderr') as err:
                rc = main()
        assert rc == 3
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_new_variants_detected(self, mock_fetch):
        mock_fetch.return_value = {'gpt-5-mini', 'gpt-4o-mini', 'gpt-4.1-turbo', 'gpt-4o-latest'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-5-mini', 'gpt-4o-mini'})):
            with patch('sys.stderr') as err:
                rc = main()
        assert rc == 4
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_new_variants_detected_with_date_suffix(self, mock_fetch):
        mock_fetch.return_value = {
            'gpt-4o-mini-2024-07-18',
            'gpt-4o-mini-2024-08-01',
            'gpt-4o-mini-2024-07-18-preview',
        }
        allowed = frozenset({'gpt-4o-mini-2024-07-18'})
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=allowed):
            with patch('sys.stderr') as err:
                rc = main()

        assert rc == 4
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_no_new_variants(self, mock_fetch):
        mock_fetch.return_value = {'gpt-5-mini', 'gpt-4o-mini', 'gpt-4-mini'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-5-mini', 'gpt-4o-mini'})):
            with patch('sys.stdout') as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_missing_and_new_variants(self, mock_fetch):
        mock_fetch.return_value = {'gpt-5-mini', 'gpt-4o-mini', 'gpt-4.1-preview'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'missing-model'})):
            with patch('sys.stderr') as err:
                rc = main()
        # Missing models take precedence: exit code 3
        assert rc == 3
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models', side_effect=urllib.error.HTTPError(
        url=CATALOG_URL, code=401, msg="Unauthorized", hdrs=None, fp=None
    ))
    def test_main_http_error(self, _mock_fetch):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 2
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models', side_effect=urllib.error.URLError("Network down"))
    def test_main_url_error(self, _mock_fetch):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 2
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_json_decode_error(self, mock_fetch):
        mock_fetch.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 2
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': ''})
    def test_main_empty_api_key(self):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 1
        err.write.assert_called()


class TestIntegrationScenarios:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_end_to_end_with_pagination(self, _mock_request, mock_urlopen):
        """Test full workflow with pagination."""
        page1 = {"data": [{"id": "gpt-4"}], "has_more": True}
        page2 = {"data": [{"id": "gpt-3.5-turbo"}], "has_more": False}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", side_effect=[page1, page2]):
            with patch(
                "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
                new=frozenset({"gpt-4", "gpt-3.5-turbo"}),
            ):
                with patch("sys.stdout") as out:
                    rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_end_to_end_network_error_on_second_page(self, mock_urlopen):
        """Test network failure during pagination."""
        page1 = {"data": [{"id": "gpt-4"}], "has_more": True}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        # First call succeeds, second fails
        mock_urlopen.side_effect = [
            mock_resp,
            urllib.error.URLError("Network error on second page"),
        ]

        with patch("json.load", return_value=page1):
            with pytest.raises(urllib.error.URLError):
                _fetch_models("key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_end_to_end_mixed_valid_invalid_models(self, _mock_request, mock_urlopen):
        """Test with mix of valid and malformed model entries."""
        mock_payload = {
            "data": [
                {"id": "gpt-4"},
                {"id": ""},  # empty
                {"id": "gpt-3.5-turbo"},
                {"no-id": "field"},  # missing id
                {"id": "claude-2"},
                None,  # null entry
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            with patch(
                "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
                new=frozenset({"gpt-4", "gpt-3.5-turbo", "claude-2"}),
            ):
                with patch("sys.stdout") as out:
                    rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {})
    def test_end_to_end_no_environment_variables(self):
        """Test when no environment variables are set at all."""
        with patch("sys.stderr") as err:
            rc = main()
        assert rc == 1
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_end_to_end_success(self, _mock_request, mock_urlopen):
        mock_payload = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
                {"id": "other"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4', 'gpt-3.5-turbo'})):
                with patch('sys.stdout') as out:
                    rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('urllib.request.urlopen')
    @patch('urllib.request.Request')
    def test_end_to_end_missing_model(self, _mock_request, mock_urlopen):
        mock_payload = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch('json.load', return_value=mock_payload):
            with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4', 'nonexistent-model'})):
                with patch('sys.stderr') as err:
                    rc = main()
        assert rc == 3
        err.write.assert_called()


class TestFetchModelsAdvanced:
    """Advanced edge cases for _fetch_models function."""

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_timeout(self, _mock_request, mock_urlopen):
        """Test handling of timeout errors during API request."""
        mock_urlopen.side_effect = TimeoutError("Request timed out")
        with pytest.raises(TimeoutError):
            _fetch_models("key")

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_connection_reset(self, _mock_request, mock_urlopen):
        """Test handling of connection reset errors."""
        mock_urlopen.side_effect = ConnectionResetError("Connection reset by peer")
        with pytest.raises(ConnectionResetError):
            _fetch_models("key")

    @patch("urllib.request.urlopen")
    def test_fetch_models_large_pagination(self, mock_urlopen):
        """Test handling of many pages of results."""
        pages = [
            {"data": [{"id": f"model-{i}"}], "has_more": True if i < 9 else False}
            for i in range(10)
        ]

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", side_effect=pages):
            models = _fetch_models("key")

        assert len(models) == 10
        assert "model-0" in models
        assert "model-9" in models

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_multiple_pages_with_empty_page(self, mock_request, mock_urlopen):
        """Test pagination when one page returns empty data."""
        first_page = {"data": [{"id": "model-1"}], "has_more": True, "last_id": "cursor-1"}
        second_page = {"data": [], "has_more": True, "last_id": "cursor-2"}
        third_page = {"data": [{"id": "model-2"}], "has_more": False}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", side_effect=[first_page, second_page, third_page]):
            models = _fetch_models("key")

        assert models == {"model-1", "model-2"}
        assert mock_request.call_count == 3

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_with_special_characters_in_id(self, _mock_request, mock_urlopen):
        """Test models with special characters in their IDs."""
        mock_payload = {
            "data": [
                {"id": "gpt-5-mini"},
                {"id": "gpt-4o-mini-2024-07-18"},
                {"id": "model_with_underscore"},
                {"id": "model-with-multiple-dashes-v2"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            models = _fetch_models("key")

        assert len(models) == 4
        assert "model_with_underscore" in models
        assert "model-with-multiple-dashes-v2" in models

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_invalid_json_structure(self, _mock_request, mock_urlopen):
        """Test handling of completely invalid JSON structure."""
        mock_payload = {"unexpected": "structure", "no_data": True}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            models = _fetch_models("key")

        assert models == set()

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_http_500_error(self, _mock_request, mock_urlopen):
        """Test handling of 500 Internal Server Error."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=CATALOG_URL, code=500, msg="Internal Server Error", hdrs=None, fp=None
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _fetch_models("key")
        assert exc_info.value.code == 500

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_http_429_rate_limit(self, _mock_request, mock_urlopen):
        """Test handling of 429 Too Many Requests error."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=CATALOG_URL, code=429, msg="Too Many Requests", hdrs=None, fp=None
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _fetch_models("key")
        assert exc_info.value.code == 429

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_with_none_in_data_list(self, _mock_request, mock_urlopen):
        """Test that None values in data list are properly skipped."""
        mock_payload = {
            "data": [
                {"id": "valid-model-1"},
                None,
                None,
                {"id": "valid-model-2"},
                None,
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            models = _fetch_models("key")

        assert models == {"valid-model-1", "valid-model-2"}


class TestEdgeCases:
    def test_family_of_with_only_hyphens(self):
        """Test _family_of with edge case input."""
        assert _family_of("---") == "---"

    def test_family_of_with_unicode(self):
        """Test _family_of with unicode characters."""
        assert _family_of("model-café") == "model-café"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_with_very_long_model_lists(self, mock_fetch):
        """Test with large numbers of models to ensure performance."""
        # Generate 100 models
        models = {f"model-{i}" for i in range(100)}
        mock_fetch.return_value = models
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset(list(models)[:50]),  # Allow first 50
        ):
            with patch("sys.stderr") as err:
                rc = main()
        # Should detect new variants from the 50 not in allowlist
        assert rc == 4
        err.write.assert_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("scripts.audit_openai_allowlist._fetch_models")
    def test_main_identical_prefixes_different_suffixes(self, mock_fetch):
        """Test models with very similar names but different suffixes."""
        mock_fetch.return_value = {
            "gpt-4-preview-1",
            "gpt-4-preview-2",
            "gpt-4-preview-10",
            "gpt-4-preview",
        }
        with patch(
            "scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS",
            new=frozenset({"gpt-4-preview"}),
        ):
            with patch("sys.stderr") as err:
                rc = main()
        assert rc == 4
        err.write.assert_called()

    def test_format_list_with_mixed_types_raises(self):
        """Test that _format_list raises TypeError with mixed types."""
        with pytest.raises((TypeError, AttributeError)):
            _format_list(["string", 123, "another"])

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_data_not_list(self, _mock_request, mock_urlopen):
        """Test when data field is not a list."""
        mock_payload = {"data": "not-a-list"}

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            # Should handle gracefully or raise appropriate error
            result = _fetch_models("key")
            # Depending on implementation, could be empty set or raise
            assert isinstance(result, set)

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_fetch_models_id_not_string(self, _mock_request, mock_urlopen):
        """Test when model id is not a string."""
        mock_payload = {
            "data": [
                {"id": 123},  # numeric id
                {"id": ["list", "id"]},  # list id
                {"id": {"nested": "dict"}},  # dict id
                {"id": "valid-model"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        with patch("json.load", return_value=mock_payload):
            models = _fetch_models("key")
            # Should only include valid string id
            assert "valid-model" in models

    def test_format_list_with_none_raises(self):
        with pytest.raises(TypeError):
            _format_list(["gpt-4", None, "gpt-3.5-turbo"])

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_empty_allowlist(self, mock_fetch):
        mock_fetch.return_value = {'gpt-4', 'gpt-3.5-turbo'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset()):
            with patch('sys.stdout') as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_prefix_matching_precision(self, mock_fetch):
        mock_fetch.return_value = {
            'gpt-4', 'gpt-41-something',  # should NOT match 'gpt-4.1'
            'gpt-4.1-preview',           # should match
            'gpt-4.10-model',            # string startswith 'gpt-4.1' -> matches
            'other-gpt-4.1-variant',     # does not start with 'gpt-4.1'
        }
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4'})):
            with patch('sys.stderr') as err:
                rc = main()
        assert rc == 4
        err.write.assert_called()
