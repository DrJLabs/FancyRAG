#\!/usr/bin/env python3
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

# Ensure repository root is on sys.path so 'scripts' is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.audit_openai_allowlist import (
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
                "User-Agent": "graphrag-allowlist-audit",
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
                {"id": "gpt-4.1-mini"},
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

        assert models == {"gpt-4.1-mini", "gpt-4o-mini"}
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
            ("gpt-4.1-mini", "gpt-4.1"),
            ("gpt-4o-mini-2024-07-18", "gpt-4o"),
            ("gpt-4o-mini-2024-07-18-preview", "gpt-4o"),
            ("gpt-4o", "gpt"),
            ("gpt-4o-realtime-preview", "gpt-4o-realtime"),
            ("o1-preview", "o1"),
        ],
    )
    def test_family_of(self, model: str, expected: str) -> None:
        assert _family_of(model) == expected


class TestFormatList:
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
    @patch.dict(os.environ, {}, clear=True)
    def test_main_missing_api_key(self):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 1
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_success_all_models_available(self, mock_fetch):
        mock_fetch.return_value = {'gpt-4.1-mini', 'gpt-4o-mini', 'other'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4.1-mini', 'gpt-4o-mini'})):
            with patch('sys.stdout') as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_missing_models(self, mock_fetch):
        mock_fetch.return_value = {'gpt-4.1-mini', 'gpt-4o-mini'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS',
                   new=frozenset({'gpt-4.1-mini', 'gpt-4o-mini', 'missing-model'})):
            with patch('sys.stderr') as err:
                rc = main()
        assert rc == 3
        err.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_new_variants_detected(self, mock_fetch):
        mock_fetch.return_value = {'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4.1-turbo', 'gpt-4o-latest'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4.1-mini', 'gpt-4o-mini'})):
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
        mock_fetch.return_value = {'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4-mini'}
        with patch('scripts.audit_openai_allowlist.ALLOWED_CHAT_MODELS', new=frozenset({'gpt-4.1-mini', 'gpt-4o-mini'})):
            with patch('sys.stdout') as out:
                rc = main()
        assert rc == 0
        out.write.assert_called()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('scripts.audit_openai_allowlist._fetch_models')
    def test_main_missing_and_new_variants(self, mock_fetch):
        mock_fetch.return_value = {'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4.1-preview'}
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

    @patch.dict(os.environ, {'OPENAI_API_KEY': ''})
    def test_main_empty_api_key(self):
        with patch('sys.stderr') as err:
            rc = main()
        assert rc == 1
        err.write.assert_called()


class TestIntegrationScenarios:
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


class TestEdgeCases:
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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])