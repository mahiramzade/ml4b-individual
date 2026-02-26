"""
Pytest configuration: load .env for OPENAI_API_KEY; provide fixtures for integration tests.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip integration tests when no API key (e.g. in CI without secrets).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
skip_if_no_api_key = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set (use .env or export); integration tests require it",
)

# Register integration marker (avoids PytestUnknownMarkWarning)
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration (real API calls)")


@pytest.fixture
def api_key():
    """API key for OpenAI; integration tests that need it should use skip_if_no_api_key."""
    return OPENAI_API_KEY
