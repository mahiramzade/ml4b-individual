"""Shared constants for the market-research assistant and tests."""

# Number of Wikipedia sources to fetch per query and to require in full answers (or fewer with "limited information").
EXPECTED_SOURCE_COUNT = 5

# Maximum words allowed in the Summary section of a full answer.
SUMMARY_WORD_LIMIT = 500

# Maximum tokens allowed per chat session before the user must reset; used for progress bar and cost control.
TOKEN_LIMIT = 20000
