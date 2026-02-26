"""
Tests for the market-research assistant:

- Structured output: Summary, Sources, markdown links (required only for full answers, not clarification).
- Relevance: summary and sources match the user request (heuristic).
- Refusal: non-industry requests get a clear refusal message.
- Grounding: summary content traceable to sources (heuristic; cannot distinguish paraphrasing from hallucination).

We do not test WikipediaRetriever(top_k_results=5); it is a provided dependency.
"""

import re  # Regex for parsing response text (e.g. source links, word tokenization, sentence splits)
import pytest  # Test runner for discovery and running tests
from stop_words import get_stop_words

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EXPECTED_SOURCE_COUNT = 5
SUMMARY_WORD_LIMIT = 500
_STOPWORDS = set(get_stop_words("english"))

# -----------------------------------------------------------------------------
# 1. Structured output helpers
# Format "Summary; Sources" required only when the LLM retrieved from Wikipedia and gave
# a full answer. Do not run structure checks (Summary, Sources, word count, source count)
# when the response is a clarification or a refusal (non-industry); use
# _should_apply_structure_checks() before applying these checks.
# -----------------------------------------------------------------------------

def _count_source_links(text: str) -> int:
    """Count markdown links [text](url) in the response."""
    link_pattern = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    return len(link_pattern.findall(text))


def _get_summary_content(text: str) -> str | None:
    """Summary section text (between Summary heading and Sources), or None if not parseable."""
    lower = text.lower()
    if "sources" not in lower or "summary" not in lower:
        return None
    idx = lower.find("sources")
    before_sources = text[:idx]
    if "summary" not in before_sources.lower():
        return None
    lines = before_sources.strip().splitlines()
    if len(lines) <= 1:
        return None
    return "\n".join(lines[1:]).strip() or None


def _is_clarification_response(response: str) -> bool:
    """True if the model is asking for clarification (e.g. which industry) rather than giving a full answer."""
    lower = response.lower()
    clarification_phrases = [
        "clarify", "which industry", "please specify", "could you tell me",
        "what do you mean", "which one", "need more information", "unclear",
        "ambiguous", "please choose", "specify which", "narrow down",
    ]
    if not any(p in lower for p in clarification_phrases):
        return False
    return _count_source_links(response) <= 1


def _has_summary_section(text: str) -> bool:
    """True if response has a Summary section with at least 20 chars of content before Sources."""
    content = _get_summary_content(text)
    return content is not None and len(content) >= 20


def _has_sources_section(text: str) -> bool:
    """True if response has a Sources section with content after the title."""
    lower = text.lower()
    if "sources" not in lower:
        return False
    idx = lower.find("sources")
    after = text[idx + len("sources") :].lstrip(": \t\n")
    return len(after.strip()) >= 1


def _indicates_limited_sources(text: str) -> bool:
    """
    True if the response explains that little information was found for the industry.
    In that case we allow fewer than EXPECTED_SOURCE_COUNT sources without failing the check.
    """
    lower = text.lower()
    phrases = [
        "limited information", "few sources", "not much information", "could not find",
        "little information", "limited sources", "only a few", "only found",
        "limited results", "few results",
    ]
    return any(p in lower for p in phrases)


def _has_valid_source_count(text: str, required: int = EXPECTED_SOURCE_COUNT) -> bool:
    """True if link count >= required, or fewer with 'limited information' explanation."""
    count = _count_source_links(text)
    if count >= required:
        return True
    return count >= 1 and _indicates_limited_sources(text)


def _summary_word_count(text: str) -> int | None:
    """Word count of the Summary section, or None if not parseable."""
    content = _get_summary_content(text)
    return len(content.split()) if content else None


def _get_source_link_texts(text: str) -> list[str]:
    """Extract link titles from each [title](url) in the response."""
    link_pattern = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    return link_pattern.findall(text)


# -----------------------------------------------------------------------------
# 1. Structured output tests
# -----------------------------------------------------------------------------

def test_clarification_response_not_required_to_have_summary_sources():
    """Clarification replies (no retrieval yet) must not be required to have Summary/Sources."""
    response = (
        "Could you clarify which industry you mean? For example: pharmaceutical, "
        "fintech, or fashion? I'll then search for relevant market information."
    )
    assert _is_clarification_response(response)
    assert not _has_sources_section(response)
    assert not _has_summary_section(response)


def test_structured_output_has_summary_and_sources():
    """Full answers must have both Summary and Sources sections with content."""
    good = """Summary
The pharmaceutical industry develops drugs and is regulated by agencies.

Sources:
* [Pharmaceutical industry](https://en.wikipedia.org/wiki/Pharmaceutical_industry)
* [Drug development](https://en.wikipedia.org/wiki/Drug_development)
"""
    assert _has_sources_section(good)
    assert _has_summary_section(good)

    bad_no_sources = "Summary\nThe industry is growing.\n"
    assert not _has_sources_section(bad_no_sources)

    bad_no_summary = """Sources:
* [Article one](https://en.wikipedia.org/wiki/One)
* [Article two](https://en.wikipedia.org/wiki/Two)
"""
    assert _has_sources_section(bad_no_summary)
    assert not _has_summary_section(bad_no_summary)


def test_structured_output_has_source_links():
    """Full answers should have 5 sources; fewer only if 'limited information' is stated."""
    good_5 = """Summary\nSome analysis.\n\nSources:
* [1](https://a)\n* [2](https://b)\n* [3](https://c)\n* [4](https://d)\n* [5](https://e)
"""
    assert _count_source_links(good_5) == EXPECTED_SOURCE_COUNT
    assert _has_valid_source_count(good_5)

    few_with_explanation = """Summary\nLimited information available.\n\nSources:\n* [1](https://a)\n* [2](https://b)"""
    assert _has_valid_source_count(few_with_explanation)

    few_no_explanation = """Summary\nSome analysis.\n\nSources:\n* [1](https://a)\n* [2](https://b)"""
    assert not _has_valid_source_count(few_no_explanation)


def test_structured_output_rejects_summary_heading_with_no_content():
    """Summary heading with only blank space before Sources is not a valid Summary section."""
    response = "Summary\n\nSources:\n* [Some article](https://en.wikipedia.org/wiki/Some)"
    assert _has_sources_section(response)
    assert not _has_summary_section(response)


def test_structured_output_rejects_sources_heading_with_no_content():
    """Sources heading with nothing after it is not a valid Sources section."""
    response = "Summary\nThe pharmaceutical industry is regulated.\n\nSources:\n"
    assert _has_summary_section(response)
    assert not _has_sources_section(response)


def test_structured_output_summary_under_500_words():
    """Summary section must have at most 500 words."""
    good = """Summary\nThe pharmaceutical industry develops drugs and is regulated.\n\nSources:\n* [X](https://x)"""
    count = _summary_word_count(good)
    assert count is not None and count <= SUMMARY_WORD_LIMIT

    long_summary = "Summary\n" + " word" * 501 + "\n\nSources:\n* [Link](https://example.com)"
    assert _summary_word_count(long_summary) == 501


# -----------------------------------------------------------------------------
# 2. Relevance helpers and test
# Heuristic: request terms should appear in summary and in at least one source title.
# -----------------------------------------------------------------------------

def _extract_request_terms(request: str) -> set[str]:
    """Meaningful terms from the user request (lowercase, no stopwords, len > 1)."""
    words = set(re.findall(r"[a-z0-9]+", request.lower()))
    return {w for w in words if len(w) > 1 and w not in _STOPWORDS}


def _summary_relevant_to_request(summary_text: str, request_terms: set[str], min_terms: int = 1) -> bool:
    """True if at least min_terms request terms appear in the summary."""
    if not summary_text or not request_terms:
        return not request_terms
    summary_terms = set(re.findall(r"[a-z0-9]+", summary_text.lower()))
    return len(request_terms & summary_terms) >= min_terms


def _sources_relevant_to_request(link_texts: list[str], request_terms: set[str], min_sources: int = 1) -> bool:
    """True if at least min_sources link titles contain at least one request term."""
    if not request_terms or not link_texts:
        return not link_texts or not request_terms
    n = sum(1 for t in link_texts if request_terms & set(re.findall(r"[a-z0-9]+", t.lower())))
    return n >= min_sources


def test_relevance_summary_and_sources_match_user_request():
    """Summary and at least one source should be relevant to the user request (heuristic)."""
    request = "Give me a market overview of the pharmaceutical industry."
    terms = _extract_request_terms(request)
    assert "pharmaceutical" in terms or "industry" in terms

    good = """Summary\nThe pharmaceutical industry develops drugs and is regulated.\n\nSources:
* [Pharmaceutical industry](https://en.wikipedia.org/wiki/Pharmaceutical_industry)
* [Drug development](https://en.wikipedia.org/wiki/Drug_development)
"""
    summary = _get_summary_content(good)
    links = _get_source_link_texts(good)
    assert summary and _summary_relevant_to_request(summary, terms, min_terms=1)
    assert _sources_relevant_to_request(links, terms, min_sources=1)

    bad = """Summary\nThe moon is made of cheese.\n\nSources:\n* [Moon](https://m)\n* [Cheese](https://c)"""
    assert not _summary_relevant_to_request(_get_summary_content(bad) or "", terms, min_terms=1)
    assert not _sources_relevant_to_request(_get_source_link_texts(bad), terms, min_sources=1)


# -----------------------------------------------------------------------------
# 3. Refusal for non-industry requests
# -----------------------------------------------------------------------------

def _is_refusal_for_non_industry(response: str) -> bool:
    """True if the response refuses because the request is not industry-related."""
    lower = response.lower()
    phrases = [
        "only for industry", "only for market research", "industry and market research",
        "market research only", "industry-related", "industry related", "not able to help",
        "cannot help", "decline", "designed for industry", "support only", "industry or market",
    ]
    return any(p in lower for p in phrases)


def _should_apply_structure_checks(response: str) -> bool:
    """True if Summary/Sources/word-count/source-count checks should be run. False for clarification or refusal."""
    return not _is_clarification_response(response) and not _is_refusal_for_non_industry(response)


def test_refusal_for_non_industry_request():
    """Non-industry requests should get a refusal message, not a full answer."""
    refusal = (
        "I'm set up to support industry and market research only. "
        "I can't answer general knowledge questions. "
        "Please ask an industry-related question (e.g. pharmaceutical, fintech, fashion)."
    )
    assert _is_refusal_for_non_industry(refusal)
    assert not _is_refusal_for_non_industry("Summary\nParis is the capital.\n\nSources:\n* [France](https://f)")


def test_refusal_response_exempt_from_structure_checks():
    """When the LLM refuses (non-industry), structure checks must be skipped; missing Summary/Sources is OK."""
    refusal = (
        "I'm set up to support industry and market research only. "
        "Please ask an industry-related question."
    )
    assert not _should_apply_structure_checks(refusal)
    assert not _has_sources_section(refusal)
    assert not _has_summary_section(refusal)


# -----------------------------------------------------------------------------
# 4. Grounding / anti-hallucination
# Heuristics only: high overlap = likely grounded; low = possible hallucination.
# Limitations: Cannot distinguish paraphrasing from hallucination; different
# summaries each run are expected; high overlap does not prove no invented details.
# -----------------------------------------------------------------------------

def _normalize_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _word_overlap_ratio(summary: str, source_texts: list[str]) -> float:
    """Fraction of summary words that appear in any source (0.0–1.0)."""
    summary_words = _normalize_words(summary)
    if not summary_words:
        return 1.0
    source_words = set().union(*(_normalize_words(s) for s in source_texts))
    return sum(1 for w in summary_words if w in source_words) / len(summary_words)


def _sentence_grounding_score(summary: str, source_texts: list[str]) -> float:
    """Fraction of sentences with >= half of their words in sources."""
    source_words = set().union(*(_normalize_words(s) for s in source_texts))
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", summary) if s.strip()]
    if not sentences:
        return 1.0
    grounded = 0
    for sent in sentences:
        words = _normalize_words(sent)
        if not words or sum(1 for w in words if w in source_words) / len(words) >= 0.5:
            grounded += 1
    return grounded / len(sentences)


def test_summary_grounded_in_sources():
    """Grounded summary (paraphrase of sources) gets high overlap and sentence score."""
    sources = [
        "The pharmaceutical industry researches and develops drugs. It is highly regulated.",
        "Major players include large companies. Revenue is in the billions.",
    ]
    summary = "The pharmaceutical industry develops drugs and is highly regulated. Major players have large revenue."
    assert _word_overlap_ratio(summary, sources) >= 0.5
    assert _sentence_grounding_score(summary, sources) >= 0.9


def test_summary_hallucination_detected():
    """Summary with an invented sentence gets sentence score < 1.0."""
    sources = ["The pharmaceutical industry develops drugs. It is regulated."]
    summary = "The pharmaceutical industry develops drugs. In 2030 global revenue reached 999 trillion dollars."
    assert _sentence_grounding_score(summary, sources) < 1.0
    assert _word_overlap_ratio(summary, sources) < 1.0


def test_summary_fully_grounded_scores_high():
    """Summary that only paraphrases sources gets very high scores."""
    sources = ["Biotechnology uses living systems. It applies to medicine and agriculture."]
    summary = "Biotechnology uses living systems and applies to medicine and agriculture."
    assert _word_overlap_ratio(summary, sources) >= 0.8
    assert _sentence_grounding_score(summary, sources) >= 0.99


def test_summary_fully_hallucinated_scores_low():
    """Completely unrelated summary gets low overlap and sentence score."""
    sources = ["Pharmaceutical industry is regulated."]
    summary = "The moon is made of cheese. In 2050 everyone will live on Mars."
    assert _word_overlap_ratio(summary, sources) < 0.3
    assert _sentence_grounding_score(summary, sources) < 0.5
