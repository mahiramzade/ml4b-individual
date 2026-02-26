"""
Tests for the market-research assistant:

- Wikipedia tool: mocked retriever, expected shape (title, url, summary), callable with different industries.
- Structured output: assistant response has Summary, Sources, and markdown links.
- Grounding: summary content is traceable to provided sources (no hallucination).
"""

import re
from unittest.mock import MagicMock, patch

import pytest

# We import get_wikipedia_posts inside each test, after patching main.wikipedia_retriever.
# That way the real Wikipedia retriever is never used, and we avoid network calls and slow tests.


def _make_mock_document(title: str, url: str, summary: str) -> MagicMock:
    """
    Build a Document-like object with .metadata for get_wikipedia_posts.
    The real retriever returns documents that have a .metadata dict; the tool uses post.metadata for each result.
    """
    doc = MagicMock()
    doc.metadata = {"title": title, "url": url, "summary": summary}
    return doc


# --- Wikipedia tool tests (mock retriever, shape, callable with different industries) ---


@patch("main.wikipedia_retriever")
def test_get_wikipedia_posts_returns_expected_shape(mock_retriever):
    """Returned list has correct length; each item has title, url, summary as strings."""
    # Arrange: when the retriever is invoked, it returns two fake documents with the expected metadata shape.
    mock_retriever.invoke.return_value = [
        _make_mock_document("Pharma Industry", "https://en.wikipedia.org/wiki/Pharma", "Summary about pharma."),
        _make_mock_document("Drug Development", "https://en.wikipedia.org/wiki/Drug", "Summary about drugs."),
    ]
    from main import get_wikipedia_posts

    # Act: call the tool the same way the agent would (LangChain tools take a dict of argument names to values).
    result = get_wikipedia_posts.invoke({"industry_name": "pharmaceutical"})

    # Assert: result must be a list of dicts, each with exactly title, url, summary as non-empty strings.
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, dict)
        assert "title" in item and "url" in item and "summary" in item
        assert isinstance(item["title"], str) and item["title"]
        assert isinstance(item["url"], str) and item["url"]
        assert isinstance(item["summary"], str)


@patch("main.wikipedia_retriever")
def test_get_wikipedia_posts_callable_with_different_industry_names(mock_retriever):
    """Tool is callable with different industry names; invoke is called with the given name."""
    # First call: pharmaceutical.
    mock_retriever.invoke.return_value = [
        _make_mock_document("Title", "https://example.com", "Summary"),
    ]
    from main import get_wikipedia_posts

    get_wikipedia_posts.invoke({"industry_name": "pharmaceutical"})
    # Check that the underlying retriever was called with the same industry name we passed to the tool.
    mock_retriever.invoke.assert_called_once_with("pharmaceutical")

    # Second call: different industry (fintech). Reset mock so we can assert on the new call only.
    mock_retriever.reset_mock()
    mock_retriever.invoke.return_value = [
        _make_mock_document("Fintech", "https://example.com/fintech", "Summary."),
    ]
    get_wikipedia_posts.invoke({"industry_name": "fintech"})
    mock_retriever.invoke.assert_called_once_with("fintech")


@patch("main.wikipedia_retriever")
def test_get_wikipedia_posts_empty_result_still_has_shape(mock_retriever):
    """When retriever returns no documents, tool returns empty list (valid shape)."""
    # Simulate no Wikipedia results for the query (e.g. very obscure industry name).
    mock_retriever.invoke.return_value = []
    from main import get_wikipedia_posts

    result = get_wikipedia_posts.invoke({"industry_name": "unknown industry"})
    # Tool should return an empty list, not None or an error; downstream code can safely iterate.
    assert result == []


# --- Structured output check tests ---
# These helpers and tests validate that the assistant's reply follows the required format:
# a Summary section and a Sources section with markdown links to the references.


def _has_summary_section(text: str) -> bool:
    """
    True if response contains a Summary section (heading or label).
    We require the word 'summary' to appear and to occur before 'sources' (or at the start),
    so that we distinguish a real Summary block from casual use of the word elsewhere.
    """
    return "summary" in text.lower() and (
        "summary" in text.lower().split("sources")[0] or text.strip().lower().startswith("summary")
    )


def _has_sources_section(text: str) -> bool:
    """True if the response mentions Sources (the section where references are listed)."""
    return "sources" in text.lower()


def _count_source_links(text: str) -> int:
    """
    Count markdown links of the form [link text](url) in the response.
    The system prompt asks for sources like: * [Post title 1](url1). This regex finds all such links.
    """
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    return len(link_pattern.findall(text))


# We expect 5 sources by default (system prompt asks for up to 5). Fewer allowed only when info is limited.
EXPECTED_SOURCE_COUNT = 5


def _indicates_limited_sources(text: str) -> bool:
    """
    True if the response explains that little information was found for the industry.
    In that case we allow fewer than EXPECTED_SOURCE_COUNT sources without failing the check.
    """
    lower = text.lower()
    phrases = [
        "limited information",
        "few sources",
        "not much information",
        "could not find",
        "little information",
        "limited sources",
        "only a few",
        "only found",
        "limited results",
        "few results",
    ]
    return any(phrase in lower for phrase in phrases)


def _has_valid_source_count(text: str, required: int = EXPECTED_SOURCE_COUNT) -> bool:
    """
    True if the response has enough source links: either >= required (e.g. 5), or fewer than required
    but the text indicates limited information was available (so fewer sources are acceptable).
    """
    count = _count_source_links(text)
    if count >= required:
        return True
    # Allow fewer than required only when the assistant explains that little info was found.
    if count >= 1 and _indicates_limited_sources(text):
        return True
    return False


def test_structured_output_has_summary_and_sources():
    """Assistant response must contain both Summary and Sources sections."""
    # Example of a well-formatted response that matches the system prompt.
    good_response = """Summary
The pharmaceutical industry develops drugs and is regulated by agencies.

Sources:
* [Pharmaceutical industry](https://en.wikipedia.org/wiki/Pharmaceutical_industry)
* [Drug development](https://en.wikipedia.org/wiki/Drug_development)
"""
    assert _has_sources_section(good_response)
    assert _has_summary_section(good_response)


def test_structured_output_has_source_links():
    """Assistant response should have 5 sources; fewer only if the response indicates limited information."""
    # Normal case: 5 sources as requested by the system prompt.
    good_response_5_sources = """Summary
Some analysis here.

Sources:
* [Post title 1](https://en.wikipedia.org/wiki/One)
* [Post title 2](https://en.wikipedia.org/wiki/Two)
* [Post title 3](https://en.wikipedia.org/wiki/Three)
* [Post title 4](https://en.wikipedia.org/wiki/Four)
* [Post title 5](https://en.wikipedia.org/wiki/Five)
"""
    assert _count_source_links(good_response_5_sources) == EXPECTED_SOURCE_COUNT
    assert _has_valid_source_count(good_response_5_sources)

    # Exception: fewer than 5 sources is acceptable only when the response explains limited info.
    response_few_sources_with_explanation = """Summary
Limited information available for this niche industry.

Sources:
* [Only result 1](https://en.wikipedia.org/wiki/One)
* [Only result 2](https://en.wikipedia.org/wiki/Two)
"""
    assert _count_source_links(response_few_sources_with_explanation) < EXPECTED_SOURCE_COUNT
    assert _has_valid_source_count(response_few_sources_with_explanation)

    # Invalid: fewer than 5 sources with no explanation that info was limited.
    response_few_sources_no_explanation = """Summary
Some analysis.

Sources:
* [One](https://en.wikipedia.org/wiki/One)
* [Two](https://en.wikipedia.org/wiki/Two)
"""
    assert not _has_valid_source_count(response_few_sources_no_explanation)


def test_structured_output_rejects_missing_sources():
    """Response that has no actual source links fails the strict structured check."""
    bad_response = "Summary\nOnly a summary, no sources listed."
    # The word "sources" might appear in the sentence; what we really care about is presence of links.
    assert _count_source_links(bad_response) == 0


def test_structured_output_rejects_empty_response():
    """Empty or trivial response has no structure (no Summary, no Sources)."""
    assert not _has_summary_section("")
    assert not _has_sources_section("")


# --- Grounding / anti-hallucination tests ---
# We check that the summary content is traceable to the provided sources (e.g. Wikipedia snippets).
# High overlap = summary words appear in sources; low overlap or low sentence-level score suggests hallucination.


def _normalize_words(text: str) -> set[str]:
    """
    Turn text into a set of lowercase alphanumeric tokens (words and numbers).
    Punctuation and spaces are ignored so we can compare summary vs source content fairly.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _word_overlap_ratio(summary: str, source_texts: list[str]) -> float:
    """
    Ratio of summary words that appear in any of the source texts (0.0 to 1.0).
    We do not filter stopwords, so numbers and specific terms (e.g. "2030", "trillion") affect the score
    and help detect invented statistics.
    """
    summary_words = _normalize_words(summary)
    if not summary_words:
        return 1.0
    # Build one set of all words that appear in any source.
    source_words = set()
    for s in source_texts:
        source_words |= _normalize_words(s)
    found = sum(1 for w in summary_words if w in source_words)
    return found / len(summary_words)


def _sentence_grounding_score(summary: str, source_texts: list[str]) -> float:
    """
    Fraction of sentences in the summary that have at least half of their words present in the sources.
    Splits on . ! ? so each sentence is scored separately. If one sentence is fully made up,
    that sentence will have low overlap and the overall score drops below 1.0.
    """
    source_words = set()
    for s in source_texts:
        source_words |= _normalize_words(s)
    # Split into sentences (simple: split on sentence-ending punctuation followed by space).
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", summary) if s.strip()]
    if not sentences:
        return 1.0
    grounded = 0
    for sent in sentences:
        words = _normalize_words(sent)
        if not words:
            grounded += 1
            continue
        in_source = sum(1 for w in words if w in source_words)
        # Sentence is "grounded" if at least half of its words appear in the sources.
        if in_source / len(words) >= 0.5:
            grounded += 1
    return grounded / len(sentences)


def test_summary_grounded_in_sources():
    """When the summary only uses information from the sources, both overlap and sentence score are high."""
    sources = [
        "The pharmaceutical industry researches and develops drugs. It is highly regulated.",
        "Major players include large companies. Revenue is in the billions.",
    ]
    summary = "The pharmaceutical industry develops drugs and is highly regulated. Major players have large revenue."
    # All summary content is paraphrased from the sources, so we expect high scores.
    assert _word_overlap_ratio(summary, sources) >= 0.5
    assert _sentence_grounding_score(summary, sources) >= 0.9


def test_summary_hallucination_detected():
    """When the summary adds a fact not present in sources, sentence-level grounding score drops below 1.0."""
    sources = [
        "The pharmaceutical industry develops drugs. It is regulated.",
    ]
    # First sentence is from sources; second sentence is invented (2030, 999 trillion, etc.).
    summary = "The pharmaceutical industry develops drugs. In 2030 global revenue reached 999 trillion dollars."
    overlap = _word_overlap_ratio(summary, sources)
    sentence_score = _sentence_grounding_score(summary, sources)
    # The invented sentence has many words not in sources, so not all sentences are grounded.
    assert sentence_score < 1.0
    # Overall word overlap may still be moderate because words like "revenue" and "industry" appear in both.
    assert overlap < 1.0


def test_summary_fully_grounded_scores_high():
    """A summary that only paraphrases the sources should get very high overlap and sentence scores."""
    sources = ["Biotechnology uses living systems. It applies to medicine and agriculture."]
    summary = "Biotechnology uses living systems and applies to medicine and agriculture."
    assert _word_overlap_ratio(summary, sources) >= 0.8
    assert _sentence_grounding_score(summary, sources) >= 0.99


def test_summary_fully_hallucinated_scores_low():
    """A summary with no factual overlap to the sources (complete hallucination) gets low scores."""
    sources = ["Pharmaceutical industry is regulated."]
    summary = "The moon is made of cheese. In 2050 everyone will live on Mars."
    # Almost no words in common; most sentences have no grounding in the source.
    assert _word_overlap_ratio(summary, sources) < 0.3
    assert _sentence_grounding_score(summary, sources) < 0.5
