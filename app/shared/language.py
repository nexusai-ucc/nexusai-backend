"""Output-language control for LLM prompts.

The prompts of this backend are written in Spanish. With long prompts the model
follows the language of the *instructions* and ignores a vague rule such as
"answer in the same language as the material", so an English course got Spanish
questions, summaries and chat answers.

Instead of asking the model to guess, the backend detects the language of the
source text (material, forum posts, student question) and appends an explicit,
unambiguous directive to the last user message.

The directive is only added when the source is NOT in the language the prompts are
written in (Spanish). Spanish courses already work and, measured against the real
model, an extra "write in Spanish" line made the quiz generator refuse ("not enough
content") on material it accepted without it. When the language cannot be determined
(short or mixed text) nothing is added either, so the prompt behaves as before.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

# Function words that are frequent in one language and (almost) absent in the other.
# Ambiguous tokens (a, no, me, he, son, die, ...) are left out on purpose.
_EN_WORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "is",
        "are",
        "to",
        "in",
        "that",
        "this",
        "with",
        "for",
        "as",
        "it",
        "be",
        "can",
        "which",
        "when",
        "what",
        "how",
        "why",
        "does",
        "do",
        "not",
        "by",
        "from",
        "on",
        "at",
        "if",
        "was",
        "were",
        "will",
        "would",
        "should",
        "must",
        "have",
        "has",
        "been",
        "than",
        "then",
        "their",
        "they",
        "there",
        "these",
        "those",
        "its",
        "or",
        "an",
        "you",
        "your",
        "our",
        "we",
    ]
)
_ES_WORDS = frozenset(
    [
        "el",
        "la",
        "los",
        "las",
        "del",
        "que",
        "y",
        "en",
        "una",
        "es",
        "se",
        "por",
        "con",
        "para",
        "como",
        "al",
        "lo",
        "su",
        "sus",
        "más",
        "pero",
        "cuando",
        "qué",
        "cómo",
        "cuál",
        "cuáles",
        "este",
        "esta",
        "esto",
        "estos",
        "estas",
        "hay",
        "puede",
        "pueden",
        "debe",
        "deben",
        "entre",
        "sobre",
        "también",
        "ser",
        "fue",
        "muy",
        "ya",
        "sin",
        "cada",
        "donde",
        "porque",
        "tu",
        "tus",
        "nuestro",
        "nuestra",
        "según",
        "son",
        "está",
        "están",
    ]
)

if TYPE_CHECKING:
    from starlette.requests import Request

_TOKEN = re.compile(r"[a-záéíóúñü]+")

# Below this many function-word hits the text is too short to decide. Two is enough
# for a typical chat question ("What is a deadlock?") because one language must also
# dominate the other.
_MIN_HITS = 2
# One language must beat the other by this factor.
_DOMINANCE = 2.0

_NAMES = {"en": "English", "es": "Spanish"}

# Language the prompt templates are written in: no directive needed for it.
_PROMPT_LANGUAGE = "es"


def detect_language(*texts: str | None) -> str | None:
    """Return "en" or "es" for the given source texts, or None if unclear."""
    en = es = 0
    for text in texts:
        if not text:
            continue
        for token in _TOKEN.findall(text.lower()):
            if token in _EN_WORDS:
                en += 1
            elif token in _ES_WORDS:
                es += 1
    if en + es < _MIN_HITS:
        return None
    if en >= es * _DOMINANCE:
        return "en"
    if es >= en * _DOMINANCE:
        return "es"
    return None


def language_directive(lang: str) -> str:
    """Explicit instruction, written in English so it does not depend on the prompt language."""
    name = _NAMES[lang]
    return (
        f"IMPORTANT: write your entire answer in {name} "
        "(every text field, including every JSON string value)."
    )


def with_language_directive(
    messages: list[dict[str, str]],
    *sources: str | None,
    fallback: str | None = None,
) -> list[dict[str, str]]:
    """Return a copy of ``messages`` whose last user message ends with a language directive.

    The language is detected from ``sources``. When the sources are too short or
    mixed to decide (a two-word question such as "foreign key"), ``fallback`` is
    used instead: callers pass the language of the user's interface. A text that
    clearly is in another language wins over the fallback. Nothing is added when
    the language cannot be determined or is the prompts' own language, so callers
    can use this unconditionally.
    """
    lang = detect_language(*sources) or fallback
    if lang is None or lang == _PROMPT_LANGUAGE:
        return messages
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            message["content"] = f"{message['content']}\n\n{language_directive(lang)}"
            break
    return out


def ui_language(request: Request) -> str | None:
    """Return "en" or "es" from the ``Accept-Language`` header, or None.

    The Moodle plugin sends the language of the user's interface in that header.
    It is the right signal for fixed messages that have no text to detect the
    language from (for example "this course has no indexed material yet").
    """
    header = request.headers.get("accept-language", "")
    for part in header.split(","):
        primary = part.split(";")[0].strip().lower().replace("_", "-").split("-")[0]
        if primary in _NAMES:
            return primary
    return None


def resolve_language(request: Request, *texts: str | None) -> str | None:
    """Interface language from the header, else detected from ``texts``."""
    return ui_language(request) or detect_language(*texts)


def localized(lang: str | None, spanish: str, english: str) -> str:
    """Pick the English text for "en"; Spanish otherwise (the historical default)."""
    return english if lang == "en" else spanish
