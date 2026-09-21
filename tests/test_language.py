"""Tests for app.shared.language — detection and output-language directive."""

import pytest

from app.shared.language import (
    detect_language,
    language_directive,
    localized,
    resolve_language,
    ui_language,
    with_language_directive,
)

EN_TEXT = (
    "A primary key uniquely identifies each row of a table and cannot be NULL. "
    "A foreign key references the primary key of another table."
)
ES_TEXT = (
    "Una clave primaria identifica de forma única cada fila de una tabla y no "
    "puede ser nula. Una clave foránea hace referencia a la clave primaria de otra tabla."
)


def test_detects_english():
    assert detect_language(EN_TEXT) == "en"


def test_detects_spanish():
    assert detect_language(ES_TEXT) == "es"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is a deadlock?", "en"),
        ("What does a foreign key do?", "en"),
        ("What is the difference between 2NF and 3NF?", "en"),
        ("¿Qué es un deadlock?", "es"),
        ("¿Cuándo es el parcial y cuánto dura?", "es"),
    ],
)
def test_detects_short_chat_questions(question, expected):
    assert detect_language(question) == expected


def test_combines_several_sources():
    assert detect_language("What is a deadlock?", EN_TEXT) == "en"


@pytest.mark.parametrize("text", [None, "", "hi", "SQL JOIN", "clave primaria"])
def test_too_short_is_undetermined(text):
    assert detect_language(text) is None


def test_mixed_text_is_undetermined():
    assert detect_language(EN_TEXT, ES_TEXT) is None


def test_directive_names_the_language_in_english():
    assert "English" in language_directive("en")
    assert "Spanish" in language_directive("es")


def test_appends_directive_to_the_last_user_message_only():
    messages = [
        {"role": "system", "content": "Sos un asistente."},
        {"role": "user", "content": "primero"},
        {"role": "assistant", "content": "respuesta"},
        {"role": "user", "content": "Generá preguntas."},
    ]

    out = with_language_directive(messages, EN_TEXT)

    assert out[-1]["content"].startswith("Generá preguntas.")
    assert out[-1]["content"].endswith(language_directive("en"))
    assert out[0] == messages[0]
    assert out[1] == messages[1]
    assert out[2] == messages[2]


def test_does_not_mutate_the_input():
    messages = [{"role": "user", "content": "x"}]

    with_language_directive(messages, EN_TEXT)

    assert messages == [{"role": "user", "content": "x"}]


def test_returns_messages_unchanged_when_language_is_unclear():
    messages = [{"role": "user", "content": "x"}]

    assert with_language_directive(messages, "hi") is messages
    assert with_language_directive(messages, EN_TEXT, ES_TEXT) is messages


def test_spanish_source_is_left_alone():
    """Prompts are written in Spanish; adding a directive there made the quiz refuse."""
    messages = [{"role": "user", "content": "x"}]

    assert with_language_directive(messages, ES_TEXT) is messages


def test_no_user_message_is_a_noop():
    messages = [{"role": "system", "content": "x"}]

    assert with_language_directive(messages, EN_TEXT) == messages


class _Req:
    """Minimal stand-in for a request: only its headers matter here."""

    def __init__(self, accept_language: str | None = None) -> None:
        self.headers = (
            {} if accept_language is None else {"accept-language": accept_language}
        )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("en", "en"),
        ("es", "es"),
        ("en-US,en;q=0.9", "en"),
        ("es_AR", "es"),
        ("fr, en;q=0.5", "en"),
        ("fr", None),
        ("", None),
        (None, None),
    ],
)
def test_ui_language_reads_the_accept_language_header(header, expected):
    assert ui_language(_Req(header)) == expected


def test_resolve_language_prefers_the_header_over_detection():
    assert resolve_language(_Req("es"), EN_TEXT) == "es"
    assert resolve_language(_Req(), EN_TEXT) == "en"
    assert resolve_language(_Req(), "hi") is None


def test_localized_defaults_to_spanish():
    assert localized("en", "hola", "hello") == "hello"
    assert localized("es", "hola", "hello") == "hola"
    assert localized(None, "hola", "hello") == "hola"


def test_fallback_is_used_when_the_sources_are_too_short_to_decide():
    """A two-word question such as "foreign key" has no function words to detect."""
    messages = [{"role": "user", "content": "foreign key"}]

    out = with_language_directive(messages, "foreign key", fallback="en")

    assert out[-1]["content"].endswith(language_directive("en"))


def test_fallback_does_not_override_a_clearly_detected_language():
    spanish = [{"role": "user", "content": "x"}]
    english = [{"role": "user", "content": "x"}]

    # Spanish text with an English interface stays as it is (no directive).
    assert with_language_directive(spanish, ES_TEXT, fallback="en") is spanish
    # English text with a Spanish interface still gets the English directive.
    out = with_language_directive(english, EN_TEXT, fallback="es")
    assert out[-1]["content"].endswith(language_directive("en"))


def test_spanish_or_missing_fallback_adds_nothing():
    messages = [{"role": "user", "content": "foreign key"}]

    assert with_language_directive(messages, "foreign key", fallback="es") is messages
    assert with_language_directive(messages, "foreign key", fallback=None) is messages
