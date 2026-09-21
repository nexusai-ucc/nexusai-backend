"""Tests for app.shared.language — detection and output-language directive."""

import pytest

from app.shared.language import (
    detect_language,
    language_directive,
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
