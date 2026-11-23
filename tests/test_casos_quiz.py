"""
Casos de prueba de caja negra de la generación de quiz (TPC-04, #576 · HU-030).

Cada test implementa un caso de `docs/calidad/CASOS_DE_PRUEBA.md` §4 y lleva
su ID (CP-QUIZ-xx). Técnicas: valores límite (VL) sobre `num_questions` y
`topic`, y clases de equivalencia (CE) sobre `difficulty` y `question_type`,
contra el modelo real `QuizRequest` (`app/quiz/router.py`).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.quiz.router import QuizRequest

_VALID = {"course_id": 2, "user_id": 5}


@pytest.mark.parametrize(
    "field, value, accepted",
    [
        pytest.param("num_questions", 0, False, id="CP-QUIZ-01 num_questions 0"),
        pytest.param("num_questions", 1, True, id="CP-QUIZ-02 num_questions 1"),
        pytest.param("num_questions", 10, True, id="CP-QUIZ-03 num_questions 10"),
        pytest.param("num_questions", 11, False, id="CP-QUIZ-04 num_questions 11"),
        pytest.param("topic", "t" * 200, True, id="CP-QUIZ-05 topic 200"),
        pytest.param("topic", "t" * 201, False, id="CP-QUIZ-06 topic 201"),
    ],
)
def test_valores_limite(field, value, accepted):
    """Criterio: el borde del rango se acepta y un paso afuera se rechaza."""
    # Arrange.
    payload = {**_VALID, field: value}

    # Act / Assert.
    if accepted:
        assert getattr(QuizRequest.model_validate(payload), field) == value
    else:
        with pytest.raises(ValidationError):
            QuizRequest.model_validate(payload)


def test_cp_quiz_07_valores_por_defecto():
    """CP-QUIZ-07 (CE válida): sin parámetros opcionales, 5 preguntas de
    opción múltiple, dificultad media y sin tema (todo el material)."""
    # Act.
    request = QuizRequest.model_validate(_VALID)

    # Assert.
    assert (
        request.num_questions,
        request.question_type,
        request.difficulty,
        request.topic,
    ) == (5, "multiple_choice", "medium", None)


@pytest.mark.parametrize(
    "question_type",
    [
        pytest.param(t, id=f"CP-QUIZ-08 {t}")
        for t in (
            "multiple_choice",
            "true_false",
            "open",
            "mix",
            "flashcard",
            "fill_blank",
        )
    ],
)
def test_tipos_de_pregunta_validos(question_type):
    """CP-QUIZ-08 (CE válida): cada tipo admitido es su propia clase."""
    # Act.
    request = QuizRequest.model_validate({**_VALID, "question_type": question_type})

    # Assert.
    assert request.question_type == question_type


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("question_type", "essay", id="CP-QUIZ-09 tipo inexistente"),
        pytest.param(
            "question_type", "Multiple_Choice", id="CP-QUIZ-10 tipo con mayusculas"
        ),
        pytest.param("difficulty", "", id="CP-QUIZ-11 dificultad vacia"),
        pytest.param("difficulty", "HARD", id="CP-QUIZ-12 dificultad con mayusculas"),
    ],
)
def test_clases_invalidas(field, value):
    """Criterio (CE inválida): los valores fuera del conjunto se rechazan.
    La comparación es exacta: las mayúsculas son otra clase."""
    # Act / Assert.
    with pytest.raises(ValidationError):
        QuizRequest.model_validate({**_VALID, field: value})
