"""
Casos de prueba de caja negra del chat (TPC-04, #576 · HU-001).

Cada test implementa un caso de `docs/calidad/CASOS_DE_PRUEBA.md` §2 y lleva
su ID (CP-CHAT-xx) como id del parametrize. Técnicas:
  - Valores límite (VL): n−1, n y n+1 de cada restricción de `ChatRequest`.
  - Clases de equivalencia (CE): un representante por clase válida e inválida.

Se prueba el contrato de entrada real (`app/chat/schemas.py`): es lo que
FastAPI aplica antes de llegar al router, y lo que convierte en 422.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.chat.schemas import ChatRequest

_VALID = {"question": "¿Qué es una pila?", "course_id": 2, "user_id": 5}


def _history(n: int, content: str = "hola", role: str = "user") -> list[dict]:
    return [{"role": role, "content": content} for _ in range(n)]


# ============================================================
# Valores límite
# ============================================================


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("question", "a", id="CP-CHAT-02 question 1 caracter"),
        pytest.param("question", "a" * 2000, id="CP-CHAT-03 question 2000"),
        pytest.param("course_id", 1, id="CP-CHAT-06 course_id 1"),
        pytest.param("user_id", 1, id="CP-CHAT-08 user_id 1"),
        pytest.param("history", _history(10), id="CP-CHAT-11 history 10 items"),
        pytest.param("history", _history(1, "a"), id="CP-CHAT-13 content 1"),
        pytest.param(
            "history", _history(1, "a" * 20000), id="CP-CHAT-14 content 20000"
        ),
        pytest.param("token_limit_hourly", 1, id="CP-CHAT-17 limite horario 1"),
        pytest.param(
            "token_limit_hourly", 10_000_000, id="CP-CHAT-18 limite horario 10M"
        ),
    ],
)
def test_limite_valido_se_acepta(field, value):
    """Criterio: el valor en el borde del rango válido se acepta."""
    # Arrange.
    payload = {**_VALID, field: value}

    # Act.
    request = ChatRequest.model_validate(payload)

    # Assert.
    assert getattr(request, field) is not None


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("question", "", id="CP-CHAT-01 question vacia"),
        pytest.param("question", "a" * 2001, id="CP-CHAT-04 question 2001"),
        pytest.param("course_id", 0, id="CP-CHAT-05 course_id 0"),
        pytest.param("user_id", 0, id="CP-CHAT-07 user_id 0"),
        pytest.param("history", _history(11), id="CP-CHAT-12 history 11 items"),
        pytest.param("history", _history(1, ""), id="CP-CHAT-15 content vacio"),
        pytest.param(
            "history", _history(1, "a" * 20001), id="CP-CHAT-16 content 20001"
        ),
        pytest.param("token_limit_hourly", 0, id="CP-CHAT-19 limite horario 0"),
        pytest.param(
            "token_limit_hourly", 10_000_001, id="CP-CHAT-20 limite horario 10M+1"
        ),
    ],
)
def test_limite_invalido_se_rechaza(field, value):
    """Criterio: un paso afuera del rango se rechaza (FastAPI responde 422)."""
    # Arrange.
    payload = {**_VALID, field: value}

    # Act / Assert.
    with pytest.raises(ValidationError) as exc:
        ChatRequest.model_validate(payload)
    assert exc.value.errors()[0]["loc"][0] == field


# ============================================================
# Clases de equivalencia
# ============================================================


def test_cp_chat_09_sin_history_es_conversacion_en_el_backend():
    """CP-CHAT-09 (CE válida): sin `history`, la conversación la guarda el backend."""
    # Act.
    request = ChatRequest.model_validate(_VALID)

    # Assert.
    assert request.stateless is False


def test_cp_chat_10_history_vacia_es_conversacion_en_moodle():
    """CP-CHAT-10 (CE válida): `history=[]` ya es modo stateless (DATA-04),
    distinto de no mandarla."""
    # Act.
    request = ChatRequest.model_validate({**_VALID, "history": []})

    # Assert.
    assert request.stateless is True


@pytest.mark.parametrize(
    "role, accepted",
    [
        pytest.param("user", True, id="CP-CHAT-21 rol user"),
        pytest.param("assistant", True, id="CP-CHAT-22 rol assistant"),
        pytest.param("system", False, id="CP-CHAT-23 rol system"),
    ],
)
def test_rol_del_historial(role, accepted):
    """Criterio: el historial solo admite turnos de alumno y asistente. Un
    turno `system` inyectado desde el cliente se rechaza (no puede pisar el
    prompt del sistema)."""
    # Arrange.
    payload = {**_VALID, "history": _history(1, role=role)}

    # Act / Assert.
    if accepted:
        assert ChatRequest.model_validate(payload).history[0].role == role
    else:
        with pytest.raises(ValidationError):
            ChatRequest.model_validate(payload)
