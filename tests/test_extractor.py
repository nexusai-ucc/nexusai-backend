"""
Tests del extractor de PDFs — la pieza que convierte bytes de PDF en texto plano.

Generamos PDFs in-memory con `reportlab` para no depender de archivos en disco.
Si reportlab no está instalado, los tests se saltean con un mensaje claro
(no es dependencia del producto, solo de tests).
"""

from __future__ import annotations

from io import BytesIO

import pytest

from app.documents.extractor import extract_text


# ============================================================
# Helpers — generar PDFs in-memory
# ============================================================

def _make_pdf_bytes(text_per_page: list[str]) -> bytes:
    """
    Crea un PDF con N páginas, una página por entrada de `text_per_page`.
    Devuelve los bytes serializados.
    """
    reportlab = pytest.importorskip(
        "reportlab",
        reason="reportlab no instalado — pip install reportlab para correr estos tests",
    )
    from reportlab.pdfgen import canvas  # type: ignore[import-not-found]
    from reportlab.lib.pagesizes import A4  # type: ignore[import-not-found]

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for page_text in text_per_page:
        # Escribir el texto a una altura razonable. Si hay líneas largas
        # las cortamos manualmente para que pdfplumber las lea bien.
        y = 800
        for line in page_text.split("\n"):
            c.drawString(50, y, line)
            y -= 20
        c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================
# Happy path
# ============================================================

def test_extract_single_page_pdf():
    pdf_bytes = _make_pdf_bytes(["Una derivada mide la tasa instantánea."])
    text = extract_text(pdf_bytes)
    assert "derivada" in text.lower()
    assert "tasa" in text.lower()


def test_extract_multi_page_pdf_joins_pages():
    pdf_bytes = _make_pdf_bytes([
        "Pagina uno con contenido inicial.",
        "Pagina dos con mas contenido.",
        "Pagina tres ultima.",
    ])
    text = extract_text(pdf_bytes)
    assert "uno" in text.lower()
    assert "dos" in text.lower()
    assert "tres" in text.lower()
    # Las páginas se separan con doble salto de línea.
    assert "\n\n" in text


def test_extract_returns_stripped_text():
    """No queremos whitespace inicial/final en la salida."""
    pdf_bytes = _make_pdf_bytes(["contenido"])
    text = extract_text(pdf_bytes)
    assert text == text.strip()


# ============================================================
# Validaciones de input
# ============================================================

def test_empty_bytes_raises():
    with pytest.raises(ValueError, match="vacío"):
        extract_text(b"")


def test_garbage_bytes_raises():
    """Bytes que no son un PDF válido → ValueError descriptivo."""
    with pytest.raises(ValueError, match=r"PDF|corrupto"):
        extract_text(b"esto no es un PDF, son bytes random")


def test_pdf_with_no_extractable_text_raises():
    """
    Un PDF sin texto (ej. solo imagen escaneada) → ValueError indicando que
    no se admiten escaneados. Para simularlo creamos un PDF con una página
    vacía (sin drawString).
    """
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas  # type: ignore[import-not-found]

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()  # página totalmente vacía
    c.save()
    empty_pdf = buf.getvalue()

    with pytest.raises(ValueError, match=r"texto|escaneados|extraíble"):
        extract_text(empty_pdf)
