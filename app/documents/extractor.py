"""Extracción de texto de documentos: PDF, DOCX y TXT — CONT-03.

Limitaciones:
  - PDF: solo texto seleccionable. PDFs escaneados (solo imagen) lanzan ValueError.
  - DOCX: extrae párrafos y celdas de tablas. No extrae headers/footers ni imágenes.
  - TXT: UTF-8 con fallback a latin-1.
"""

from __future__ import annotations

from io import BytesIO

import pdfplumber


_MIME_PDF = "application/pdf"
_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIME_TXT = "text/plain"

SUPPORTED_MIME_TYPES = {_MIME_PDF, _MIME_DOCX, _MIME_TXT}


def extract_text(file_bytes: bytes, mime_type: str = _MIME_PDF) -> str:
    """Extrae texto de un documento según su mime_type.

    Args:
        file_bytes: Contenido binario del archivo.
        mime_type: MIME type del archivo (application/pdf, application/vnd..docx, text/plain).

    Returns:
        Texto extraído como string, listo para el pipeline de chunking.

    Raises:
        ValueError: si el archivo está vacío, corrupto, no tiene texto extraíble,
                    o el mime_type no está soportado.
    """
    if not file_bytes:
        raise ValueError("El archivo está vacío o no contiene datos válidos.")

    if mime_type == _MIME_PDF:
        return _extract_pdf(file_bytes)
    elif mime_type == _MIME_DOCX:
        return _extract_docx(file_bytes)
    elif mime_type == _MIME_TXT:
        return _extract_txt(file_bytes)
    else:
        raise ValueError(
            f"Tipo de archivo no soportado: {mime_type!r}. "
            f"Tipos aceptados: {sorted(SUPPORTED_MIME_TYPES)}"
        )


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())
    except Exception as exc:
        raise ValueError(
            "No se pudo leer el PDF. El archivo puede estar corrupto o no ser un PDF válido."
        ) from exc

    if not pages_text:
        raise ValueError(
            "El PDF no tiene texto extraíble. Solo se admiten PDFs con texto, no escaneados."
        )

    return "\n\n".join(pages_text).strip()


def _extract_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document  # python-docx — importación lazy para no afectar startup
        doc = Document(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(
            "No se pudo leer el DOCX. El archivo puede estar corrupto o no ser un DOCX válido."
        ) from exc

    parts: list[str] = []

    # Párrafos del cuerpo principal
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    # Celdas de todas las tablas
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text:
                    parts.append(text)

    if not parts:
        raise ValueError(
            "El DOCX no tiene texto extraíble. Verificá que el archivo tenga contenido de texto."
        )

    return "\n\n".join(parts).strip()


def _extract_txt(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "El archivo de texto no pudo decodificarse en UTF-8 ni latin-1."
    )
