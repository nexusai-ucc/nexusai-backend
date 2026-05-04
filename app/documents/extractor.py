"""PDF text extraction helpers.

This extractor only works for PDFs that already contain selectable text. It does
not perform OCR, so scanned/image-only PDFs will raise a ValueError.
"""

from __future__ import annotations

from io import BytesIO

import pdfplumber


def extract_text(file_bytes: bytes) -> str:
    """Extract clean text from a PDF byte payload.

    The function expects a text-based PDF. If the file is empty, corrupt, or
    contains no extractable text, it raises ValueError with a descriptive message.
    """
    if not file_bytes:
        raise ValueError("El PDF está vacío o no contiene datos válidos.")

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())
    except Exception as exc:
        raise ValueError("No se pudo leer el PDF. El archivo puede estar corrupto o no ser un PDF válido.") from exc

    if not pages_text:
        raise ValueError("El PDF no tiene texto extraíble. Solo se admiten PDFs con texto, no escaneados.")

    return "\n\n".join(pages_text).strip()