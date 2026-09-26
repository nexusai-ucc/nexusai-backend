"""
Visibilidad del material (VIS-01, issue #536).

El backend no sabe qué actividades de Moodle puede ver cada usuario: eso lo
decide Moodle (ojito de la actividad, sección oculta, restricciones de fecha,
grupo o condiciones) y es distinto para cada persona. Por eso el plugin
calcula, en cada pedido, la lista de `cmid` (ids de actividad) visibles para
ese usuario y la manda en el cuerpo firmado como `visible_cmids`. Cada
documento guarda el `cmid` de la actividad de la que salió (`Document.cmid`).

Reglas:

- Un documento entra en una respuesta solo si su `cmid` está en la lista. Un
  documento sin `cmid` (que no salió de una actividad) no entra en ninguna.
- Lista vacía: no hay nada visible.
- Pedido sin la lista: se rechaza con 422 si `REQUIRE_VISIBLE_CMIDS` está
  encendido (por defecto). Con el ajuste apagado devuelve None y no se filtra
  nada; existe solo para un despliegue escalonado.
- Las rutas del docente que administran material (lista, detalle, vista
  previa, estadísticas, examen sobre documentos elegidos) no filtran.

La lista viaja firmada con HMAC y sale del servidor de Moodle, nunca del
navegador: el backend puede confiar en ella igual que confía en `user_id`.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import HTTPException, status
from pydantic import Field

from app.shared.config import get_settings

# Tope defensivo: un curso con más de 5.000 actividades no es realista.
VisibleCmids = Annotated[
    Optional[list[Annotated[int, Field(gt=0)]]], Field(default=None, max_length=5000)
]


def enforce_visible_cmids(visible_cmids: Optional[list[int]]) -> Optional[list[int]]:
    """Devuelve la lista tal cual, o None si no vino y la exigencia está apagada.

    Raises:
        HTTPException 422: si no vino y `REQUIRE_VISIBLE_CMIDS` está encendido.
    """
    if visible_cmids is None and get_settings().require_visible_cmids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Falta visible_cmids: la lista de actividades del curso que el "
                "usuario puede ver."
            ),
        )
    return visible_cmids
