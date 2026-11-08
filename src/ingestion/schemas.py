"""Esquemas Pydantic v2 para la ingesta híbrida y corrección automática de erratas."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class DocumentBoundary(BaseModel):
    """Metadatos y límites exactos de páginas de un documento normativo dentro de un Boletín Oficial."""

    doc_type: Literal[
        "estatuto_trabajadores",
        "aleh_estatal",
        "convenio_hosteleria",
        "correccion_erratas",
    ] = Field(
        description=(
            "Tipo de documento normativo: 'estatuto_trabajadores' para el Estatuto de los Trabajadores, "
            "'aleh_estatal' para el Acuerdo Laboral Estatal de Hostelería (ALEH), "
            "'convenio_hosteleria' para convenios colectivos provinciales o autonómicos de hostelería, "
            "o 'correccion_erratas' para resoluciones de corrección de erratas."
        )
    )
    territorio: str = Field(
        description=(
            "Identificador territorial en minúsculas y sin tildes: 'estatal' para normas de ámbito nacional "
            "(Estatuto o ALEH), o el nombre de la provincia/comunidad autónoma en minúsculas sin tildes "
            "(ej. 'madrid', 'catalunya', 'leon', 'sevilla', 'valencia')."
        )
    )
    start_page: int = Field(
        ge=1,
        description=(
            "Número de página (1-indexed) donde comienza realmente el texto articulado o título directo "
            "del Convenio Colectivo de Hostelería (o norma), DESPUÉS de haber saltado sumarios del boletín, "
            "anuncios de otras administraciones (Ayuntamientos, Juntas Vecinales, etc.) e índices de artículos."
        ),
    )
    end_page: int = Field(
        ge=1,
        description=(
            "Número de página (1-indexed, inclusivo) donde termina el Convenio Colectivo de Hostelería "
            "(incluyendo sus Anexos y Tablas Salariales), ANTES de cualquier anuncio posterior de otras "
            "administraciones o ministerios."
        ),
    )
    justificacion: str = Field(
        description="Breve explicación de qué empieza en start_page y qué termina en end_page."
    )


class ErrataPatch(BaseModel):
    """Representa una sustitución individual indicada en una corrección de erratas oficial."""

    contexto_articulo: str = Field(
        description="Artículo, anexo o sección donde se ubica la errata (ej. 'Anexo I, Artículo 6')."
    )
    donde_dice: str = Field(
        description="Cadena literal exacta con error tal como apareció publicada originalmente."
    )
    debe_decir: str = Field(
        description="Cadena literal exacta corregida por la que debe sustituirse."
    )


class ErrataPatchList(BaseModel):
    """Lista estructurada de correcciones de erratas extraídas de un boletín oficial."""

    territorio: str = Field(
        description="Territorio del convenio al que aplica la corrección de erratas (ej. 'catalunya')."
    )
    parches: list[ErrataPatch] = Field(
        description="Lista de sustituciones literales (donde dice -> debe decir) que deben aplicarse."
    )
