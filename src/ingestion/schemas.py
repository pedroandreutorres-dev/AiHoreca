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


class DocumentManifestEntry(DocumentBoundary):
    """Entrada declarativa del manifiesto de ingesta (`boundaries_cache.json`).

    Extiende `DocumentBoundary` con metadatos opcionales de arquitectura de datos
    (rol documental, familia de maquetación y directivas de extracción por página)
    manteniendo 100% de retrocompatibilidad con entradas heredadas.
    """

    doc_role: Literal[
        "estatuto",
        "aleh",
        "completo",
        "base",
        "tablas",
        "modificacion",
        "erratas",
    ] = Field(
        default="completo",
        description=(
            "Rol del documento en el corpus: 'estatuto', 'aleh', 'completo' (texto + tablas vigentes), "
            "'base' (solo articulado sin tablas vigentes), 'tablas' (anexo/tablas salariales), "
            "'modificacion' o 'erratas'."
        ),
    )
    layout: Literal["single_column", "double_column", "scanned_ocr"] = Field(
        default="single_column",
        description="Familia de maquetación del PDF para despachar el adaptador de extracción adecuado.",
    )
    force_gemini_pages: list[int] = Field(
        default_factory=list,
        description="Lista de páginas (1-indexed respecto al PDF) cuya extracción se fuerza con Gemini Vision.",
    )
    skip_pages: list[int] = Field(
        default_factory=list,
        description="Lista de páginas (1-indexed respecto al PDF) dentro de [start_page, end_page] a omitir.",
    )
    expected_years: list[int] = Field(
        default_factory=list,
        description="Años de vigencia esperados en las tablas salariales de este documento (ej. [2025, 2026]).",
    )


ValidationSeverity = Literal["ERROR", "WARNING", "INFO"]


class ValidationIssue(BaseModel):
    """Hallazgo individual emitido por el Gate de Calidad de ingesta."""

    code: str
    severity: ValidationSeverity
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class DocumentValidationReport(BaseModel):
    """Resultado de la validación determinista en CPU para un documento Silver `.md`."""

    md_file: str
    pdf_file: str | None = None
    doc_role: str = "completo"
    passed: bool
    chunks_count: int = 0
    pdf_decimals_count: int | None = None
    md_decimals_count: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "WARNING")


class CorpusValidationReport(BaseModel):
    """Resumen agregado de validación para todo `data/processed/`."""

    strict: bool = False
    passed: bool
    total_documents: int
    total_chunks: int
    total_errors: int
    total_warnings: int
    documents: list[DocumentValidationReport] = Field(default_factory=list)


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
