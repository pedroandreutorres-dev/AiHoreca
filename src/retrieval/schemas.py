"""Esquemas Pydantic v2 para los fragmentos jurídicos (chunks) y resultados de búsqueda híbrida."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

NormativeLayer = Literal[
    "estatuto_trabajadores",
    "aleh_estatal",
    "convenio_provincial",
]

ContentType = Literal[
    "articulo",
    "tabla_salarial",
    "anexo_o_disposicion",
]


class LegalChunk(BaseModel):
    """Representa un fragmento jurídico autocontenido listo para indexación híbrida en Qdrant."""

    chunk_id: str = Field(
        description="UUID determinista generado a partir del documento, jerarquía y orden del fragmento."
    )
    source_file: str = Field(
        description="Nombre del archivo Markdown de origen (ej. 'convenio_madrid.md')."
    )
    norma_titulo: str = Field(
        description="Nombre legible de la norma jurídica."
    )
    capa_normativa: NormativeLayer = Field(
        description="Capa dentro de la jerarquía normativa laboral (1: Estatuto, 2: ALEH, 3: Convenio)."
    )
    territorio: str = Field(
        description="Código normalizado del territorio ('estatal', 'madrid', 'catalunya', 'leon', etc.)."
    )
    provincias_aplicables: list[str] = Field(
        description="Lista de provincias donde aplica la norma (ej. ['barcelona', 'girona', 'tarragona', 'catalunya'])."
    )
    titulo_o_capitulo: str = Field(
        description="Título, Capítulo o Anexo al que pertenece el fragmento."
    )
    articulo: str = Field(
        description="Identificador del artículo, disposición o tabla (ej. 'Artículo 14', 'Tabla Salarial Anexo I')."
    )
    tipo_contenido: ContentType = Field(
        description="Clasificación estructural del contenido ('articulo', 'tabla_salarial', 'anexo_o_disposicion')."
    )
    chunk_index: int = Field(
        description="Índice correlativo del fragmento dentro del documento."
    )
    text: str = Field(
        description="Texto Markdown enriquecido con cabecera de contexto normativo para embedding denso y BM25."
    )


class RetrievedDocument(BaseModel):
    """Resultado devuelto por el buscador híbrido (Qdrant RRF + Cohere Rerank opcional)."""

    chunk_id: str
    score: float
    rerank_score: Optional[float] = None
    capa_normativa: NormativeLayer
    territorio: str
    norma_titulo: str
    titulo_o_capitulo: str
    articulo: str
    tipo_contenido: ContentType
    source_file: str
    text: str
