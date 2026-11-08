"""Chunker jurídico jerárquico y agnóstico a la provincia para documentos Markdown de AiHoreca.

Principios de diseño:
1. Unidad lógica: fragmenta por Artículo, Disposición o Tabla Salarial (nunca corta una fila de tabla a la mitad).
2. Inyección de contexto: cada chunk comienza con una cabecera estructurada
   `[NORMA: ... | CAPA: ... | TERRITORIO: ... | SECCIÓN: ... | ARTÍCULO: ...]`
   para maximizar la precisión tanto en embeddings densos (`gemini-embedding-001`) como en BM25 (`FastEmbed`).
3. Preservación de cabeceras de tabla: si una tabla salarial muy extensa debe dividirse en varios sub-chunks,
   repite automáticamente la fila de cabeceras (`|...|`) y el separador (`|---|---|`) en cada sub-chunk.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from src.config import PROCESSED_DATA_DIR
from src.retrieval.schemas import ContentType, LegalChunk, NormativeLayer

logger = logging.getLogger(__name__)

_RESIDUAL_NOISE_LINE = re.compile(
    r"(?i)^(?:"
    r"dl\s+[a-z]{1,2}\s+\d+-\d+|"
    r"issn\s+\d{4}-\d{3}[\dx]|"
    r"\d{1,3}/\d{1,3}|"
    r"legislaci[óo]n\s+consolidada|"
    r"p[áa]gina\s+\d+|"
    r"verificable\s+en\s+https?://\S+|"
    r"sec\.\s+[ivx]+\.\s+p[áa]g\.\s*\d+|"
    r"n[úu]m(?:ero)?\.?\s*\d+\s*(?:•|-)?\s*p[áa]gina\s*\d+.*|"
    r"boletin@dipuleon\.es|"
    r"bocm-\d+-\d+|"
    r"bocm|"
    r"#*\s*b\s*<mark>o</mark>\s*\*\*<mark>cm</mark>\*\*|"
    r"(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo),?\s+\d+\s+de\s+\w+\s+de\s+\d{4}|"
    r"===\s*document\s+parser\s+messages\s*===|"
    r"warning:\s*rapidocr\s+is\s+missing.*"
    r")$"
)

_CHAPTER_OR_TITLE_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?((?:T[ÍI]TULO|CAP[ÍI]TULO|ANEXO|PROTOCOLO)\s+[IVXLCDM0-9]+[^\n]*)$"
)

_TABLE_YEAR_OR_PROVINCE_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?\**(?:TABLAS?\s+SALARIALES?\s+(?:A[ÑN]O\s+|PARA\s+EL\s+A[ÑN]O\s+)?2\.?02[3-9][^\n]*|[A-D]\)\s*(?:Barcelona|Tarragona|Girona|Lleida)[^\n]*)\**$"
)

_ARTICLE_START_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?((?:Art[íi]culo|Art\.)\s*\d+(?:\s*(?:º|\.º|bis|ter|quater))?(?:\s*$|(?:\.|\s*[-–—])\s*(?!(?:de\s+la\s+Ley|del\s+Estatuto|del\s+Real\s+Decreto))[^\n]*))$"
)

_DISPOSITION_START_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?(Disposici[óo]n\s+(?:adicional|transitoria|derogatoria|final)\s+[^\n]+)$"
)

_MARKDOWN_SUBHEADING_RE = re.compile(r"^#{2,4}\s+(.+)$")


def _infer_document_metadata(
    md_path: Path,
    boundaries_cache: dict[str, dict],
) -> tuple[str, NormativeLayer, str, list[str]]:
    """Infiere título, capa normativa, territorio y provincias aplicables de forma agnóstica."""
    pdf_name = f"{md_path.stem}.pdf"
    cached = boundaries_cache.get(pdf_name, {})
    doc_type = cached.get("doc_type", "")
    territorio = (cached.get("territorio") or "").lower().strip()

    stem_lower = md_path.stem.lower()
    if not doc_type:
        if "estatuto" in stem_lower:
            doc_type = "estatuto_trabajadores"
            territorio = "estatal"
        elif "aleh" in stem_lower:
            doc_type = "aleh_estatal"
            territorio = "estatal"
        else:
            doc_type = "convenio_hosteleria"
            territorio = stem_lower.replace("convenio_", "").replace("_texto", "")

    if doc_type == "estatuto_trabajadores":
        return (
            "Estatuto de los Trabajadores (RDL 2/2015)",
            "estatuto_trabajadores",
            "estatal",
            ["estatal"],
        )
    if doc_type == "aleh_estatal":
        return (
            "VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)",
            "aleh_estatal",
            "estatal",
            ["estatal"],
        )

    # Convenio provincial o autonómico multiprovincial
    if territorio in {"catalunya", "cataluna", "barcelona", "catalunya_bcn"}:
        return (
            "Convenio Colectivo Interprovincial de Hostelería y Turismo de Cataluña",
            "convenio_provincial",
            "catalunya",
            ["barcelona", "girona", "tarragona", "lleida", "catalunya"],
        )
    if territorio == "madrid":
        return (
            "Convenio Colectivo de Hostelería y Actividades Turísticas de la Comunidad de Madrid",
            "convenio_provincial",
            "madrid",
            ["madrid"],
        )
    if territorio == "leon":
        return (
            "Convenio Colectivo Provincial de Hostelería y Turismo de León",
            "convenio_provincial",
            "leon",
            ["leon"],
        )

    pretty_terr = territorio.replace("_", " ").title()
    return (
        f"Convenio Colectivo de Hostelería de {pretty_terr}",
        "convenio_provincial",
        territorio,
        [territorio],
    )


def _clean_markdown_lines(raw_text: str) -> list[str]:
    """Limpia líneas residuales de numeración o cabeceras de boletín oficial y normaliza espacios Unicode."""
    normalized = (
        raw_text.replace("\u2003", " ")
        .replace("\u2002", " ")
        .replace("\u00a0", " ")
    )
    cleaned: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if _RESIDUAL_NOISE_LINE.match(stripped):
            continue
        cleaned.append(line)
    return cleaned


def _split_large_block(
    body_lines: list[str],
    max_chars: int = 2600,
    overlap_lines: int = 2,
) -> list[str]:
    """Divide un bloque extenso respetando tablas Markdown (repitiendo su cabecera) y párrafos."""
    full_text = "\n".join(body_lines).strip()
    if len(full_text) <= max_chars:
        return [full_text] if full_text else []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    active_table_header: list[str] = []

    for line in body_lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if len(active_table_header) < 2:
                active_table_header.append(line)
        else:
            if stripped:
                active_table_header = []

        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines).strip())
            if is_table_line and len(active_table_header) == 2:
                current_lines = list(active_table_header)
                current_len = sum(len(h) + 1 for h in current_lines)
            else:
                tail = current_lines[-overlap_lines:] if overlap_lines > 0 else []
                current_lines = list(tail)
                current_len = sum(len(t) + 1 for t in current_lines)

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return [c for c in chunks if c]


def chunk_markdown_document(
    md_path: Path,
    boundaries_cache: dict[str, dict] | None = None,
    max_chunk_chars: int = 2600,
) -> list[LegalChunk]:
    """Fragmenta un documento Markdown normativo en objetos LegalChunk jerárquicos."""
    cache = boundaries_cache or {}
    norma_titulo, capa_normativa, territorio, provincias = _infer_document_metadata(
        md_path, cache
    )

    raw_text = md_path.read_text(encoding="utf-8")
    lines = _clean_markdown_lines(raw_text)

    current_base_chapter = "Disposiciones Generales / Preámbulo"
    current_chapter = current_base_chapter
    current_article = "Preámbulo / Encabezado"
    current_buffer: list[str] = []

    raw_sections: list[tuple[str, str, list[str]]] = []

    def flush_section() -> None:
        nonlocal current_buffer
        non_empty = [l for l in current_buffer if l.strip()]
        if non_empty:
            raw_sections.append((current_chapter, current_article, list(current_buffer)))
        current_buffer = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        chap_match = _CHAPTER_OR_TITLE_RE.match(stripped)
        if chap_match:
            flush_section()
            chap_title = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            if idx + 1 < len(lines):
                next_str = lines[idx + 1].strip()
                if (
                    next_str
                    and len(next_str) < 90
                    and not next_str.startswith("#")
                    and not next_str.startswith("|")
                    and not _ARTICLE_START_RE.match(next_str)
                    and not _CHAPTER_OR_TITLE_RE.match(next_str)
                ):
                    chap_title = f"{chap_title} - {next_str}"
                    idx += 1
            current_base_chapter = chap_title
            current_chapter = chap_title
            current_article = chap_title
            idx += 1
            continue

        if _TABLE_YEAR_OR_PROVINCE_RE.match(stripped):
            flush_section()
            sub_ctx = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            current_chapter = f"{current_base_chapter} | {sub_ctx}"
            current_article = sub_ctx
            current_buffer.append(f"## {sub_ctx}")
            idx += 1
            continue

        art_match = _ARTICLE_START_RE.match(stripped)
        if art_match:
            flush_section()
            art_title = re.sub(r"^#{1,4}\s*", "", stripped).strip()
            if re.fullmatch(r"(?i)(?:Art[íi]culo|Art\.)\s*\d+(?:\.º|º)?\.?", art_title):
                lookahead = idx + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead < len(lines):
                    rubrica = lines[lookahead].strip()
                    if (
                        len(rubrica) < 100
                        and not rubrica.startswith("#")
                        and not rubrica.startswith("|")
                    ):
                        art_title = f"{art_title}. {rubrica}"
                        idx = lookahead
            current_article = art_title
            current_buffer.append(f"### {art_title}")
            idx += 1
            continue

        disp_match = _DISPOSITION_START_RE.match(stripped)
        if disp_match:
            flush_section()
            disp_title = re.sub(r"^#{1,4}\s*", "", stripped).strip()
            current_article = disp_title
            current_buffer.append(f"### {disp_title}")
            idx += 1
            continue

        sub_match = _MARKDOWN_SUBHEADING_RE.match(stripped)
        if sub_match and ("anexo" in current_base_chapter.lower() or "tabla" in current_chapter.lower()):
            sub_title = sub_match.group(1).strip("* ")
            # Si el buffer actual ya tiene suficiente contenido (>600 chars), abrimos sub-sección nueva
            if sum(len(x) for x in current_buffer) >= 600:
                flush_section()
                current_article = sub_title
            elif current_article == current_base_chapter:
                current_article = sub_title
            current_buffer.append(line)
            idx += 1
            continue

        current_buffer.append(line)
        idx += 1

    flush_section()

    legal_chunks: list[LegalChunk] = []
    chunk_counter = 0

    for chapter, article, sec_lines in raw_sections:
        sub_blocks = _split_large_block(sec_lines, max_chars=max_chunk_chars)
        total_parts = len(sub_blocks)

        for part_idx, sub_text in enumerate(sub_blocks, start=1):
            pipe_count = sum(
                1 for l in sub_text.splitlines() if l.strip().startswith("|")
            )
            if pipe_count >= 3:
                tipo_contenido: ContentType = "tabla_salarial"
            elif "anexo" in chapter.lower() or "disposición" in article.lower():
                tipo_contenido = "anexo_o_disposicion"
            else:
                tipo_contenido = "articulo"

            part_suffix = f" (Parte {part_idx}/{total_parts})" if total_parts > 1 else ""
            art_label = f"{article}{part_suffix}"

            context_header = (
                f"[NORMA: {norma_titulo} | CAPA: {capa_normativa} | "
                f"TERRITORIO: {territorio.upper()} | SECCIÓN: {chapter} | "
                f"REFERENCIA: {art_label}]"
            )
            enriched_text = f"{context_header}\n\n{sub_text}"

            deterministic_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{md_path.name}::{chapter}::{art_label}::{chunk_counter}",
                )
            )

            legal_chunks.append(
                LegalChunk(
                    chunk_id=deterministic_id,
                    source_file=md_path.name,
                    norma_titulo=norma_titulo,
                    capa_normativa=capa_normativa,
                    territorio=territorio,
                    provincias_aplicables=provincias,
                    titulo_o_capitulo=chapter,
                    articulo=art_label,
                    tipo_contenido=tipo_contenido,
                    chunk_index=chunk_counter,
                    text=enriched_text,
                )
            )
            chunk_counter += 1

    logger.info(
        "Documento %s fragmentado en %d chunks jurídicos (tablas=%d, artículos=%d, anexos=%d).",
        md_path.name,
        len(legal_chunks),
        sum(1 for c in legal_chunks if c.tipo_contenido == "tabla_salarial"),
        sum(1 for c in legal_chunks if c.tipo_contenido == "articulo"),
        sum(1 for c in legal_chunks if c.tipo_contenido == "anexo_o_disposicion"),
    )
    return legal_chunks


def chunk_all_processed_documents(
    processed_dir: Path = PROCESSED_DATA_DIR,
    max_chunk_chars: int = 2600,
) -> list[LegalChunk]:
    """Fragmenta todos los archivos .md en data/processed/ devolviendo la lista consolidada de LegalChunk."""
    cache_path = processed_dir / "boundaries_cache.json"
    boundaries_cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            boundaries_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            boundaries_cache = {}

    all_chunks: list[LegalChunk] = []
    for md_path in sorted(processed_dir.glob("*.md")):
        doc_chunks = chunk_markdown_document(
            md_path,
            boundaries_cache=boundaries_cache,
            max_chunk_chars=max_chunk_chars,
        )
        all_chunks.extend(doc_chunks)

    logger.info("Total de chunks generados en el corpus: %d", len(all_chunks))
    return all_chunks
