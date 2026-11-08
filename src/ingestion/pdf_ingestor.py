"""Motor de ingesta híbrida automatizada (PyMuPDF en CPU + Google GenAI Multimodal).

Escalable a las 50 provincias sin reglas ni expresiones regulares hardcodeadas por territorio:
1. Detección automática de límites [start_page, end_page] y clasificación con PyMuPDF + Gemini + Pydantic.
2. Extracción híbrida: texto articulado directo en CPU con PyMuPDF y páginas con tablas vía PDF nativo en Gemini.
3. Auto-Errata Patcher: extracción estructurada de parches {donde_dice, debe_decir} con Pydantic y aplicación automática.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pymupdf
import pymupdf4llm
from google import genai
from google.genai import types

from src.config import (
    GEMINI_FALLBACK_MODELS,
    GEMINI_FAST_MODEL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    get_genai_client,
)
from src.ingestion.schemas import DocumentBoundary, ErrataPatchList

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_BULLETIN_HEADER_NOISE = re.compile(
    r"(?i)^(?:"
    r"bolet[íi]n\s+oficial\s+de|"
    r"b\.?\s*o\.?\s*c\.?\s*m\.?(?:\s+n[úu]m)?|"
    r"bocm-\d+|"
    r"diari\s+oficial\s+de\s+la\s+generalitat|"
    r"cve-dogc-|cve:\s*boe-|issn\s+\d|"
    r"https?://\S+|www\.\S+|e-mail:\s*\S+|tel[ée]fono:\s*\d|"
    r"p[áa]g(?:ina)?\.?\s*\d+(?:\s*de\s*\d+)?|"
    r"n[úu]m(?:ero)?\.?\s*\d+\s*[-•]?\s*p[áa]gina\s*\d+|"
    r"(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo),?\s+\d+\s+(?:de\s+)?\w+\s+de\s+\d{4}"
    r").*$"
)


_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


def _generate_with_fallback(
    client: genai.Client,
    contents: object,
    config: types.GenerateContentConfig,
    models: tuple[str, ...] = GEMINI_FALLBACK_MODELS,
) -> types.GenerateContentResponse:
    """Ejecuta generate_content con fallback adaptativo y cooldown ante picos 503/429 del Free Tier."""
    now = time.monotonic()
    ordered_models = sorted(
        models,
        key=lambda m: 1 if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) > now else 0,
    )
    last_err: Optional[Exception] = None
    for model_name in ordered_models:
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_err = exc
            _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + 90.0
            logger.warning(
                "Aviso: modelo %s devolvió error (%s). Activando cooldown de 90s y probando siguiente modelo...",
                model_name,
                exc,
            )
    raise RuntimeError(f"Todos los modelos de fallback fallaron: {last_err}") from last_err


def build_pdf_page_index(pdf_path: Path) -> tuple[int, str]:
    """Construye en CPU (<0.1s) un resumen ligero de cabeceras y pies de cada página del PDF."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    summary_lines: list[str] = []

    for idx in range(total_pages):
        page_num = idx + 1
        page = doc[idx]
        raw_text = page.get_text("text")
        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
            and not _BULLETIN_HEADER_NOISE.match(line.strip())
            and not re.fullmatch(r"\d{1,5}", line.strip())
        ]
        head = " | ".join(lines[:5])[:260]
        tail = " | ".join(lines[-2:])[:140] if len(lines) > 5 else ""
        summary_lines.append(
            f"PÁG {page_num}/{total_pages}: [INICIO] {head}"
            + (f" ... [FINAL] {tail}" if tail else "")
        )

    doc.close()
    return total_pages, "\n".join(summary_lines)


def detect_document_boundaries(
    pdf_path: Path,
    client: Optional[genai.Client] = None,
) -> DocumentBoundary:
    """Detecta automáticamente el tipo de documento, territorio y páginas [start_page, end_page]."""
    genai_client = client or get_genai_client()
    total_pages, page_index_text = build_pdf_page_index(pdf_path)

    prompt = (
        f"Analiza el siguiente índice por páginas del archivo PDF '{pdf_path.name}' "
        f"(total: {total_pages} páginas).\n\n"
        "REGLAS CRÍTICAS DE DELIMITACIÓN:\n"
        "1. Si el PDF es un Boletín Oficial que contiene anuncios de otras administraciones "
        "(Ayuntamientos, Juntas Vecinales, Carreteras, Confederaciones Hidrográficas, sumarios del boletín) "
        "además del Convenio Colectivo de Hostelería, debes fijar 'start_page' en la página exacta donde "
        "empieza el título y Artículo 1 del Convenio Colectivo de Hostelería (saltándote también la página "
        "de resolución administrativa previa y el índice/sumario de capítulos si existen antes del articulado).\n"
        "2. Fija 'end_page' en la última página que pertenezca al Convenio Colectivo de Hostelería "
        "(incluyendo todos sus Anexos y Tablas Salariales), deteniéndote ANTES de cualquier anuncio "
        "posterior de otros organismos públicos.\n"
        "3. En cualquier documento (incluido el Estatuto de los Trabajadores o convenios aislados), salta "
        "las páginas iniciales de resolución administrativa de registro, preámbulos de Real Decreto e "
        "índices/sumarios de contenidos con puntos ('...'), fijando 'start_page' en la página donde "
        "comienza el Título I, Capítulo I o Artículo 1 (salvo en correcciones de erratas o acuerdos "
        "modificativos que empiecen directamente en la página 1).\n\n"
        f"ÍNDICE DE PÁGINAS:\n{page_index_text}"
    )

    response = _generate_with_fallback(
        client=genai_client,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=DocumentBoundary,
        ),
    )

    boundary = DocumentBoundary.model_validate_json(response.text)
    boundary.start_page = max(1, min(boundary.start_page, total_pages))
    boundary.end_page = max(boundary.start_page, min(boundary.end_page, total_pages))
    return boundary


def _format_plain_page_to_markdown(page_text: str) -> str:
    """Formatea en CPU una página de texto articulado eliminando ruido genérico de boletín oficial."""
    lines = page_text.splitlines()
    cleaned_lines: list[str] = []

    bulletin_noise = re.compile(
        r"(?i)^(?:"
        r"bolet[íi]n\s+oficial\s+de|"
        r"b\.?\s*o\.?\s*c\.?\s*m\.?\s*n[úu]m|"
        r"diari\s+oficial\s+de\s+la\s+generalitat|"
        r"cve-dogc-|cve:\s*boe-| ISSN\s+\d|"
        r"https?://\S+|www\.\S+|"
        r"p[áa]g\.?\s*\d+\s*(?:de\s*\d+)?|"
        r"n[úu]m\.?\s*\d+\s+.*\d{4}"
        r").*$"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if bulletin_noise.match(stripped):
            continue
        if re.fullmatch(r"\d{1,5}", stripped):
            continue

        # Formatear encabezados estructurales jurídicos
        if re.match(r"(?i)^(?:CAP[ÍI]TULO|T[ÍI]TULO|ANEXO)\s+[IVXLCDM0-9]+", stripped):
            cleaned_lines.append(f"\n## {stripped}\n")
        elif re.match(r"(?i)^Art[íi]culo\s+\d+", stripped):
            cleaned_lines.append(f"\n### {stripped}\n")
        else:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def _extract_table_pages_with_gemini(
    doc: pymupdf.Document,
    page_indices: list[int],
    client: genai.Client,
) -> str:
    """Recorta en memoria un conjunto de páginas con tablas y las convierte a Markdown vía Gemini."""
    sub_doc = pymupdf.open()
    for p_idx in page_indices:
        sub_doc.insert_pdf(doc, from_page=p_idx, to_page=p_idx)
    pdf_bytes = sub_doc.tobytes()
    sub_doc.close()

    prompt = (
        "Extrae íntegramente el contenido de estas páginas pertenecientes a un Convenio Colectivo "
        "de Hostelería en formato GitHub-Flavored Markdown.\n"
        "REGLAS ESTRICTAS:\n"
        "1. Ignora cabeceras y pies de página del boletín oficial (números de página, URLs, códigos CVE).\n"
        "2. Formatea TODAS las tablas salariales o de datos utilizando exclusivamente sintaxis de tabla "
        "Markdown estándar (`| Columna 1 | Columna 2 | ... |`). Prohibido usar etiquetas HTML `<table>`.\n"
        "3. Si una celda abarca varias filas (rowspan), repite el valor correspondiente en cada fila para "
        "que todas las filas tengan exactamente el mismo número de columnas y ningún importe salarial "
        "quede desplazado.\n"
        "4. Devuelve únicamente el Markdown limpio, sin bloques de código envolventes (```markdown)."
    )

    try:
        response = _generate_with_fallback(
            client=client,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception as exc:
        logger.warning(
            "  -> Respaldo por texto estructurado activado tras timeout/error en bytes PDF (%s)...",
            exc,
        )
        raw_layout_pages = [
            f"--- PÁGINA {p_idx + 1} ---\n" + doc[p_idx].get_text("text", sort=True)
            for p_idx in page_indices
        ]
        response = _generate_with_fallback(
            client=client,
            contents=prompt + "\n\nTEXTO CON LAYOUT DE PÁGINAS:\n" + "\n\n".join(raw_layout_pages),
            config=types.GenerateContentConfig(temperature=0.0),
        )

    text = (response.text or "").strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _page_has_complex_table(page: pymupdf.Page) -> bool:
    """Determina en CPU si una página contiene tablas reales o cuadrículas salariales mediante PyMuPDF."""
    try:
        finder = page.find_tables()
        if finder and any(t.row_count >= 3 and t.col_count >= 2 for t in finder.tables):
            return True
    except Exception:
        pass
    text = page.get_text("text")
    # Respaldo para tablas salariales sin bordes vectoriales: densidad de importes (€ o decimales XXX,XX)
    euro_count = text.count("€")
    decimal_amounts = len(re.findall(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b", text))
    return euro_count >= 6 or decimal_amounts >= 8


def _has_clean_vector_table_grid(page: pymupdf.Page) -> bool:
    """Verifica en CPU si las tablas de la página tienen rejilla vectorial completa sin celdas colapsadas."""
    try:
        finder = page.find_tables()
        if not finder or not finder.tables:
            return False
        for table in finder.tables:
            if table.col_count < 3 or table.row_count < 3:
                return False
            for row in table.extract():
                for cell in row:
                    if cell and len(re.findall(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b", cell)) >= 2:
                        return False
        return True
    except Exception:
        return False


def _extract_clean_vector_table_page_cpu(doc: pymupdf.Document, page_idx: int) -> str:
    """Extrae en CPU mediante pymupdf4llm una página cuya tabla tiene rejilla vectorial limpia."""
    raw_md = pymupdf4llm.to_markdown(doc, pages=[page_idx])
    cleaned_lines: list[str] = []
    for line in raw_md.splitlines():
        stripped = line.strip()
        if _BULLETIN_HEADER_NOISE.match(re.sub(r"[*_#<>/\s]+", " ", stripped).strip()):
            continue
        if re.fullmatch(r"[*_#\s]*\d{1,5}[*_#\s]*", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_hybrid_markdown(
    pdf_path: Path,
    boundary: DocumentBoundary,
    client: Optional[genai.Client] = None,
    max_table_batch_size: int = 3,
) -> str:
    """Extrae el documento en el rango [start_page, end_page] combinando CPU (texto/rejillas limpias) y Gemini (tablas complejas)."""
    genai_client = client or get_genai_client()
    doc = pymupdf.open(pdf_path)

    start_idx = boundary.start_page - 1
    end_idx = boundary.end_page - 1

    markdown_parts: list[str] = []
    pending_table_pages: list[int] = []

    def flush_table_batch() -> None:
        nonlocal pending_table_pages
        if not pending_table_pages:
            return
        human_pages = [p + 1 for p in pending_table_pages]
        logger.info(
            "  -> Extrayendo lote de páginas con tablas sin rejilla %s vía Gemini...",
            human_pages,
        )
        table_md = _extract_table_pages_with_gemini(
            doc, pending_table_pages, genai_client
        )
        markdown_parts.append(table_md)
        pending_table_pages = []

    for p_idx in range(start_idx, end_idx + 1):
        page = doc[p_idx]
        if _page_has_complex_table(page):
            if _has_clean_vector_table_grid(page):
                flush_table_batch()
                logger.info(
                    "  -> Extrayendo tabla con rejilla vectorial limpia (pág. %d) en CPU (pymupdf4llm)...",
                    p_idx + 1,
                )
                markdown_parts.append(_extract_clean_vector_table_page_cpu(doc, p_idx))
            else:
                pending_table_pages.append(p_idx)
                if len(pending_table_pages) >= max_table_batch_size:
                    flush_table_batch()
        else:
            flush_table_batch()
            raw_page_text = page.get_text("text")
            markdown_parts.append(_format_plain_page_to_markdown(raw_page_text))

    flush_table_batch()
    doc.close()

    full_md = "\n\n".join(part for part in markdown_parts if part.strip())
    full_md = re.sub(r"\n{3,}", "\n\n", full_md).strip() + "\n"
    return full_md


def extract_errata_patches(
    errata_pdf_path: Path,
    client: Optional[genai.Client] = None,
) -> ErrataPatchList:
    """Extrae de un PDF de corrección de erratas la lista estructurada de sustituciones en castellano."""
    genai_client = client or get_genai_client()
    doc = pymupdf.open(errata_pdf_path)
    full_text = "\n\n".join(page.get_text("text") for page in doc)
    doc.close()

    prompt = (
        f"Analiza el siguiente documento oficial de Corrección de Erratas ('{errata_pdf_path.name}').\n"
        "Extrae todas las sustituciones indicadas ('donde dice' -> 'debe decir').\n"
        "IMPORTANTE: Si el documento de erratas menciona que el error ocurrió en una versión cooficial "
        "(o cita la frase en castellano/catalán), proporciona las cadenas exactas en CASTELLANO "
        "(tanto la versión con mayúscula inicial como en minúsculas si aplica, ej. 'Faltas muy leves' -> "
        "'Faltas muy graves' y 'faltas muy leves' -> 'faltas muy graves') para asegurar que se corrijan "
        "en el Markdown en castellano del convenio.\n\n"
        f"TEXTO DEL DOCUMENTO DE ERRATAS:\n{full_text}"
    )

    response = _generate_with_fallback(
        client=genai_client,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ErrataPatchList,
        ),
    )
    return ErrataPatchList.model_validate_json(response.text)


def apply_errata_patches_to_markdown(
    markdown_text: str,
    patch_list: ErrataPatchList,
) -> str:
    """Aplica dinámicamente todos los parches extraídos sobre el Markdown del convenio correspondiente."""
    updated = markdown_text
    for patch in patch_list.parches:
        if patch.donde_dice and patch.debe_decir:
            count = updated.count(patch.donde_dice)
            if count > 0:
                updated = updated.replace(patch.donde_dice, patch.debe_decir)
                logger.info(
                    "  -> Parche aplicado en [%s]: '%s' -> '%s' (%d ocurrencias)",
                    patch.contexto_articulo,
                    patch.donde_dice,
                    patch.debe_decir,
                    count,
                )
            elif patch.debe_decir in updated:
                logger.info(
                    "  -> Verificado en [%s]: el texto corregido ('%s') ya consta en la edición en castellano.",
                    patch.contexto_articulo,
                    patch.debe_decir,
                )
            else:
                logger.warning(
                    "  -> No se encontró coincidencia literal para [%s]: '%s'",
                    patch.contexto_articulo,
                    patch.donde_dice,
                )
    return updated


def inspect_all_boundaries(raw_dir: Path = RAW_DATA_DIR) -> dict[str, DocumentBoundary]:
    """Ejecuta únicamente la Etapa 1 (detección de límites y metadatos) sobre todos los PDFs de raw_dir."""
    client = get_genai_client()
    results: dict[str, DocumentBoundary] = {}
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        logger.info("Analizando límites de: %s", pdf_path.name)
        boundary = detect_document_boundaries(pdf_path, client=client)
        results[pdf_path.name] = boundary
        logger.info(
            "  => %s | tipo=%s | territorio=%s | págs=%d..%d | %s",
            pdf_path.name,
            boundary.doc_type,
            boundary.territorio,
            boundary.start_page,
            boundary.end_page,
            boundary.justificacion,
        )
    return results


def run_ingestion_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    skip_existing: bool = True,
) -> list[Path]:
    """Ejecuta el pipeline completo de ingesta híbrida (Etapas 1, 2 y 3) para todos los PDFs en raw_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = get_genai_client()

    cache_path = output_dir / "boundaries_cache.json"
    cached_boundaries: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached_boundaries = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached_boundaries = {}

    boundaries: dict[Path, DocumentBoundary] = {}
    errata_files: list[tuple[Path, DocumentBoundary]] = []
    saved_by_territory: dict[str, Path] = {}
    saved_files: list[Path] = []

    # Etapa 1: Clasificación y detección de límites [start_page, end_page] (con caché en disco)
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        if skip_existing and pdf_path.name in cached_boundaries:
            boundary = DocumentBoundary.model_validate(cached_boundaries[pdf_path.name])
            logger.info(
                "[Etapa 1 - Caché] %s -> tipo=%s, territorio=%s, páginas=%d-%d",
                pdf_path.name,
                boundary.doc_type,
                boundary.territorio,
                boundary.start_page,
                boundary.end_page,
            )
        else:
            boundary = detect_document_boundaries(pdf_path, client=client)
            cached_boundaries[pdf_path.name] = boundary.model_dump()
            cache_path.write_text(
                json.dumps(cached_boundaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "[Etapa 1] %s -> tipo=%s, territorio=%s, páginas=%d-%d",
                pdf_path.name,
                boundary.doc_type,
                boundary.territorio,
                boundary.start_page,
                boundary.end_page,
            )
        boundaries[pdf_path] = boundary
        if boundary.doc_type == "correccion_erratas":
            errata_files.append((pdf_path, boundary))

    # Etapa 2: Extracción híbrida (PyMuPDF CPU + Gemini en tablas) de los documentos principales
    for pdf_path, boundary in boundaries.items():
        if boundary.doc_type == "correccion_erratas":
            continue

        output_path = output_dir / f"{pdf_path.stem}.md"
        if skip_existing and output_path.exists() and output_path.stat().st_size > 100:
            logger.info(
                "[Etapa 2] Ya existe %s (%d bytes), omitiendo re-extracción.",
                output_path.name,
                output_path.stat().st_size,
            )
        else:
            logger.info(
                "[Etapa 2] Extrayendo Markdown híbrido de %s (págs. %d-%d)...",
                pdf_path.name,
                boundary.start_page,
                boundary.end_page,
            )
            md_text = extract_hybrid_markdown(pdf_path, boundary, client=client)
            output_path.write_text(md_text, encoding="utf-8")
            logger.info(
                "[Etapa 2] Guardado %s (%d caracteres).",
                output_path.name,
                len(md_text),
            )

        saved_by_territory[boundary.territorio] = output_path
        saved_files.append(output_path)

    # Etapa 3: Auto-Errata Patcher sobre los convenios correspondientes
    for errata_path, errata_boundary in errata_files:
        patch_list = extract_errata_patches(errata_path, client=client)
        target_territory = patch_list.territorio or errata_boundary.territorio
        target_md_path = saved_by_territory.get(target_territory)

        if not target_md_path:
            # Búsqueda flexible por subcadena de territorio
            for terr, path_candidate in saved_by_territory.items():
                if terr in target_territory or target_territory in terr:
                    target_md_path = path_candidate
                    break

        if target_md_path and target_md_path.exists():
            logger.info(
                "[Etapa 3] Aplicando %d parches de erratas de %s sobre %s...",
                len(patch_list.parches),
                errata_path.name,
                target_md_path.name,
            )
            original_md = target_md_path.read_text(encoding="utf-8")
            patched_md = apply_errata_patches_to_markdown(original_md, patch_list)
            target_md_path.write_text(patched_md, encoding="utf-8")
        else:
            logger.warning(
                "[Etapa 3] No se encontró convenio matriz para aplicar erratas de %s (territorio=%s)",
                errata_path.name,
                target_territory,
            )

    return saved_files


if __name__ == "__main__":
    run_ingestion_pipeline()
