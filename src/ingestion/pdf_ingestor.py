"""Motor de ingesta híbrida automatizada (PyMuPDF en CPU + Google GenAI Multimodal).

Escalable a las 50 provincias sin reglas ni expresiones regulares hardcodeadas por territorio:
1. Detección automática de límites [start_page, end_page] y clasificación con PyMuPDF + Gemini + Pydantic.
2. Extracción híbrida: texto articulado directo en CPU con PyMuPDF y páginas con tablas vía PDF nativo en Gemini.
3. Auto-Errata Patcher: extracción estructurada de parches {donde_dice, debe_decir} con Pydantic y aplicación automática.
"""

from __future__ import annotations

import argparse
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
from src.ingestion.schemas import (
    DocumentBoundary,
    DocumentManifestEntry,
    ErrataPatchList,
)

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
    r"cve-dogc-|cve:\s*[a-z0-9-]+|issn\s+\d|"
    r"centro\s+de\s+ediciones\s+de\s+la\s+diputaci[óo]n|"
    r"https?://\S+|www\.\S+|e-mail:\s*\S+|tel[ée]fonos?:\s*\d|"
    r"p[áa]g(?:ina)?\.?\s*\d+(?:\s*de\s*\d+)?|"
    r"fasc[íi]culo\s+\d+|"
    r"n[º°o]\.?\s*\d+\s*[-•]\s*(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)\b|"
    r"n[úu]m(?:ero)?\.?\s*\d+\s*[-•]?\s*p[áa]gina\s*\d+|"
    r"n[úu]m(?:ero)?\.?\s*\d+$|"
    r"\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}$|"
    r"d\.?\s*l\.?\s*:?\s*[a-z]{1,3}\s*\d+|"
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

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _BULLETIN_HEADER_NOISE.match(stripped):
            continue
        if re.fullmatch(r"\d{1,5}", stripped):
            continue

        # Formatear encabezados estructurales jurídicos (exigiendo mayúscula inicial para no promocionar frases partidas)
        if re.match(
            r"^(?:CAP[ÍI]TULO|Cap[íi]tulo|T[ÍI]TULO|T[íi]tulo|ANEXO|Anexo)\s+(?!d[ée]cimo\b)[IVXLCDM0-9]+(?![IVXLCDM0-9])(?!\s+del\s+(?:presente\s+)?convenio\b)",
            stripped,
        ):
            cleaned_lines.append(f"\n## {stripped}\n")
        elif re.match(
            r"^(?:Art[íi]culo|ART[ÍI]CULO|Art\.|ART\.)\s*\d+(?!\d)(?!\s*(?:º|\.º)?\s+(?:ET\b|del?\s+(?:vigente\s+|presente\s+)?(?:Estatuto|Real\s+Decreto|convenio|Ley)))",
            stripped,
        ):
            cleaned_lines.append(f"\n### {stripped}\n")
        else:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


_THOUSANDS_SPACE_RE = re.compile(r"(?<!\d)(\d{1,2})(?:\.\s+|\s+\.)(\d{3},\d{2})\b")

_TABLE_EXTRACTION_PROMPT = (
    "Extrae íntegramente el contenido de estas páginas pertenecientes a un Convenio Colectivo "
    "de Hostelería en formato GitHub-Flavored Markdown.\n"
    "REGLAS ESTRICTAS:\n"
    "1. Ignora cabeceras y pies de página del boletín oficial (números de página, URLs, códigos CVE).\n"
    "2. Formatea TODAS las tablas salariales o de datos utilizando exclusivamente sintaxis de tabla "
    "Markdown estándar (`| Columna 1 | Columna 2 | ... |`). Prohibido usar etiquetas HTML `<table>`.\n"
    "3. Si una celda abarca varias filas (rowspan), repite el valor correspondiente en cada fila para "
    "que todas las filas tengan exactamente el mismo número de columnas y ningún importe salarial "
    "quede desplazado.\n"
    "4. Devuelve únicamente el Markdown limpio, sin bloques de código envolventes (```markdown).\n"
    "5. Conserva siempre encima de cada tabla Markdown su propio encabezado `### <Tipo de establecimiento / Sección>` "
    "y periodo de vigencia (`## SECCIÓN...`, `### Hoteles/Apartamentos 5 y 4 estrellas`, `### Desde 01/01/2026 "
    "hasta 31/12/2026 (provisionales)`, `### Conceptos económicos...`), evitando agrupar varias sub-tablas sin título "
    "Markdown propio; incluye subtítulos diferenciados `### SALARIOS MES` y `### SALARIOS HORA` cuando una página "
    "contenga ambas modalidades, y formatea los importes españoles sin espacio tras el punto de millar "
    "(`1.623,07`, nunca `1. 623,07`)."
)

_SCANNED_OCR_PROMPT = (
    "Transcribe íntegramente el texto y las tablas de estas páginas escaneadas pertenecientes a un "
    "Convenio Colectivo de Hostelería en formato GitHub-Flavored Markdown.\n"
    "REGLAS ESTRICTAS:\n"
    "1. Ignora cabeceras, pies de página, sellos o marcas laterales del boletín oficial.\n"
    "2. Reconstruye la estructura con títulos (`##`, `###`) para capítulos, secciones y artículos.\n"
    "3. Formatea TODAS las tablas con sintaxis Markdown estándar (`| Columna 1 | Columna 2 | ... |`), "
    "sin HTML `<table>`, repitiendo categorías o celdas combinadas para conservar alineación.\n"
    "4. Devuelve únicamente el Markdown limpio, sin bloques envolventes (```markdown), y formatea los "
    "importes españoles sin espacio tras el punto de millar (`1.623,07`, nunca `1. 623,07`)."
)


def _normalize_spanish_thousands(text: str) -> str:
    """Corrige el artefacto tipográfico `1. 623,07` o `1 .623,07` -> `1.623,07`."""
    return _THOUSANDS_SPACE_RE.sub(r"\1.\2", text)


def _clean_gemini_markdown_output(text: str) -> str:
    """Elimina cercos ```markdown residuales y normaliza millares españoles."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    return _normalize_spanish_thousands(cleaned).strip()


def _extract_pdf_pages_with_gemini(
    doc: pymupdf.Document,
    page_indices: list[int],
    client: genai.Client,
    prompt: str,
    *,
    allow_text_layout_fallback: bool = True,
) -> str:
    """Recorta en memoria páginas de un PDF y las transcribe a Markdown vía Gemini."""
    sub_doc = pymupdf.open()
    for p_idx in page_indices:
        sub_doc.insert_pdf(doc, from_page=p_idx, to_page=p_idx)
    pdf_bytes = sub_doc.tobytes()
    sub_doc.close()

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
        if allow_text_layout_fallback:
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
        else:
            logger.warning(
                "  -> Respaldo por imagen PNG activado tras error en bytes PDF escaneados (%s)...",
                exc,
            )
            image_parts = [
                types.Part.from_bytes(
                    data=doc[p_idx].get_pixmap(dpi=200).tobytes("png"),
                    mime_type="image/png",
                )
                for p_idx in page_indices
            ]
            response = _generate_with_fallback(
                client=client,
                contents=[*image_parts, prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )

    return _clean_gemini_markdown_output(response.text or "")


def _extract_table_pages_with_gemini(
    doc: pymupdf.Document,
    page_indices: list[int],
    client: genai.Client,
) -> str:
    """Recorta en memoria un conjunto de páginas con tablas y las convierte a Markdown vía Gemini."""
    return _extract_pdf_pages_with_gemini(
        doc=doc,
        page_indices=page_indices,
        client=client,
        prompt=_TABLE_EXTRACTION_PROMPT,
        allow_text_layout_fallback=True,
    )


def _extract_scanned_pages_with_gemini(
    doc: pymupdf.Document,
    page_indices: list[int],
    client: genai.Client,
) -> str:
    """Transcribe mediante OCR multimodal en Gemini páginas escaneadas (imagen sin capa de texto)."""
    return _extract_pdf_pages_with_gemini(
        doc=doc,
        page_indices=page_indices,
        client=client,
        prompt=_SCANNED_OCR_PROMPT,
        allow_text_layout_fallback=False,
    )


def _is_scanned_page(page: pymupdf.Page, min_chars: int = 80) -> bool:
    """Detecta si una página carece de capa de texto útil y contiene imágenes escaneadas."""
    text_len = len(page.get_text("text").strip())
    return text_len < min_chars and bool(page.get_images())


def _extract_double_column_page_text(page: pymupdf.Page) -> str:
    """Extrae texto respetando el orden columna izquierda -> columna derecha en boletines a 2 columnas.

    Mantiene encabezados/pies de ancho completo en su posición relativa y revierte al texto plano
    si la página no muestra bloques separados en ambas mitades.
    """
    blocks = [
        b
        for b in page.get_text("blocks")
        if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()
    ]
    if not blocks:
        return ""

    page_width = float(page.rect.width or 0.0)
    if page_width <= 0.0:
        return page.get_text("text")

    mid_x = page_width / 2.0
    wide_blocks: list[tuple] = []
    left_blocks: list[tuple] = []
    right_blocks: list[tuple] = []

    for block in blocks:
        x0, y0, x1, y1 = float(block[0]), float(block[1]), float(block[2]), float(block[3])
        block_width = x1 - x0
        if block_width >= page_width * 0.55 and x0 < mid_x - 15.0 and x1 > mid_x + 15.0:
            wide_blocks.append(block)
        elif (x0 + x1) / 2.0 < mid_x:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    if not left_blocks or not right_blocks:
        return page.get_text("text")

    left_blocks.sort(key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))
    right_blocks.sort(key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))

    col_top_y = min(float(left_blocks[0][1]), float(right_blocks[0][1]))
    col_bottom_y = max(float(left_blocks[-1][3]), float(right_blocks[-1][3]))

    top_wide: list[tuple] = []
    mid_wide: list[tuple] = []
    bottom_wide: list[tuple] = []
    for wb in sorted(wide_blocks, key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1))):
        if float(wb[3]) <= col_top_y + 5.0:
            top_wide.append(wb)
        elif float(wb[1]) >= col_bottom_y - 5.0:
            bottom_wide.append(wb)
        else:
            mid_wide.append(wb)

    ordered = [*top_wide, *left_blocks, *mid_wide, *right_blocks, *bottom_wide]
    return "\n".join(str(b[4]).strip() for b in ordered)


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
    """Verifica en CPU si las tablas de la página tienen rejilla vectorial completa sin celdas colapsadas o fusionadas."""
    try:
        finder = page.find_tables()
        if not finder or not finder.tables:
            return False
        for table in finder.tables:
            if table.col_count < 3 or table.row_count < 3:
                return False
            rows = table.extract()
            if not rows or any(c is None or not str(c).strip() for c in rows[0]):
                return False
            for row in rows:
                for cell in row:
                    if cell and (
                        "\n," in cell
                        or ",\n" in cell
                        or len(re.findall(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b", cell)) >= 2
                    ):
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
        if _BULLETIN_HEADER_NOISE.match(re.sub(r"[*_#<>\s]+", " ", stripped).strip()):
            continue
        if re.fullmatch(r"[*_#\s]*\d{1,5}[*_#\s]*", stripped):
            continue
        cleaned_lines.append(line)
    out = "\n".join(cleaned_lines).strip()
    return _normalize_spanish_thousands(out)


def extract_hybrid_markdown(
    pdf_path: Path,
    boundary: DocumentBoundary | DocumentManifestEntry,
    client: Optional[genai.Client] = None,
    max_table_batch_size: int = 2,
) -> str:
    """Extrae el rango [start_page, end_page] combinando adaptadores CPU (1 col / 2 col / rejilla limpia) y Gemini (tablas complejas / OCR)."""
    genai_client = client
    doc = pymupdf.open(pdf_path)

    layout = getattr(boundary, "layout", "single_column")
    force_gemini_pages = set(getattr(boundary, "force_gemini_pages", []) or [])
    skip_pages = set(getattr(boundary, "skip_pages", []) or [])

    start_idx = boundary.start_page - 1
    end_idx = boundary.end_page - 1

    markdown_parts: list[str] = []
    pending_table_pages: list[int] = []
    pending_scanned_pages: list[int] = []

    def _client() -> genai.Client:
        nonlocal genai_client
        if genai_client is None:
            genai_client = get_genai_client()
        return genai_client

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
            doc, pending_table_pages, _client()
        )
        markdown_parts.append(table_md)
        pending_table_pages = []

    def flush_scanned_batch() -> None:
        nonlocal pending_scanned_pages
        if not pending_scanned_pages:
            return
        human_pages = [p + 1 for p in pending_scanned_pages]
        logger.info(
            "  -> Transcribiendo lote de páginas escaneadas %s vía Gemini OCR...",
            human_pages,
        )
        scanned_md = _extract_scanned_pages_with_gemini(
            doc, pending_scanned_pages, _client()
        )
        markdown_parts.append(scanned_md)
        pending_scanned_pages = []

    for p_idx in range(start_idx, end_idx + 1):
        human_page = p_idx + 1
        if human_page in skip_pages:
            flush_table_batch()
            flush_scanned_batch()
            logger.info("  -> Omitiendo página %d por directiva skip_pages del manifiesto.", human_page)
            continue

        page = doc[p_idx]

        if human_page in force_gemini_pages:
            flush_scanned_batch()
            pending_table_pages.append(p_idx)
            if len(pending_table_pages) >= max_table_batch_size:
                flush_table_batch()
            continue

        if layout == "scanned_ocr" or _is_scanned_page(page):
            flush_table_batch()
            pending_scanned_pages.append(p_idx)
            if len(pending_scanned_pages) >= max_table_batch_size:
                flush_scanned_batch()
            continue

        flush_scanned_batch()
        if _page_has_complex_table(page):
            if _has_clean_vector_table_grid(page):
                flush_table_batch()
                logger.info(
                    "  -> Extrayendo tabla con rejilla vectorial limpia (pág. %d) en CPU (pymupdf4llm)...",
                    human_page,
                )
                markdown_parts.append(_extract_clean_vector_table_page_cpu(doc, p_idx))
            else:
                pending_table_pages.append(p_idx)
                if len(pending_table_pages) >= max_table_batch_size:
                    flush_table_batch()
        else:
            flush_table_batch()
            raw_page_text = (
                _extract_double_column_page_text(page)
                if layout == "double_column"
                else page.get_text("text")
            )
            markdown_parts.append(_format_plain_page_to_markdown(raw_page_text))

    flush_table_batch()
    flush_scanned_batch()
    doc.close()

    full_md = "\n\n".join(part for part in markdown_parts if part.strip())
    full_md = _normalize_spanish_thousands(full_md)
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


def _default_doc_role(doc_type: str) -> str:
    """Asigna un rol documental por defecto coherente con `doc_type`."""
    return {
        "estatuto_trabajadores": "estatuto",
        "aleh_estatal": "aleh",
        "correccion_erratas": "erratas",
    }.get(doc_type, "completo")


def _serialize_manifest_entry(entry: DocumentManifestEntry) -> dict:
    """Serializa una entrada del manifiesto conservando campos clave y directivas no vacías."""
    raw = entry.model_dump()
    core_keys = {"doc_type", "doc_role", "territorio", "start_page", "end_page", "justificacion"}
    return {
        key: value
        for key, value in raw.items()
        if key in core_keys or (key == "layout" and value != "single_column") or bool(value)
    }


def _matches_only_filter(pdf_path: Path, only_files: list[str]) -> bool:
    """Determina si `pdf_path` coincide con algún token de `--only` (nombre, stem o subcadena)."""
    name_lower = pdf_path.name.lower()
    stem_lower = pdf_path.stem.lower()
    for raw_token in only_files:
        token = Path(raw_token).name.lower().strip()
        if not token:
            continue
        if token in {name_lower, stem_lower} or token in stem_lower:
            return True
    return False


def run_ingestion_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    skip_existing: bool = True,
    only_files: Optional[list[str]] = None,
    validate: bool = False,
    redetect_boundaries: bool = False,
) -> list[Path]:
    """Ejecuta el pipeline de ingesta híbrida (Etapas 1, 2 y 3) con soporte selectivo `--only` y Gate de Calidad."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client: Optional[genai.Client] = None

    def _get_client() -> genai.Client:
        nonlocal client
        if client is None:
            client = get_genai_client()
        return client

    cache_path = output_dir / "boundaries_cache.json"
    cached_boundaries: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached_boundaries = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached_boundaries = {}

    all_pdfs = sorted(raw_dir.glob("*.pdf"))
    target_pdfs = (
        [p for p in all_pdfs if _matches_only_filter(p, only_files)]
        if only_files
        else all_pdfs
    )
    if only_files and not target_pdfs:
        logger.warning("Ningún PDF en %s coincidió con --only=%s", raw_dir, only_files)
        return []

    boundaries: dict[Path, DocumentManifestEntry] = {}
    errata_files: list[tuple[Path, DocumentManifestEntry]] = []
    saved_by_territory: dict[str, Path] = {}
    saved_files: list[Path] = []
    freshly_extracted_paths: set[Path] = set()

    # Etapa 1: Manifiesto declarativo / detección automática de límites (respetando entradas curadas en caché)
    for pdf_path in target_pdfs:
        if pdf_path.name in cached_boundaries and not redetect_boundaries:
            manifest_entry = DocumentManifestEntry.model_validate(
                cached_boundaries[pdf_path.name]
            )
            logger.info(
                "[Etapa 1 - Manifiesto] %s -> rol=%s, layout=%s, territorio=%s, páginas=%d-%d",
                pdf_path.name,
                manifest_entry.doc_role,
                manifest_entry.layout,
                manifest_entry.territorio,
                manifest_entry.start_page,
                manifest_entry.end_page,
            )
        else:
            detected = detect_document_boundaries(pdf_path, client=_get_client())
            manifest_entry = DocumentManifestEntry(
                **detected.model_dump(),
                doc_role=_default_doc_role(detected.doc_type),
            )
            cached_boundaries[pdf_path.name] = _serialize_manifest_entry(manifest_entry)
            cache_path.write_text(
                json.dumps(cached_boundaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "[Etapa 1] %s -> rol=%s, territorio=%s, páginas=%d-%d",
                pdf_path.name,
                manifest_entry.doc_role,
                manifest_entry.territorio,
                manifest_entry.start_page,
                manifest_entry.end_page,
            )
        boundaries[pdf_path] = manifest_entry
        if manifest_entry.doc_type == "correccion_erratas":
            errata_files.append((pdf_path, manifest_entry))

    # Etapa 2: Extracción híbrida por familia de formato
    for pdf_path, manifest_entry in boundaries.items():
        if manifest_entry.doc_type == "correccion_erratas":
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
                "[Etapa 2] Extrayendo Markdown (%s) de %s (págs. %d-%d)...",
                manifest_entry.layout,
                pdf_path.name,
                manifest_entry.start_page,
                manifest_entry.end_page,
            )
            md_text = extract_hybrid_markdown(
                pdf_path,
                manifest_entry,
                client=client,
            )
            output_path.write_text(md_text, encoding="utf-8")
            freshly_extracted_paths.add(output_path)
            logger.info(
                "[Etapa 2] Guardado %s (%d caracteres).",
                output_path.name,
                len(md_text),
            )

        if manifest_entry.territorio not in saved_by_territory or manifest_entry.doc_role in {
            "completo",
            "base",
        }:
            saved_by_territory[manifest_entry.territorio] = output_path
        saved_files.append(output_path)

    # Etapa 3: Auto-Errata Patcher (siempre que el convenio matriz esté en esta ejecución)
    for errata_path, errata_boundary in errata_files:
        prelim_target = saved_by_territory.get(errata_boundary.territorio)
        if not prelim_target:
            for terr, path_candidate in saved_by_territory.items():
                if terr in errata_boundary.territorio or errata_boundary.territorio in terr:
                    prelim_target = path_candidate
                    break
        if skip_existing and (
            not freshly_extracted_paths
            or (prelim_target and prelim_target not in freshly_extracted_paths)
        ):
            logger.info(
                "[Etapa 3] Sin convenio matriz recién extraído para '%s'; omitiendo re-evaluación de erratas (%s).",
                errata_boundary.territorio,
                errata_path.name,
            )
            continue
        patch_list = extract_errata_patches(errata_path, client=_get_client())
        target_territory = patch_list.territorio or errata_boundary.territorio
        target_md_path = saved_by_territory.get(target_territory)

        if not target_md_path:
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

    if validate and saved_files:
        from src.ingestion.validator import format_validation_summary, validate_document

        reports = [
            validate_document(md_path, raw_dir=raw_dir)
            for md_path in saved_files
            if md_path.exists()
        ]
        for rep in reports:
            logger.info("\n%s", format_validation_summary(rep))
        if any(not rep.passed for rep in reports):
            failed_names = [rep.md_file for rep in reports if not rep.passed]
            raise RuntimeError(
                f"Gate de Calidad rechazó los documentos con ERROR: {', '.join(failed_names)}"
            )

    return saved_files


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Motor de ingesta híbrida de convenios PDF -> Markdown Silver."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Procesa únicamente los PDFs indicados (nombre, stem o subcadena; p. ej. --only sevilla).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza re-extracción en Etapa 2 aunque el .md ya exista (mantiene boundaries_cache.json).",
    )
    parser.add_argument(
        "--redetect-boundaries",
        action="store_true",
        help="Re-ejecuta Etapa 1 con Gemini aunque el PDF ya exista en boundaries_cache.json.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Ejecuta el Gate de Calidad en CPU al terminar y finaliza con código != 0 si hay ERROR.",
    )
    args = parser.parse_args()
    run_ingestion_pipeline(
        skip_existing=not args.force,
        only_files=args.only,
        validate=args.validate,
        redetect_boundaries=args.redetect_boundaries,
    )


if __name__ == "__main__":
    _cli()
