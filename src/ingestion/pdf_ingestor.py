"""Motor de ingesta híbrida automatizada (PyMuPDF en CPU + Google GenAI Multimodal).

Escalable a las 50 provincias sin reglas ni expresiones regulares hardcodeadas por territorio:
1. Detección automática de límites [start_page, end_page] y clasificación con PyMuPDF + Gemini + Pydantic.
2. Extracción híbrida: texto articulado directo en CPU con PyMuPDF y páginas con tablas vía PDF nativo en Gemini.
3. Auto-Errata Patcher: extracción estructurada de parches {donde_dice, debe_decir} con Pydantic y aplicación automática.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    r"bolet[íi]n\s+oficial\s+d[ae]\b|"
    r"b\.?\s*o\.?\s*p\.?(?:\s+d[ae]\b|\s+n[úu]m\.?\b)|"
    r"bop$|"
    r"butllet[íi]\s+oficial\s+de|"
    r"edita\s+excma\.?\s+diputaci[óo]n?\s+provincial|"
    r"b\.?\s*o\.?\s*c\.?\s*m\.?(?:\s+n[úu]m)?|"
    r"bocm-\d+|"
    r"diari\s+oficial\s+de\s+la\s+generalitat|"
    r"cve-dogc-|cve:\s*[a-z0-9-]+|issn\s+\d|"
    r"centro\s+de\s+ediciones\s+de\s+la\s+diputaci[óo]n|"
    r"https?://\S+|www\.\S+|e-mail:\s*\S+|tel[ée]fonos?:\s*\d|"
    r"p[áa]g(?:ina)?\.?\s*\d+(?:\s*(?:de|/)\s*\d+)?|"
    r"fasc[íi]culo\s+\d+|"
    r"n[º°o]\.?\s*\d+\s+de\s+\d{1,2}/\d{2}/\d{4}|"
    r"n[º°o]\.?\s*\d+\s*[-•]\s*(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)\b|"
    r"\[?n[úu]m(?:ero)?\.?\s*\d+\]?(?:\s*[-•]?\s*p[áa]gina\s*\d+)?$|"
    r"n[úu]m(?:ero)?\.?\s*\d+\s*[-•]?\s*p[áa]gina\s*\d+|"
    r"n[úu]mero\s+de\s+anuncio\b|"
    r"\d{1,5}\s*/\s*20[12]\d$|"
    r"20[12]\d/\d{3,6}$|"
    r"\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|xaneiro|febreiro|maio|xu[ñn]o|xullo|setembro|outubro|novembro|decembro)\s+de\s+\d{4}$|"
    r"d\.?\s*l\.?\s*:?\s*[a-z]{1,3}[\s-]*\d+|"
    r"20[12]\d-\d{5}$|"
    r"\d{1,3}/\d{1,3}$|"
    r"(?:lunes|luns|martes|mi[ée]rcoles|m[ée]rcores|jueves|xoves|viernes|venres|s[áa]bado|domingo),?\s+\d+\s+(?:de\s+)?\w+\s+de\s+\d{4}|"
    r"(?:e?k\\?cv\s+gao-[a-z0-9-]+)|"
    r"gipuzkoako\s+aldizkari\s+ofiziala|"
    r"n[úu]mero\s*\|\s*zenbakia\b|"
    r"www\.gipuzkoa\.eus\b|"
    r"lg\.?:?\s*s\.?s\.?\s*\d+|"
    r"eusko\s+jaurlaritza"
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


def _apply_text_markers(
    text: str,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
) -> str:
    """Recorta texto por marcadores opcionales de inicio/fin al principio de su línea."""
    result = text
    if start_marker:
        match_start = re.search(re.escape(start_marker), result, re.IGNORECASE)
        if match_start:
            line_start = result.rfind("\n", 0, match_start.start())
            result = result[line_start + 1 if line_start != -1 else 0 :]
    if end_marker:
        match_end = re.search(re.escape(end_marker), result, re.IGNORECASE)
        if match_end:
            line_start = result.rfind("\n", 0, match_end.start())
            result = result[: line_start if line_start != -1 else 0]
    return result


def _format_plain_page_to_markdown(page_text: str) -> str:
    """Formatea en CPU una página de texto articulado eliminando ruido genérico de boletín oficial."""
    normalized_text = re.sub(r"[\u2000-\u200b\u00a0]+", " ", page_text or "").replace("\xad", "")
    raw_lines = normalized_text.splitlines()
    merged_lines: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        curr = raw_lines[idx].strip()
        if re.fullmatch(r"(?i)(?:art[íi]culo|artigo|article)", curr) and idx + 1 < len(raw_lines):
            nxt = raw_lines[idx + 1].strip()
            if re.fullmatch(
                r"(?i)\d+(?:\s*(?:º|\.º|BIS|TER|QUATER|bis|ter|quater))?\.?(?:\s+.*)?",
                nxt,
            ):
                combined = f"{curr} {nxt}".strip()
                consumed = 2
                if idx + 2 < len(raw_lines):
                    nxt2 = raw_lines[idx + 2].strip()
                    if (
                        nxt2
                        and len(nxt2) <= 60
                        and nxt2 == nxt2.upper()
                        and any(ch.isalpha() for ch in nxt2)
                        and not _BULLETIN_HEADER_NOISE.match(nxt2)
                        and not re.match(
                            r"(?i)^(?:art[íi]culo|artigo|article|cap[íi]tulo|t[íi]tulo|anexo|disposici[óo]n)\b",
                            nxt2,
                        )
                    ):
                        combined = f"{combined} {nxt2}".strip()
                        consumed = 3
                merged_lines.append(combined)
                idx += consumed
                continue
        merged_lines.append(raw_lines[idx])
        idx += 1

    cleaned_lines: list[str] = []

    for line in merged_lines:
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
            r"^(?:CAP[ÍI]TULO|Cap[íi]tulo|T[ÍI]TULO|T[íi]tulo|ANEXO|Anexo)\s+(?!d[ée]cimo\b)[IVXLCDM0-9]+(?![IVXLCDM0-9])(?!\s*(?:,|(?i:del?\s+(?:presente\s+|vigente\s+|actual\s+|citado\s+)?(?:convenio|estatuto|real\s+decreto|ley)\b|de\s+este\s+convenio\b)))",
            stripped,
        ):
            cleaned_lines.append(f"\n## {stripped}\n")
        elif re.match(
            r"^(?:Art[íi]culo|ART[ÍI]CULO|Artigo|ARTIGO|Article|ARTICLE|Art\.|ART\.)(?:\.-|\.|:)?\s*\d+(?!\d)(?!\s*(?:º|\.º)?\s+(?:ET\b|del?\s+(?:vigente\s+|presente\s+)?(?:Estatuto|Real\s+Decreto|convenio|Ley)))",
            stripped,
        ):
            cleaned_lines.append(f"\n### {stripped}\n")
        elif (
            len(stripped) <= 120
            and stripped.startswith(("DISPOSICIÓN", "Disposición"))
            and re.match(
                r"^(?:DISPOSICI[ÓO]N|Disposici[óo]n)\s+(?:ADICIONAL|Adicional|TRANSITORIA|Transitoria|DEROGATORIA|Derogatoria|FINAL|Final)\s+(?:[ÚU]NICA|[ÚU]nica|PRIMEIRA|Primeira|PRIMERA|Primera|SEGUNDA|Segunda|TERCEIRA|Terceira|TERCERA|Tercera|CUARTA|Cuarta|QUINTA|Quinta|SEXTA|Sexta|S[ÉE]PTIMA|S[ée]ptima|OCTAVA|Octava|NOVENA|Novena|D[ÉE]CIMA|D[ée]cima|\d+[ªº]?)\b(?![,;])",
                stripped,
            )
            and not re.search(
                r"(?i)\bdel?\s+(?:presente\s+|vigente\s+|citado\s+)?(?:convenio|estatuto|real\s+decreto|ley)\b",
                stripped,
            )
        ):
            cleaned_lines.append(f"\n### {stripped}\n")
        elif (
            stripped == stripped.upper()
            and re.match(
                r"^(?:T[ÁA]BOAS?\s+SALARIA(?:IS|LES)\.?\b[^\n]*|TABLAS?\s+SALARIAL(?:ES)?\s+20[23]\d\b[^\n]*|SALARIO\s+BASE\s+(?:MENSUAL|MANSUAL|ANUAL)\s+20[23]\d\b[^\n]*|HORAS?\s+EXTRAS?\s+20[23]\d\b[^\n]*|INCREMENTO\s+20[23]\d\b[^\n]*)$",
                stripped,
            )
        ):
            cleaned_lines.append(f"\n### {stripped}\n")
        elif (
            len(stripped) <= 90
            and not stripped.endswith(".")
            and re.match(
                r"^(?:Tabla(?:s)?\s+salarial(?:es)?|TABLA(?:S)?\s+SALARIAL(?:ES)?|T[áa]boa(?:s)?\s+salarial(?:is|es)?|T[ÁA]BOA(?:S)?\s+SALARIA(?:IS|LES)?)\b[^\n]*\b20[23]\d\b[^\n]*$",
                stripped,
            )
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
    "5. Formatea el título general del bloque o documento (incluyendo su periodo/año de vigencia, p. ej. "
    "`## TABLA DE RETRIBUCIONES - Del 1 de enero a 31 de diciembre de 2026` o `## SECCIÓN...`) siempre como "
    "encabezado Markdown nivel 2 (`##`), y coloca encima de cada sub-tabla su propio encabezado nivel 3 (`###`, "
    "p. ej. `### Salarios base mensuales`, `### Hoteles/Apartamentos 5 y 4 estrellas`, `### Desde 01/01/2026 hasta "
    "31/12/2026 (provisionales)`, `### Conceptos económicos...`), evitando agrupar varias sub-tablas sin título "
    "Markdown propio; incluye subtítulos diferenciados `### SALARIOS MES` y `### SALARIOS HORA` cuando una página "
    "contenga ambas modalidades, y formatea los importes españoles sin espacio tras el punto de millar "
    "(`1.623,07`, nunca `1. 623,07`).\n"
    "6. NUNCA omitas ni descartes la primera columna izquierda de puestos de trabajo, ocupaciones, "
    "categorías profesionales, niveles o conceptos económicos, aunque en el PDF su celda de cabecera aparezca "
    "en blanco o combinada: crea siempre una primera columna explícita (p. ej. `Categoría / Puesto` u "
    "`Ocupación / Puesto`) y transcribe íntegramente el nombre literal de cada fila y de cada cabecera de "
    "columna sin recortar letras ni truncar encabezados."
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


_HEADING_PERIOD_SUBTITLE_RE = re.compile(
    r"(?im)^(##{1,2}\s+[^\n]+)\n+(?![#|>\-*])"
    r"(?=(?:del?\s+\d|desde\s+\d|a[ñn]o\s+20[23]\d|vigencia\b|ejercicio\s+20[23]\d|tablas?\s+20[23]\d))"
    r"([^\n]{4,120}\b20[23]\d\b[^\n]{0,30})$"
)


def _normalize_spanish_thousands(text: str) -> str:
    """Corrige el artefacto tipográfico `1. 623,07` o `1 .623,07` -> `1.623,07`."""
    return _THOUSANDS_SPACE_RE.sub(r"\1.\2", text)


def _merge_heading_period_subtitles(text: str) -> str:
    """Fusiona un subtítulo de vigencia anual inmediatamente posterior a un encabezado ##/###."""
    return _HEADING_PERIOD_SUBTITLE_RE.sub(
        lambda m: (
            m.group(0)
            if re.search(r"\b20[23]\d\b", m.group(1))
            else f"{m.group(1).strip()} — {m.group(2).strip()}"
        ),
        text,
    )


def _clean_gemini_markdown_output(text: str) -> str:
    """Elimina cercos ```markdown residuales, fusiona vigencias en títulos y normaliza millares españoles."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)
    cleaned = _merge_heading_period_subtitles(cleaned)
    return _normalize_spanish_thousands(cleaned).strip()


_SPANISH_DECIMALS_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}\b")


def _count_missing_spanish_decimals(pdf_text: str, md_text: str) -> tuple[int, int, list[str]]:
    """Compara importes decimales españoles `X,XX` entre texto PDF y Markdown extraído."""
    normalized_pdf = _normalize_spanish_thousands(pdf_text or "")
    pdf_cnt = Counter(_SPANISH_DECIMALS_RE.findall(normalized_pdf))
    md_cnt = Counter(_SPANISH_DECIMALS_RE.findall(md_text or ""))
    missing = pdf_cnt - md_cnt
    return sum(missing.values()), sum(pdf_cnt.values()), list(missing.elements())[:8]


def _warn_if_max_tokens(response: types.GenerateContentResponse, page_indices: list[int]) -> None:
    """Registra una advertencia si algún candidato de Gemini terminó por límite de tokens."""
    for cand in getattr(response, "candidates", None) or []:
        finish_reason = getattr(cand, "finish_reason", None)
        finish_name = getattr(finish_reason, "name", str(finish_reason or "")).upper()
        if "MAX_TOKENS" in finish_name:
            logger.warning(
                "  -> AVISO: Gemini finalizó por MAX_TOKENS en páginas %s; revisar posible truncamiento.",
                [p_idx + 1 for p_idx in page_indices],
            )
            break


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

    raw_layout_pages = (
        [
            f"--- PÁGINA {p_idx + 1} ---\n" + doc[p_idx].get_text("text", sort=True)
            for p_idx in page_indices
        ]
        if allow_text_layout_fallback
        else []
    )
    layout_anchor = (
        "\n\nREFERENCIA LITERAL DEL TEXTO ALINEADO (úsala junto al PDF visual para no omitir ninguna columna "
        "izquierda de puestos/categorías ni recortar cabeceras o importes):\n"
        + "\n\n".join(raw_layout_pages)
        if raw_layout_pages
        else ""
    )

    try:
        response = _generate_with_fallback(
            client=client,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt + layout_anchor,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
    except Exception as exc:
        if allow_text_layout_fallback:
            logger.warning(
                "  -> Respaldo por texto estructurado activado tras timeout/error en bytes PDF (%s)...",
                exc,
            )
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

    _warn_if_max_tokens(response, page_indices)
    md_out = _clean_gemini_markdown_output(response.text or "")

    if allow_text_layout_fallback and raw_layout_pages:
        layout_joined = "\n".join(raw_layout_pages)
        missing_total, pdf_total, sample_missing = _count_missing_spanish_decimals(layout_joined, md_out)
        if pdf_total >= 4 and missing_total > 0:
            human_pages = [p_idx + 1 for p_idx in page_indices]
            logger.warning(
                "  -> Déficit decimal detectado en páginas %s (%d/%d faltantes, ejemplos=%s); "
                "reintentando con modelo alternativo y refuerzo anti-fusión...",
                human_pages,
                missing_total,
                pdf_total,
                sample_missing[:5],
            )
            repair_addendum = (
                f"\n\nATENCIÓN CRÍTICA DE AUDITORÍA: En tu intento anterior faltaron {missing_total} de {pdf_total} "
                f"importes decimales del PDF (ejemplos omitidos: {', '.join(sample_missing[:6])}). "
                "PROHIBIDO fusionar o eliminar columnas adyacentes aunque compartan importes idénticos "
                "(p. ej. si 2 o 3 columnas de establecimiento tienen exactamente el mismo importe, debes emitir "
                "CADA columna por separado). Emite obligatoriamente tantas columnas monetarias por fila como importes "
                "haya en cada línea de la REFERENCIA LITERAL DEL TEXTO ALINEADO."
            )
            retry_models = (
                (*GEMINI_FALLBACK_MODELS[1:], GEMINI_FALLBACK_MODELS[0])
                if len(GEMINI_FALLBACK_MODELS) > 1
                else GEMINI_FALLBACK_MODELS
            )
            try:
                retry_resp = _generate_with_fallback(
                    client=client,
                    contents=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                        prompt + layout_anchor + repair_addendum,
                    ],
                    config=types.GenerateContentConfig(temperature=0.0),
                    models=retry_models,
                )
                _warn_if_max_tokens(retry_resp, page_indices)
                retry_md = _clean_gemini_markdown_output(retry_resp.text or "")
                retry_missing, _, _ = _count_missing_spanish_decimals(layout_joined, retry_md)
                if retry_missing < missing_total:
                    logger.info(
                        "  -> Reintento mejoró paridad en páginas %s: faltantes %d -> %d.",
                        human_pages,
                        missing_total,
                        retry_missing,
                    )
                    md_out = retry_md
            except Exception as exc:
                logger.warning("  -> Falló el reintento de paridad en páginas %s (%s).", human_pages, exc)

    return md_out


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


_NON_SPANISH_COOFFICIAL_TOKENS: frozenset[str] = frozenset(
    {
        "eranskina", "artikulua", "kapitulua", "atala", "xedapen", "gehigarria",
        "maila", "mailak", "mailako", "ordu", "orduko", "orduetan",
        "hitzarmen", "hitzarmena", "hitzarmenaren", "hitzarmeneko",
        "kolektiboa", "kolektiboaren", "kolektiboko",
        "langile", "langileen", "langileek", "langileak", "lanbide",
        "soldata", "oinarrizko", "ordain", "ordainduko",
        "zerbitz", "zerbitzaria", "zerbitzariak", "zerbitzu", "zerbitzuak", "zerbitzuen",
        "aparteko", "kapital", "aseguratua", "indargabetua", "zenbatekoa",
        "eta", "edo", "dira", "dute", "baino", "arte", "arteko",
        "urteko", "urtean", "urterako", "hilean", "hileko", "balioa",
        "mantenua", "hirurtekoa", "hirurtekotan", "hirurtekoaren",
        "okupazioak", "lanpostuak", "taldea", "taldeak",
        "kategoria", "kategoriak", "korrelatiboa", "ostalaritzaren", "ostalaritzako",
        "gipuzkoako", "ostatuen", "jatetxe", "sukalde", "sukaldaria", "laguntzailea",
        "eltzezaina", "garbitzailea", "solairu", "egun", "eremu", "funtzionala",
        "antzinatasuna", "antzinatasunik", "gabeko", "gaueko", "lanaren",
        "guztira", "hil", "osoa", "partziala", "kontzeptua", "heriotza",
        "baliaezintasun", "istripuak", "jaiegunaren", "abenduaren", "azpisektoreko",
        "enpresen", "aldirako", "bitarteko", "egindako", "gidaria", "atezaina",
        "zilargintza", "lentzeriako", "plantxako", "joskintzako", "txikiagoa", "handiagoa",
        "lehen", "bigarren", "hirugarren", "laugarren", "bosgarren",
        "annex", "capítol", "treballadors", "conveni", "artigo", "táboa", "traballadores",
    }
)

_SPANISH_LANGUAGE_TOKENS: frozenset[str] = frozenset(
    {
        "anexo", "anexos", "artículo", "articulo", "capítulo", "capitulo",
        "disposición", "disposicion", "nivel", "niveles", "categoría", "categoria",
        "categorías", "categorias", "convenio", "colectivo", "grupo", "profesional",
        "salario", "salariales", "salarios", "ocupaciones", "puestos", "capital",
        "asegurado", "manutención", "manutencion", "trienio", "trienios", "valor",
        "mensual", "anual", "antigüedad", "antiguedad", "camarero", "camarera",
        "cocinero", "ayudante", "marmitón", "pinche", "limpieza",
        "del", "las", "los", "para", "con", "por", "sobre", "entre", "hasta", "según", "segun",
        "servicios", "extraordinarios", "extraordinarias", "trabajo", "trabajadores",
        "trabajadoras", "días", "dias", "horas", "hora", "año", "años", "mes",
        "completa", "parcial", "nocturnidad", "nocturno", "festivo", "festivos",
        "diciembre", "extinto", "hostelería", "hosteleria", "alojamientos",
        "restauración", "restauracion", "gastos", "mantenimiento", "administración",
        "administracion", "recepción", "recepcion", "muerte", "accidente",
        "enfermedad", "invalidez", "conductor", "portero", "correlativa", "tablas",
    }
)


def _normalize_unicode_spaces(text: str) -> str:
    """Normaliza espacios tipográficos Unicode (p. ej. U+2005/NBSP) y guiones blandos."""
    cleaned = re.sub(r"[\u2000-\u200b\u00a0]+", " ", text or "")
    # Conserva como raya (–) el carácter U+00AD cuando actúa como símbolo de "sin importe" al final de puntos o espacios.
    cleaned = re.sub(r"(?:(?<=^)|(?<=\s)|(?<=\.))\xad(?=\s|$)", "–", cleaned)
    return cleaned.replace("\xad", "")


def _is_coofficial_non_spanish_text(text: str) -> bool:
    """Clasifica determinísticamente si un bloque está redactado en lengua cooficial distinta del castellano."""
    cleaned = _normalize_unicode_spaces(text).lower()
    tokens = re.findall(r"[a-záéíóúüñ]{2,}", cleaned)
    if not tokens:
        return False
    non_es_score = sum(1 for tok in tokens if tok in _NON_SPANISH_COOFFICIAL_TOKENS)
    es_score = sum(1 for tok in tokens if tok in _SPANISH_LANGUAGE_TOKENS)
    return (non_es_score >= 2 and non_es_score > es_score) or (
        non_es_score >= 1 and es_score == 0 and len(tokens) <= 4
    )


def _extract_bilingual_right_es_page_text(
    page: pymupdf.Page,
    exclude_rects: Optional[list[pymupdf.Rect]] = None,
) -> str:
    """Extrae únicamente el texto en castellano (columna derecha y bloques/tablas anchos en castellano) de páginas bilingües."""
    page_width = float(page.rect.width or 0.0)
    page_height = float(page.rect.height or 0.0)
    if page_width <= 0.0:
        return _normalize_unicode_spaces(page.get_text("text"))

    mid_x = page_width / 2.0
    y_top = 34.0
    y_bot = page_height - 24.0 if page_height > 80.0 else page_height
    x_right_margin = page_width - 26.0 if page_width > 100.0 else page_width

    effective_exclude = exclude_rects
    if effective_exclude is None:
        auto_exclude: list[pymupdf.Rect] = []
        try:
            finder = page.find_tables()
            for t in (finder.tables if finder and finder.tables else []):
                tr = pymupdf.Rect(t.bbox)
                if (
                    (tr.x1 - tr.x0) >= page_width * 0.55
                    and tr.x0 < mid_x - 12.0
                    and tr.x1 > mid_x + 12.0
                ):
                    tbl_txt = _normalize_unicode_spaces(page.get_text("text", clip=tr))
                    if _is_coofficial_non_spanish_text(tbl_txt):
                        auto_exclude.append(tr)
        except Exception:
            pass
        effective_exclude = auto_exclude

    dict_data = page.get_text("dict")
    items: list[tuple[float, float, str]] = []

    for blk in dict_data.get("blocks", []):
        if blk.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = [float(v) for v in blk.get("bbox", (0, 0, 0, 0))]
        if by1 <= y_top or by0 >= y_bot or bx0 >= x_right_margin:
            continue
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if effective_exclude and any(
            r.x0 - 2.0 <= cx <= r.x1 + 2.0 and r.y0 - 2.0 <= cy <= r.y1 + 2.0
            for r in effective_exclude
        ):
            continue

        raw_lines = blk.get("lines", [])
        if not raw_lines:
            continue

        # Descarta bloque completo en columna izquierda
        if bx1 <= mid_x + 12.0:
            continue

        is_wide = (bx1 - bx0) >= page_width * 0.50 and bx0 < mid_x - 12.0 and bx1 > mid_x + 12.0
        if is_wide:
            # Detectar caso de fórmulas o columnas paralelas mezcladas por PyMuPDF en el mismo bloque (ej. izquierda euskera / derecha castellano)
            left_lines: list[tuple[float, float, str]] = []
            right_lines: list[tuple[float, float, str]] = []
            for ln in raw_lines:
                lx0, ly0, lx1, ly1 = [float(v) for v in ln["bbox"]]
                ltxt = _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
                if not ltxt or _BULLETIN_HEADER_NOISE.match(ltxt):
                    continue
                if lx1 <= mid_x - 5.0:
                    left_lines.append((ly0, lx0, ltxt))
                elif lx0 >= mid_x - 12.0:
                    right_lines.append((ly0, lx0, ltxt))
            if left_lines and right_lines and _is_coofficial_non_spanish_text(" ".join(t for _, _, t in left_lines)):
                kept_lines = [t for _, _, t in sorted(right_lines)]
                joined = "\n".join(kept_lines).strip()
                if joined:
                    items.append((by0, mid_x, joined))
                continue

            full_txt = "\n".join(
                _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
                for ln in raw_lines
            ).strip()
            if not full_txt or _is_coofficial_non_spanish_text(full_txt):
                continue

        clean_blk_lines: list[str] = []
        for ln in raw_lines:
            lx0, ly0 = float(ln["bbox"][0]), float(ln["bbox"][1])
            if lx0 >= x_right_margin:
                continue
            ltxt = _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
            if not ltxt:
                continue
            if _BULLETIN_HEADER_NOISE.match(ltxt):
                continue
            if re.fullmatch(r"\d{1,5}", ltxt) and (
                ly0 <= 60.0 or ly0 >= page_height - 55.0 or len(raw_lines) == 1
            ):
                continue
            clean_blk_lines.append(ltxt)

        if clean_blk_lines:
            joined_blk = "\n".join(clean_blk_lines)
            if _is_coofficial_non_spanish_text(joined_blk):
                continue
            items.append((by0, bx0, joined_blk))

    items.sort(key=lambda it: (round(it[0], 1), round(it[1], 1)))
    return "\n\n".join(txt for _, _, txt in items if txt.strip())


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
    text_sort = page.get_text("text", sort=True)
    clock_hours_re = re.compile(r"\b\d{1,2},\d{2}\s*h(?:oras?|rs?)?(?=\b|\.)", re.IGNORECASE)
    text_no_hours = clock_hours_re.sub("", text)
    lines = [ln.strip() for ln in text_sort.splitlines() if ln.strip()]
    multi_dec_lines = sum(
        1 for ln in lines if len(_SPANISH_DECIMALS_RE.findall(clock_hours_re.sub("", ln))) >= 2
    )
    if multi_dec_lines >= 3:
        return True
    # Respaldo para tablas salariales sin bordes vectoriales: distinguir listas tabulares cortas de prosa jurídica
    euro_count = text.count("€")
    decimal_amounts = len(_SPANISH_DECIMALS_RE.findall(text_no_hours))
    short_amount_lines = sum(
        1
        for ln in lines
        if len(ln) <= 85
        and re.search(
            r"(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}\s*(?:€|euros?)?\.?\s*$",
            ln,
            re.IGNORECASE,
        )
    )
    if euro_count < 6 and decimal_amounts < 8 and not (short_amount_lines >= 5 and len(text) < 2500):
        return False
    return short_amount_lines >= 5 or len(text) < 4500


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
                        or len(_SPANISH_DECIMALS_RE.findall(cell)) >= 2
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


def _extract_vector_table_md_by_clip(
    page: pymupdf.Page,
    table: object,
    *,
    bilingual_right_es: bool = False,
) -> str:
    """Reconstruye en CPU una tabla vectorial extrayendo texto por rectángulo de celda (`clip`) para evitar fusión de decimales."""
    table_rect = pymupdf.Rect(table.bbox)
    table_width = max(1.0, float(table_rect.width or 1.0))
    page_width = float(page.rect.width or 0.0)
    mid_x = page_width / 2.0

    # Si es tabla ancha bilingüe lado-a-lado (columnas izq en lengua cooficial y col der en castellano),
    # conserva solo las columnas derechas en castellano cuando estén claramente separadas por mid_x.
    valid_orig_cols: Optional[list[int]] = None
    if (
        bilingual_right_es
        and page_width > 0.0
        and table_width >= page_width * 0.55
        and table_rect.x0 < mid_x - 12.0
        and table_rect.x1 > mid_x + 12.0
        and table.rows
    ):
        ncols_probe = max((len(r.cells) for r in table.rows), default=0)
        if ncols_probe >= 3:
            col_cxs: dict[int, list[float]] = {c: [] for c in range(ncols_probe)}
            col_texts: dict[int, list[str]] = {c: [] for c in range(ncols_probe)}
            for r in table.rows:
                for ci, cb in enumerate(r.cells):
                    if cb is None:
                        continue
                    cr = pymupdf.Rect(cb)
                    ctxt = _normalize_unicode_spaces(page.get_text("text", clip=cr)).strip()
                    if ctxt:
                        col_cxs[ci].append((cr.x0 + cr.x1) / 2.0)
                        col_texts[ci].append(ctxt)
            left_cols = [
                ci for ci in range(ncols_probe)
                if col_cxs[ci] and (sum(col_cxs[ci]) / len(col_cxs[ci])) < mid_x - 10.0
            ]
            right_cols = [
                ci for ci in range(ncols_probe)
                if col_cxs[ci] and (sum(col_cxs[ci]) / len(col_cxs[ci])) >= mid_x - 15.0
            ]
            if left_cols and right_cols:
                left_blob = " ".join(t for ci in left_cols for t in col_texts[ci])
                right_blob = " ".join(t for ci in right_cols for t in col_texts[ci])
                if _is_coofficial_non_spanish_text(left_blob) and not _is_coofficial_non_spanish_text(right_blob):
                    valid_orig_cols = right_cols

    raw_rows: list[list[str]] = []
    raw_row_segs: list[list[list[tuple[float, str]]]] = []
    cell_x_bounds: list[tuple[float, float]] = []
    row_colspans: list[bool] = []
    for row in table.rows:
        orig_cells = (
            [row.cells[ci] for ci in valid_orig_cols]
            if valid_orig_cols is not None
            else list(row.cells)
        )
        is_colspan_row = (
            len(orig_cells) >= 2
            and orig_cells[0] is not None
            and all(c is None for c in orig_cells[1:])
            and float(pymupdf.Rect(orig_cells[0]).width) >= (table_width * 0.70 if valid_orig_cols is None else table_width * 0.30)
        )
        col_lines: list[list[tuple[int, str, list[tuple[float, str]]]]] = []
        any_multi = False
        for c_pos, cb in enumerate(orig_cells):
            if cb is None:
                col_lines.append([])
                continue
            cell_rect = pymupdf.Rect(cb)
            while len(cell_x_bounds) <= c_pos:
                cell_x_bounds.append((float(cell_rect.x0), float(cell_rect.x1)))
            old_x0, old_x1 = cell_x_bounds[c_pos]
            cell_x_bounds[c_pos] = (min(old_x0, float(cell_rect.x0)), max(old_x1, float(cell_rect.x1)))

            d = page.get_text("dict", clip=cell_rect)
            y_buckets: dict[int, list[tuple[float, float, str]]] = {}
            for blk in d.get("blocks", []):
                if blk.get("type") != 0:
                    continue
                for ln in blk.get("lines", []):
                    lx0, ly0, lx1, ly1 = ln["bbox"]
                    cx = (lx0 + lx1) / 2.0
                    cy = (ly0 + ly1) / 2.0
                    if not (cell_rect.x0 <= cx <= cell_rect.x1 and cell_rect.y0 <= cy <= cell_rect.y1):
                        continue
                    raw_txt = _normalize_unicode_spaces(
                        "".join(sp.get("text", "") for sp in ln.get("spans", []))
                    )
                    txt = raw_txt.strip()
                    if not txt:
                        continue
                    eff_lx0 = float(lx0)
                    lead_ws = len(raw_txt) - len(raw_txt.lstrip())
                    if lead_ws >= 4 and len(raw_txt) > lead_ws:
                        eff_lx0 = float(lx0) + (lead_ws / len(raw_txt)) * max(0.0, float(lx1) - float(lx0))
                    if bilingual_right_es:
                        stripped_dots = re.sub(r"\.{3,}", " ", txt).strip()
                        if not stripped_dots:
                            continue
                        txt = stripped_dots
                    y_key = round(float(ly0) / 3.0) * 3
                    y_buckets.setdefault(y_key, []).append((eff_lx0, float(lx1), txt))
            ordered: list[tuple[int, str, list[tuple[float, str]]]] = []
            for yk in sorted(y_buckets):
                segs_raw = sorted(y_buckets[yk], key=lambda item: item[0])
                segs_clean: list[tuple[float, str]] = []
                for sx0, _, stxt in segs_raw:
                    # Divide un mismo span que mezcle "etiqueta    valor" separados por líder de puntos o >=4 espacios
                    if bilingual_right_es and re.search(r"\S(?:\.{2,}|\s{4,})\S", stxt):
                        parts = [p.strip() for p in re.split(r"\s{3,}", re.sub(r"\.{2,}", "    ", stxt)) if p.strip()]
                        for pi, pt in enumerate(parts):
                            segs_clean.append((sx0 + (pi * 80.0), pt))
                    else:
                        segs_clean.append((sx0, stxt))
                joined_line = " ".join(t for _, _, t in segs_raw).strip()
                if bilingual_right_es:
                    joined_line = re.sub(r"\s{2,}", " ", joined_line)
                ordered.append((yk, joined_line, segs_clean))
            if len(ordered) >= 2:
                any_multi = True
            col_lines.append(ordered)

        all_ys = sorted({yk for cl in col_lines for yk, _, _ in cl})
        y_clusters: list[list[int]] = []
        for y in all_ys:
            if not y_clusters or y - y_clusters[-1][-1] > 5:
                y_clusters.append([y])
            else:
                y_clusters[-1].append(y)

        multi_col_clusters = sum(
            1
            for clus in y_clusters
            if sum(1 for cl in col_lines if any(yk in clus for yk, _, _ in cl)) >= 2
        )

        if any_multi and multi_col_clusters >= 2:
            for clus in y_clusters:
                cset = set(clus)
                sub = [" ".join(t for yk, t, _ in cl if yk in cset).strip() for cl in col_lines]
                sub_segs = [
                    [sg for yk, _, sgs in cl if yk in cset for sg in sgs]
                    for cl in col_lines
                ]
                if any(sub):
                    raw_rows.append(sub)
                    raw_row_segs.append(sub_segs)
                    row_colspans.append(False)
        else:
            joined = [" ".join(t for _, t, _ in cl).strip() for cl in col_lines]
            joined_segs = [[sg for _, _, sgs in cl for sg in sgs] for cl in col_lines]
            if any(joined):
                raw_rows.append(joined)
                raw_row_segs.append(joined_segs)
                row_colspans.append(is_colspan_row)

    if not raw_rows:
        return ""

    # Divide columnas colapsadas por líder de puntos cuando >=2 sub-columnas están alineadas horizontalmente (ej. Nivel ... Total mes (€))
    if bilingual_right_es and len(raw_rows) >= 3:
        ncols_tmp = max(len(r) for r in raw_rows)
        new_cols_per_row: list[list[str]] = [[] for _ in raw_rows]
        for ci in range(ncols_tmp):
            col_segs_rows = [
                raw_row_segs[ri][ci] if ri < len(raw_row_segs) and ci < len(raw_row_segs[ri]) else []
                for ri in range(len(raw_rows))
            ]
            two_seg_rows = [sgs for sgs in col_segs_rows if len(sgs) == 2 and (sgs[1][0] - sgs[0][0]) >= 35.0]
            can_split = (
                len(two_seg_rows) >= max(2, len(raw_rows) // 2)
                and ci < len(cell_x_bounds)
                and (cell_x_bounds[ci][1] - cell_x_bounds[ci][0]) >= 90.0
            )
            if can_split:
                left_max_x = max(sgs[0][0] for sgs in two_seg_rows)
                right_min_x = min(sgs[1][0] for sgs in two_seg_rows)
                split_x = (left_max_x + right_min_x) / 2.0
                for ri, row in enumerate(raw_rows):
                    sgs = col_segs_rows[ri]
                    val = row[ci] if ci < len(row) else ""
                    if row_colspans[ri] and ci == 0 and all(not c for c in row[1:]):
                        new_cols_per_row[ri].extend([val, ""])
                    elif len(sgs) >= 2 and (sgs[-1][0] - sgs[0][0]) >= 35.0:
                        l_part = " ".join(t for x, t in sgs if x < split_x).strip()
                        r_part = " ".join(t for x, t in sgs if x >= split_x).strip()
                        new_cols_per_row[ri].extend([l_part, r_part])
                    elif len(sgs) == 1:
                        if sgs[0][0] >= split_x:
                            new_cols_per_row[ri].extend(["", sgs[0][1]])
                        else:
                            new_cols_per_row[ri].extend([sgs[0][1], ""])
                    else:
                        new_cols_per_row[ri].extend([val, ""])
            else:
                for ri, row in enumerate(raw_rows):
                    new_cols_per_row[ri].append(row[ci] if ci < len(row) else "")
        raw_rows = new_cols_per_row

    # En tablas bilingües anchos completos que apilan euskera arriba y castellano abajo (ej. Anexo I P31),
    # descarta el bloque inicial en lengua cooficial o toda la tabla si es íntegramente no castellana.
    if bilingual_right_es:
        non_empty_rows_txt = [" ".join(r) for r in raw_rows if any(r)]
        if non_empty_rows_txt and _is_coofficial_non_spanish_text(non_empty_rows_txt[0]):
            spanish_start_idx: Optional[int] = None
            for ri, r in enumerate(raw_rows):
                r_txt = " ".join(r).strip()
                if not r_txt or _is_coofficial_non_spanish_text(r_txt):
                    continue
                es_hits = sum(
                    1
                    for tok in re.findall(r"[a-záéíóúüñ]{2,}", r_txt.lower())
                    if tok in _SPANISH_LANGUAGE_TOKENS
                )
                if es_hits >= 2:
                    spanish_start_idx = ri
                    break
            if spanish_start_idx is None:
                return ""
            raw_rows = raw_rows[spanish_start_idx:]
            row_colspans = row_colspans[spanish_start_idx:]
        else:
            whole_tbl_txt = " ".join(non_empty_rows_txt)
            if _is_coofficial_non_spanish_text(whole_tbl_txt):
                return ""

    ncols = max(len(r) for r in raw_rows)
    padded = [r + [""] * (ncols - len(r)) for r in raw_rows]
    keep_cols = [c for c in range(ncols) if any(r[c] for r in padded)]
    if not keep_cols:
        return ""
    trimmed = [[r[c] for c in keep_cols] for r in padded]

    def _esc(val: str) -> str:
        return val.replace("|", "/").strip()

    banner_headings: list[str] = []
    orig_row_idx = 0
    while (
        len(trimmed) >= 2
        and (orig_row_idx == 0 or row_colspans[0])
        and len(trimmed[0]) >= 2
        and trimmed[0][0]
        and all(not c for c in trimmed[0][1:])
        and not (
            not row_colspans[0]
            and any(_SPANISH_DECIMALS_RE.search(c) for c in trimmed[1][1:])
        )
    ):
        banner_headings.append(f"### {_esc(trimmed[0][0])}")
        trimmed = trimmed[1:]
        row_colspans = row_colspans[1:]
        orig_row_idx += 1

    banner_heading = ("\n\n".join(banner_headings) + "\n\n") if banner_headings else ""

    def _merge_header_rows(top_row: list[str], sub_row: list[str], default_col0: str = "") -> list[str]:
        merged = [top_row[0] or sub_row[0] or default_col0]
        last_top = ""
        for col_idx, sub_h in enumerate(sub_row[1:], start=1):
            if col_idx < len(top_row) and top_row[col_idx]:
                last_top = top_row[col_idx]
            if last_top and len(last_top) > 2:
                merged.append(f"{last_top} ({sub_h})" if sub_h else last_top)
            elif last_top:
                merged.append(f"{last_top} {sub_h}".strip())
            else:
                merged.append(sub_h)
        return merged

    if len(trimmed) >= 2 and not trimmed[1][0] and all(trimmed[1][1:]):
        hdr = _merge_header_rows(trimmed[0], trimmed[1])
        body = trimmed[2:]
    elif (
        len(trimmed) >= 2
        and not trimmed[0][0]
        and trimmed[1][0]
        and all(trimmed[0][1:])
        and all(trimmed[1][1:])
        and not _SPANISH_DECIMALS_RE.search(" ".join(trimmed[1]))
    ):
        hdr = _merge_header_rows(trimmed[0], trimmed[1])
        body = trimmed[2:]
    elif (
        len(trimmed) >= 3
        and not trimmed[0][0]
        and any(trimmed[0][1:])
        and trimmed[1][0]
        and all(not c for c in trimmed[1][1:])
        and not trimmed[2][0]
        and all(trimmed[2][1:])
    ):
        hdr = _merge_header_rows(trimmed[0], trimmed[2], default_col0="Categoría / Nivel")
        body = [trimmed[1], *trimmed[3:]]
    elif (
        len(trimmed) >= 2
        and any(not c for c in trimmed[0])
        and all(trimmed[0][i] or trimmed[1][i] for i in range(len(trimmed[0])))
        and not _SPANISH_DECIMALS_RE.search(" ".join(trimmed[1]))
        and not re.fullmatch(r"[IVX]+|\d+", trimmed[1][0].strip())
        and sum(1 for c in trimmed[1] if c) >= max(2, len(trimmed[0]) // 2)
    ):
        hdr = [f"{trimmed[0][i]} {trimmed[1][i]}".strip() for i in range(len(trimmed[0]))]
        body = trimmed[2:]
    elif banner_headings:
        hdr = [f"Columna {idx + 1}" for idx in range(len(trimmed[0]))]
        body = trimmed
    else:
        hdr = list(trimmed[0])
        body = trimmed[1:]

    # Rellenar cabeceras vacías en columnas [1:] para evitar SPLIT_COLSPAN_HEADER manteniendo nombres distintos
    if bilingual_right_es:
        seen_headers: set[str] = {_esc(hdr[0]).lower()} if hdr else set()
        for ci in range(1, len(hdr)):
            if not _esc(hdr[ci]):
                col_has_dec = any(_SPANISH_DECIMALS_RE.search(r[ci]) for r in body if ci < len(r))
                candidate = "Importe / Valor" if (col_has_dec and ci == 1) else f"Columna {ci + 1}"
                if candidate.lower() in seen_headers:
                    candidate = f"Observaciones ({ci + 1})"
                hdr[ci] = candidate
            seen_headers.add(_esc(hdr[ci]).lower())

        # Reagrupa filas de descripción partidas verticalmente cuando la fila i carece de valor y la fila i+1 aporta el importe
        merged_body: list[list[str]] = []
        bi = 0
        while bi < len(body):
            row = list(body[bi])
            if (
                len(row) >= 2
                and row[0]
                and all(not c for c in row[1:])
                and bi + 1 < len(body)
                and any(body[bi + 1][1:])
                and (
                    not body[bi + 1][0]
                    or row[0].endswith("-")
                    or body[bi + 1][0][:1].islower()
                )
            ):
                nxt = body[bi + 1]
                if nxt[0]:
                    if row[0].endswith("-"):
                        row[0] = f"{row[0][:-1]}{nxt[0]}".strip()
                    else:
                        row[0] = f"{row[0]} {nxt[0]}".strip()
                for ci in range(1, len(row)):
                    row[ci] = nxt[ci] if ci < len(nxt) else ""
                merged_body.append(row)
                bi += 2
                continue
            merged_body.append(row)
            bi += 1
        body = merged_body

    md_lines = [
        "| " + " | ".join(_esc(c) for c in hdr) + " |",
        "| " + " | ".join(["---"] * len(hdr)) + " |",
    ]
    for r in body:
        md_lines.append("| " + " | ".join(_esc(c) for c in r) + " |")
    return banner_heading + "\n".join(md_lines)


def _format_dotted_leader_block_to_md(block_text: str) -> Optional[str]:
    """Convierte un bloque textual con líneas de puntos (`..... importe`) en tabla Markdown GFM limpia."""
    raw_lines = [_normalize_unicode_spaces(ln).strip() for ln in block_text.splitlines() if ln.strip()]
    if len(raw_lines) < 2:
        return None
    dotted_count = sum(1 for ln in raw_lines if re.search(r"\.{3,}", ln))
    if dotted_count < 2:
        return None

    # Reensambla líneas partidas justo antes de puntos (p. ej. "Muerte o invalidez ... enfermedad ..\n...... 8.017")
    joined_lines: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        curr = raw_lines[idx]
        if (
            idx + 1 < len(raw_lines)
            and not re.search(r"(?:\d{1,3}(?:\.\d{3})*,\d{2}|\b\d{1,3}(?:\.\d{3})+\b|[–\-—])\s*$", curr)
            and re.match(r"^\.{2,}", raw_lines[idx + 1])
        ):
            nxt = raw_lines[idx + 1]
            curr_clean = re.sub(r"\.{2,}$", "", curr).rstrip("- ").strip() if curr.endswith("-") else f"{curr.rstrip('. ')} "
            joined_lines.append(f"{curr_clean}{nxt}".strip())
            idx += 2
            continue
        joined_lines.append(curr)
        idx += 1

    header_tokens: list[str] = []
    parsed_rows: list[list[str]] = []
    trailing_lines: list[str] = []

    row_re = re.compile(
        r"^(.+?)\s*\.{3,}\s*"
        r"(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|[–\-—]|0\b)(?:\s+(\d{1,3}(?:\.\d{3})*(?:,\d{2})?))?\s*$"
    )

    for ln in joined_lines:
        m = row_re.match(ln)
        if m:
            lbl = m.group(1).strip(" .")
            v1 = m.group(2).strip()
            v2 = (m.group(3) or "").strip()
            if v2:
                parsed_rows.append([lbl, v1, v2])
            else:
                parsed_rows.append([lbl, v1])
        elif not parsed_rows:
            header_tokens.append(ln)
        else:
            trailing_lines.append(ln)

    if len(parsed_rows) < 2:
        return None

    ncols = max(len(r) for r in parsed_rows)
    for r in parsed_rows:
        while len(r) < ncols:
            r.append("")

    # Intenta extraer cabecera de la última línea previa si tiene el mismo número de columnas separadas por >=2 espacios
    hdr: Optional[list[str]] = None
    extra_intro: list[str] = list(header_tokens)
    if header_tokens:
        last_hdr = header_tokens[-1]
        parts = [p.strip() for p in re.split(r"\s{2,}", last_hdr) if p.strip()]
        if len(parts) == ncols:
            hdr = parts
            extra_intro = header_tokens[:-1]
        elif ncols == 2 and len(parts) == 1 and re.search(r"(?i)(?:nivel|categor|modalidad|concepto|valor|mes|trienio|hora|€)", parts[0]):
            if re.search(r"(?i)nivel", parts[0]):
                hdr = [parts[0], "Importe (€)"]
            else:
                hdr = ["Concepto / Modalidad", parts[0]]
            extra_intro = header_tokens[:-1]
        elif ncols == 2 and len(header_tokens) >= 2:
            # Cabecera en 2 líneas cortas (ej. "2026 valor" / "mes (€/mes)")
            tail_joined = " ".join(header_tokens[-2:]).strip()
            if len(tail_joined) <= 45 and not tail_joined.endswith("."):
                hdr = ["Concepto / Modalidad", tail_joined]
                extra_intro = header_tokens[:-2]

    if hdr is None:
        hdr = ["Concepto / Nivel", "Valor", "Total"][:ncols]

    md_parts: list[str] = []
    if extra_intro:
        md_parts.append(_format_plain_page_to_markdown("\n".join(extra_intro)))
    tbl_lines = [
        "| " + " | ".join(h.replace("|", "/").strip() for h in hdr) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for r in parsed_rows:
        tbl_lines.append("| " + " | ".join(c.replace("|", "/").strip() for c in r) + " |")
    md_parts.append("\n".join(tbl_lines))
    if trailing_lines:
        md_parts.append(_format_plain_page_to_markdown("\n".join(trailing_lines)))
    return "\n\n".join(p for p in md_parts if p.strip())


def _extract_bilingual_right_es_page_cpu(page: pymupdf.Page) -> str:
    """Extrae en CPU texto y tablas exclusivamente en castellano de boletines bilingües (ej. BOG/BOTHA)."""
    page_width = float(page.rect.width or 0.0)
    page_height = float(page.rect.height or 0.0)
    mid_x = page_width / 2.0
    y_top = 34.0
    y_bot = page_height - 24.0 if page_height > 80.0 else page_height
    x_right_margin = page_width - 26.0 if page_width > 100.0 else page_width

    try:
        finder = page.find_tables()
        raw_tables = list(finder.tables) if finder and finder.tables else []
    except Exception:
        raw_tables = []

    # Descarta marcos contenedores externos (<=3 filas) que envuelven otra tabla interna más detallada
    all_rects = [pymupdf.Rect(t.bbox) for t in raw_tables]
    filtered_tables: list[tuple[object, pymupdf.Rect]] = []
    for idx_t, (t, tr) in enumerate(zip(raw_tables, all_rects)):
        if t.row_count <= 3 and any(
            idx_o != idx_t
            and raw_tables[idx_o].row_count > t.row_count
            and tr.x0 - 6.0 <= o_r.x0
            and tr.y0 - 6.0 <= o_r.y0
            and o_r.x1 <= tr.x1 + 6.0
            and o_r.y1 <= tr.y1 + 6.0
            for idx_o, o_r in enumerate(all_rects)
        ):
            continue
        filtered_tables.append((t, tr))

    # Todas las tablas detectadas (también las descartadas en euskera) bloquean el texto plano suelto en su interior
    all_detected_rects = [tr for _, tr in filtered_tables]
    items: list[tuple[float, float, str]] = []

    for table, rect in filtered_tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        if page_width > 0.0 and rect.x1 <= mid_x + 12.0:
            # Tabla íntegramente en la columna izquierda (lengua cooficial)
            continue
        is_wide = (rect.x1 - rect.x0) >= page_width * 0.55 and rect.x0 < mid_x - 12.0 and rect.x1 > mid_x + 12.0
        table_md = _extract_vector_table_md_by_clip(page, table, bilingual_right_es=True)
        if not table_md or "| ---" not in table_md:
            continue
        if is_wide and _is_coofficial_non_spanish_text(table_md):
            continue
        items.append((rect.y0, rect.x0, table_md))

    # Procesar los bloques textuales fuera de las tablas vectoriales
    dict_data = page.get_text("dict")
    for blk in dict_data.get("blocks", []):
        if blk.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = [float(v) for v in blk.get("bbox", (0, 0, 0, 0))]
        if by1 <= y_top or by0 >= y_bot or bx0 >= x_right_margin or bx1 <= mid_x + 12.0:
            continue
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if any(
            tr.x0 - 2.0 <= cx <= tr.x1 + 2.0 and tr.y0 - 2.0 <= cy <= tr.y1 + 2.0
            for tr in all_detected_rects
        ):
            continue

        raw_lines = blk.get("lines", [])
        if not raw_lines:
            continue

        is_wide = (bx1 - bx0) >= page_width * 0.50 and bx0 < mid_x - 12.0 and bx1 > mid_x + 12.0
        if is_wide:
            left_lines: list[tuple[float, str]] = []
            right_lines: list[tuple[float, str]] = []
            for ln in raw_lines:
                lx0, ly0, lx1, _ = [float(v) for v in ln["bbox"]]
                ltxt = _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
                if not ltxt or _BULLETIN_HEADER_NOISE.match(ltxt):
                    continue
                if lx1 <= mid_x - 5.0:
                    left_lines.append((ly0, ltxt))
                elif lx0 >= mid_x - 12.0:
                    right_lines.append((ly0, ltxt))
            if left_lines and right_lines and _is_coofficial_non_spanish_text(" ".join(t for _, t in left_lines)):
                block_text = "\n".join(t for _, t in sorted(right_lines))
            else:
                block_text = "\n".join(
                    _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
                    for ln in raw_lines
                )
                if _is_coofficial_non_spanish_text(block_text):
                    continue
        else:
            block_lines_clean: list[str] = []
            for ln in raw_lines:
                lx0, ly0 = float(ln["bbox"][0]), float(ln["bbox"][1])
                if lx0 >= x_right_margin:
                    continue
                ltxt = _normalize_unicode_spaces("".join(sp.get("text", "") for sp in ln.get("spans", []))).strip()
                if not ltxt or _BULLETIN_HEADER_NOISE.match(ltxt):
                    continue
                if re.fullmatch(r"\d{1,5}", ltxt) and (
                    ly0 <= 60.0 or ly0 >= page_height - 55.0 or len(raw_lines) == 1
                ):
                    continue
                block_lines_clean.append(ltxt)
            block_text = "\n".join(block_lines_clean)

        if not block_text.strip() or _is_coofficial_non_spanish_text(block_text):
            continue

        dotted_md = _format_dotted_leader_block_to_md(block_text)
        if dotted_md:
            items.append((by0, bx0, dotted_md))
        else:
            formatted = _format_plain_page_to_markdown(block_text).strip()
            if formatted:
                items.append((by0, bx0, formatted))

    items.sort(key=lambda it: (round(it[0], 1), round(it[1], 1)))
    out = "\n\n".join(content for _, _, content in items if content).strip()
    return _normalize_spanish_thousands(out)


def _extract_vector_tables_page_cpu(
    page: pymupdf.Page,
    layout: str = "single_column",
) -> Optional[str]:
    """Extrae en CPU páginas mixtas (texto + tablas vectoriales) usando `clip` por celda y guardrail de paridad."""
    try:
        finder = page.find_tables()
    except Exception:
        return None
    tables = (
        [t for t in finder.tables if t.row_count >= 3 and t.col_count >= 2]
        if finder and finder.tables
        else []
    )
    if not tables:
        return None

    page_width = float(page.rect.width or 0.0)
    mid_x = page_width / 2.0
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    raw_items: list[tuple[pymupdf.Rect, bool, str]] = []
    for table, rect in zip(tables, table_rects):
        table_md = _extract_vector_table_md_by_clip(page, table)
        if table_md:
            is_wide = (
                layout == "double_column"
                and page_width > 0.0
                and (rect.x1 - rect.x0) >= page_width * 0.55
                and rect.x0 < mid_x - 15.0
                and rect.x1 > mid_x + 15.0
            )
            raw_items.append((rect, is_wide, table_md))

    for blk in page.get_text("blocks"):
        if len(blk) < 7 or blk[6] != 0 or not str(blk[4]).strip():
            continue
        br = pymupdf.Rect(float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]))
        cx, cy = (br.x0 + br.x1) / 2.0, (br.y0 + br.y1) / 2.0
        if any(
            tr.x0 - 2.0 <= cx <= tr.x1 + 2.0 and tr.y0 - 2.0 <= cy <= tr.y1 + 2.0
            for tr in table_rects
        ):
            continue
        plain_md = _format_plain_page_to_markdown(str(blk[4]).strip())
        if not plain_md.strip():
            continue
        is_wide = (
            layout == "double_column"
            and page_width > 0.0
            and (br.x1 - br.x0) >= page_width * 0.55
            and br.x0 < mid_x - 15.0
            and br.x1 > mid_x + 15.0
        )
        raw_items.append((br, is_wide, plain_md.strip()))

    if not raw_items:
        return None

    if layout == "double_column" and page_width > 0.0:
        col_rects = [rect for rect, is_wide, _ in raw_items if not is_wide]
        col_top_y = min((r.y0 for r in col_rects), default=float("inf"))
        col_bottom_y = max((r.y1 for r in col_rects), default=float("-inf"))

        def _sort_key(item: tuple[pymupdf.Rect, bool, str]) -> tuple[int, float, float]:
            rect, is_wide, _ = item
            if is_wide:
                if rect.y1 <= col_top_y + 5.0:
                    group = 0
                elif rect.y0 >= col_bottom_y - 5.0:
                    group = 4
                else:
                    group = 2
            else:
                group = 1 if (rect.x0 + rect.x1) / 2.0 < mid_x else 3
            return (group, round(float(rect.y0), 1), round(float(rect.x0), 1))

        ordered_items = sorted(raw_items, key=_sort_key)
    else:
        ordered_items = sorted(
            raw_items,
            key=lambda item: (round(float(item[0].y0), 1), round(float(item[0].x0), 1)),
        )

    out = "\n\n".join(content for _, _, content in ordered_items if content).strip()
    if "| ---" not in out:
        return None
    missing_total, _, _ = _count_missing_spanish_decimals(page.get_text("text"), out)
    if missing_total > 0:
        return None
    return _normalize_spanish_thousands(out)


def _markdown_table_is_healthy(md_text: str) -> bool:
    """Verifica determinísticamente que ninguna tabla Markdown extraída tenga columnas colapsadas o cabeceras rotas."""
    lines = (md_text or "").splitlines()
    idx = 0
    found_table = False
    has_context_above = False
    while idx < len(lines):
        if idx + 1 < len(lines):
            hdr_ln = lines[idx].strip()
            sep_ln = lines[idx + 1].strip()
            if hdr_ln.startswith("|") and sep_ln.startswith("|") and "---" in sep_ln:
                found_table = True
                headers = [c.strip() for c in hdr_ln.strip("|").split("|")]
                if not (headers[0] if headers else "").strip() and not has_context_above:
                    return False
                if "columna 1" in [h.lower() for h in headers] and len(headers) >= 2:
                    return False
                if any(not h for h in headers[1:]):
                    return False
                if _SPANISH_DECIMALS_RE.search(hdr_ln):
                    return False
                cursor = idx + 2
                while cursor < len(lines):
                    row_ln = lines[cursor].strip()
                    if not row_ln.startswith("|") or "---" in row_ln:
                        break
                    cells = [c.strip() for c in row_ln.strip("|").split("|")]
                    if len(cells) != len(headers):
                        return False
                    col0 = cells[0] if cells else ""
                    if len(_SPANISH_DECIMALS_RE.findall(col0)) >= 2:
                        return False
                    if (
                        len(cells) >= 2
                        and _SPANISH_DECIMALS_RE.search(col0)
                        and re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}", col0)
                        and any(_SPANISH_DECIMALS_RE.search(c) for c in cells[1:])
                    ):
                        return False
                    if re.search(
                        r"(?i)\b(?:SALARIO\s+BASE\s+(?:MENSUAL|MANSUAL|ANUAL)|HORAS?\s+EXTRAS?)\s+20[23]\d\b",
                        row_ln,
                    ):
                        return False
                    cursor += 1
                has_context_above = False
                idx = cursor
                continue
        if lines[idx].strip():
            has_context_above = True
        idx += 1
    return found_table


def _parse_coordinate_scale_headers(line: str) -> Optional[list[str]]:
    """Detecta cabeceras multi-columna típicas de escalas hosteleras (estrellas, tenedores, tazas, categorías)."""
    clean = _normalize_unicode_spaces(line).strip()
    if not clean:
        return None
    if re.fullmatch(r"(?i)SALARIO\s+BASE(?:\s+ANUAL|\s+MENSUAL)?", clean):
        return ["Categoría profesional", "Hora extra (€ / Salario base)"]

    scale_pat = re.compile(
        r"(?i)\b\d+\s*\*|\b\d+\s*TAZAS?\b|\b(?:LUXO|PRIMEIRA|SEGUNDA|TERCEIRA|CUARTA|ESPECIAL|PRIMERA|TERCERA)\b"
    )
    matches = scale_pat.findall(clean)
    if len(matches) < 2:
        return None
    remainder = re.sub(r"[\s,|/.-]+", "", scale_pat.sub("", clean))
    if remainder:
        return None

    normalized_cols: list[str] = []
    for tok in matches:
        t = re.sub(r"\s+", " ", tok).strip()
        t = re.sub(r"^(\d+)\s+\*$", r"\1*", t)
        t = re.sub(r"^(\d+)(TAZAS?)$", r"\1 \2", t, flags=re.IGNORECASE)
        if t not in normalized_cols:
            normalized_cols.append(t)
    return ["Categoría profesional", *normalized_cols] if len(normalized_cols) >= 2 else None


def _extract_coordinate_table_stream_cpu(
    doc: pymupdf.Document,
    page_indices: list[int],
) -> Optional[str]:
    """Reconstruye en CPU tablas salariales continuas multi-página usando coordenadas `(y0, x0)` de PyMuPDF."""
    if not page_indices:
        return None

    ordered_lines: list[str] = []
    pdf_raw_chunks: list[str] = []
    for p_idx in page_indices:
        page = doc[p_idx]
        pdf_raw_chunks.append(page.get_text("text"))
        pw = float(page.rect.width or 595.0)
        ph = float(page.rect.height or 842.0)
        d = page.get_text("dict")
        rows_by_y: dict[float, list[tuple[float, str]]] = {}
        for blk in d.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                x0, y0, _x1, y1 = ln["bbox"]
                if pw > 0.0 and x0 > pw * 0.90 and ph > pw:
                    continue
                txt = _normalize_unicode_spaces(
                    "".join(sp.get("text", "") for sp in ln.get("spans", []))
                ).strip()
                if not txt:
                    continue
                if _BULLETIN_HEADER_NOISE.match(txt) or (
                    (y0 < 85.0 or y1 > ph - 45.0) and re.search(r"(?i)p[áa]gina|bop|número", txt)
                ):
                    continue
                matched_y = next((ky for ky in rows_by_y if abs(ky - y0) <= 2.5), None)
                if matched_y is None:
                    matched_y = float(y0)
                    rows_by_y[matched_y] = []
                rows_by_y[matched_y].append((float(x0), txt))

        for y_key in sorted(rows_by_y):
            joined = " ".join(
                seg for _, seg in sorted(rows_by_y[y_key], key=lambda item: item[0])
            )
            joined = re.sub(r"\s{2,}", " ", joined).strip()
            if not joined or _BULLETIN_HEADER_NOISE.match(joined):
                continue
            # Reensambla líneas partidas de Área Funcional (ej. '... Pista' + 'para Catering')
            if (
                ordered_lines
                and re.match(r"(?i)^[ÁA]REA\s+FUNCIONAL\b", ordered_lines[-1])
                and not _SPANISH_DECIMALS_RE.search(joined)
                and joined.strip() != "CATERING"
                and not re.match(
                    r"(?i)^(?:GRUPO\b|SECCI[ÓO]N\b|SALARIO\b|HORAS\b|BARRAS\b|T[ÁA]BOAS\b)",
                    joined,
                )
                and _parse_coordinate_scale_headers(joined) is None
                and len(joined) <= 45
            ):
                ordered_lines[-1] = f"{ordered_lines[-1]} {joined}".strip()
                continue
            ordered_lines.append(joined)

    if not ordered_lines:
        return None

    out_blocks: list[str] = []
    active_banner: Optional[str] = None
    active_section: Optional[str] = None
    active_cols: Optional[list[str]] = None
    table_rows: list[list[str]] = []
    pending_subheads: list[str] = []

    def _esc(val: str) -> str:
        return val.replace("|", "/").strip()

    def _flush_open_table() -> None:
        nonlocal table_rows, pending_subheads
        if not table_rows or not active_cols:
            table_rows = []
            pending_subheads = []
            return
        title_parts = [p for p in (active_banner, active_section) if p]
        heading = " - ".join(title_parts) if title_parts else "Tabla Salarial"
        ncols = len(active_cols)
        md_tbl = [
            f"### {_esc(heading)}",
            "",
            "| " + " | ".join(_esc(c) for c in active_cols) + " |",
            "| " + " | ".join(["---"] * ncols) + " |",
        ]
        for row in table_rows:
            padded = (row + [""] * ncols)[:ncols]
            md_tbl.append("| " + " | ".join(_esc(c) for c in padded) + " |")
        out_blocks.append("\n".join(md_tbl))
        table_rows = []
        pending_subheads = []

    for line in ordered_lines:
        dec_matches = _SPANISH_DECIMALS_RE.findall(line)
        if dec_matches:
            first_m = _SPANISH_DECIMALS_RE.search(line)
            label = line[: first_m.start()].strip(" -–:|") if first_m else ""
            if not label:
                return None
            if active_cols is None:
                if len(dec_matches) == 1:
                    active_cols = ["Categoría profesional", "Importe (€)"]
                else:
                    active_cols = ["Categoría profesional"] + [
                        f"Categoría {i + 1}" for i in range(len(dec_matches))
                    ]
            if len(dec_matches) != len(active_cols) - 1:
                return None
            for sub in pending_subheads:
                table_rows.append([f"**{sub}**"] + [""] * (len(active_cols) - 1))
            pending_subheads.clear()
            table_rows.append([label, *dec_matches])
            continue

        if re.fullmatch(r"(?i)t[áa]boas?\s+salaria(?:is|les)\.?", line):
            _flush_open_table()
            out_blocks.append("## TÁBOAS SALARIAIS")
        elif re.match(
            r"(?i)^(?:SALARIO\s+BASE\s+(?:MENSUAL|MANSUAL|ANUAL)|HORAS?\s+EXTRAS?)\s+20[23]\d\b",
            line,
        ):
            _flush_open_table()
            active_banner = re.sub(r"(?i)\bMANSUAL\b", "MENSUAL", line).strip()
            active_cols = None
        elif re.match(r"(?i)^SECCI[ÓO]N\s+\d+[ªa]?\b", line) or line.strip() in (
            "CATERING",
            "BARRAS AMERICANAS",
        ):
            if active_banner is None:
                out_blocks.append(f"- {line.strip()}")
            else:
                _flush_open_table()
                active_section = line.strip()
                active_cols = None
        elif (parsed_cols := _parse_coordinate_scale_headers(line)) is not None:
            if table_rows:
                _flush_open_table()
            active_cols = parsed_cols
        else:
            if table_rows and active_cols:
                table_rows.append([f"**{line}**"] + [""] * (len(active_cols) - 1))
            elif active_banner is not None:
                pending_subheads.append(line)
            else:
                out_blocks.append(line)

    _flush_open_table()
    result = "\n\n".join(blk for blk in out_blocks if blk.strip()).strip()
    if "| ---" not in result or not _markdown_table_is_healthy(result):
        return None
    missing_total, _, _ = _count_missing_spanish_decimals("\n".join(pdf_raw_chunks), result)
    if missing_total > 0:
        return None
    return _normalize_spanish_thousands(result)


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
    pending_coord_table_pages: list[int] = []
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

    def flush_coord_table_stream() -> None:
        nonlocal pending_coord_table_pages
        if not pending_coord_table_pages:
            return
        human_pages = [p + 1 for p in pending_coord_table_pages]
        coord_md = _extract_coordinate_table_stream_cpu(doc, pending_coord_table_pages)
        if coord_md is not None:
            flush_table_batch()
            logger.info(
                "  -> Extrayendo flujo continuo de tablas por coordenadas (págs. %s) en CPU (0%% déficit decimal)...",
                human_pages,
            )
            markdown_parts.append(coord_md)
            pending_coord_table_pages = []
            return
        pages_fallback = list(pending_coord_table_pages)
        pending_coord_table_pages = []
        for cp_idx in pages_fallback:
            enqueue_table_page(doc[cp_idx], cp_idx)

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

    def enqueue_table_page(page_obj: pymupdf.Page, page_index: int) -> None:
        is_dense_page = len(page_obj.get_text("text")) >= 6000
        if is_dense_page and pending_table_pages:
            flush_table_batch()
        pending_table_pages.append(page_index)
        if is_dense_page or len(pending_table_pages) >= max_table_batch_size:
            flush_table_batch()

    for p_idx in range(start_idx, end_idx + 1):
        human_page = p_idx + 1
        if human_page in skip_pages:
            flush_coord_table_stream()
            flush_table_batch()
            flush_scanned_batch()
            logger.info("  -> Omitiendo página %d por directiva skip_pages del manifiesto.", human_page)
            continue

        page = doc[p_idx]

        if human_page in force_gemini_pages:
            flush_coord_table_stream()
            flush_scanned_batch()
            enqueue_table_page(page, p_idx)
            continue

        if layout == "scanned_ocr" or _is_scanned_page(page):
            flush_coord_table_stream()
            flush_table_batch()
            pending_scanned_pages.append(p_idx)
            if len(pending_scanned_pages) >= max_table_batch_size:
                flush_scanned_batch()
            continue

        flush_scanned_batch()
        if layout == "bilingual_right_es":
            flush_coord_table_stream()
            flush_table_batch()
            bilingual_md = _extract_bilingual_right_es_page_cpu(page)
            if bilingual_md:
                markdown_parts.append(bilingual_md)
            continue

        if _page_has_complex_table(page):
            if _has_clean_vector_table_grid(page):
                flush_coord_table_stream()
                flush_table_batch()
                logger.info(
                    "  -> Extrayendo tabla con rejilla vectorial limpia (pág. %d) en CPU (pymupdf4llm)...",
                    human_page,
                )
                markdown_parts.append(_extract_clean_vector_table_page_cpu(doc, p_idx))
            else:
                clip_md = _extract_vector_tables_page_cpu(page, layout=layout)
                if clip_md is not None and _markdown_table_is_healthy(clip_md):
                    flush_coord_table_stream()
                    flush_table_batch()
                    logger.info(
                        "  -> Extrayendo tablas vectoriales por clip de celda (pág. %d) en CPU (0%% déficit decimal)...",
                        human_page,
                    )
                    markdown_parts.append(clip_md)
                else:
                    pending_coord_table_pages.append(p_idx)
        else:
            flush_coord_table_stream()
            flush_table_batch()
            raw_page_text = (
                _extract_double_column_page_text(page)
                if layout == "double_column"
                else page.get_text("text")
            )
            markdown_parts.append(_format_plain_page_to_markdown(raw_page_text))

    flush_coord_table_stream()
    flush_table_batch()
    flush_scanned_batch()
    doc.close()

    full_md = "\n\n".join(part for part in markdown_parts if part.strip())
    full_md = _apply_text_markers(
        full_md,
        start_marker=getattr(boundary, "start_marker", None),
        end_marker=getattr(boundary, "end_marker", None),
    )
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
        if key in core_keys
        or (key == "layout" and value != "single_column")
        or (key == "subsector" and value != "general")
        or bool(value)
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
