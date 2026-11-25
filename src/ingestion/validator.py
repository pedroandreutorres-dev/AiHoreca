"""Gate de Calidad determinista en CPU para documentos Markdown Silver generados por ingesta.

Valida integridad estructural, paridad numérica contra el PDF original (PyMuPDF),
invariantes del chunker jurídico y resolubilidad de años/tablas salariales del retriever
con coste cero de API tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pymupdf

from src.config import BOUNDARIES_CACHE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.ingestion.pdf_ingestor import (
    _apply_text_markers,
    _extract_bilingual_right_es_page_text,
)
from src.ingestion.schemas import (
    CorpusValidationReport,
    DocumentManifestEntry,
    DocumentValidationReport,
    ValidationIssue,
    ValidationSeverity,
)
from src.retrieval.chunker import chunk_markdown_document
from src.retrieval.schemas import RetrievedDocument
from src.retrieval.vector_store import AiHorecaVectorStore

# Importes en formato español con 2 decimales (con o sin punto de millar, ej. "12,50", "1279,94" o "1.623,07")
_SPANISH_DECIMAL_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:\.\d{3})+|\d+),\d{2}\b")
_THOUSANDS_SPACE_RE = re.compile(r"(?<!\d)\d{1,2}(?:\.\s+|\s+\.)\d{3},\d{2}\b")
_HTML_TABLE_RE = re.compile(r"<\s*/?(?:table|tr|td|th|thead|tbody)\b", re.IGNORECASE)
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ALPHA_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}")
_LEVEL_TOKEN_RE = re.compile(r"^(?:[IVXLCDM]+(?:\s*-\s*[A-Z])?|\d{1,2})$")
_LOWERCASE_HEADING_RE = re.compile(r"^(#{1,6})\s+([a-záéíóúüñ])")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*(?:\s*:?-{2,}:?\s*\|?)?\s*$")
_YEAR_TOKEN_RE = re.compile(r"\b(202[3-9])\b(?!,\d)")


def _load_manifest(cache_path: Path) -> dict[str, DocumentManifestEntry]:
    """Carga el manifiesto `boundaries_cache.json` convirtiendo cada entrada a `DocumentManifestEntry`."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    manifest: dict[str, DocumentManifestEntry] = {}
    for pdf_name, entry_dict in raw.items():
        if isinstance(entry_dict, dict):
            manifest[pdf_name] = DocumentManifestEntry.model_validate(entry_dict)
    return manifest


def _split_markdown_row(line: str) -> list[str]:
    """Divide una fila de tabla Markdown en celdas ya recortadas."""
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def _iter_markdown_tables(lines: list[str]) -> Iterable[tuple[int, list[str], list[list[str]]]]:
    """Itera sobre tablas GFM Markdown detectadas por cabecera + fila separadora `|---|`.

    Devuelve `(start_line_1indexed, headers, body_rows)`.
    """
    idx = 0
    total = len(lines)
    while idx + 1 < total:
        line = lines[idx].strip()
        next_line = lines[idx + 1].strip()
        if "|" in line and "|" in next_line and _TABLE_SEP_RE.match(next_line):
            start_line = idx + 1
            headers = _split_markdown_row(line)
            body_rows: list[list[str]] = []
            cursor = idx + 2
            while cursor < total:
                row_line = lines[cursor].strip()
                if not row_line or "|" not in row_line or _TABLE_SEP_RE.match(row_line):
                    break
                body_rows.append(_split_markdown_row(row_line))
                cursor += 1
            yield start_line, headers, body_rows
            idx = cursor
            continue
        idx += 1


def _strip_markdown_wrappers(text: str) -> str:
    return re.sub(r"[*_`~]+", "", text).strip()


def _check_markdown_formatting_and_tables(
    md_text: str,
    doc_role: str = "completo",
) -> list[ValidationIssue]:
    """Evalúa sintaxis limpia, integridad columnar de tablas GFM y encabezados."""
    issues: list[ValidationIssue] = []
    lines = md_text.splitlines()

    code_fence_lines = [
        i + 1 for i, ln in enumerate(lines) if ln.strip().startswith("```")
    ]
    if code_fence_lines:
        issues.append(
            ValidationIssue(
                code="RESIDUAL_CODE_FENCE",
                severity="ERROR",
                message=f"Bloques ``` residuales detectados en líneas {code_fence_lines[:5]}.",
                details={"lines": code_fence_lines},
            )
        )

    html_lines = [
        i + 1 for i, ln in enumerate(lines) if _HTML_TABLE_RE.search(ln)
    ]
    if html_lines:
        issues.append(
            ValidationIssue(
                code="HTML_TABLE_LEAK",
                severity="ERROR",
                message=f"Etiquetas HTML <table>/<tr>/<td> detectadas en líneas {html_lines[:5]}.",
                details={"lines": html_lines},
            )
        )

    space_dot_matches = _THOUSANDS_SPACE_RE.findall(md_text)
    if space_dot_matches:
        issues.append(
            ValidationIssue(
                code="THOUSANDS_SPACE_ARTIFACT",
                severity="ERROR",
                message=f"{len(space_dot_matches)} importes con espacio tras punto de millar (ej. '{space_dot_matches[0]}').",
                details={"examples": space_dot_matches[:5], "count": len(space_dot_matches)},
            )
        )

    lowercase_headings = [
        (i + 1, ln.strip()[:80])
        for i, ln in enumerate(lines)
        if _LOWERCASE_HEADING_RE.match(ln.strip())
    ]
    if lowercase_headings:
        issues.append(
            ValidationIssue(
                code="LOWERCASE_HEADING",
                severity="WARNING",
                message=f"{len(lowercase_headings)} encabezados Markdown comienzan por minúscula (posible frase partida).",
                details={"samples": lowercase_headings[:5], "count": len(lowercase_headings)},
            )
        )

    dropped_col_lines: list[int] = []
    merged_header_cells: list[dict[str, object]] = []
    merged_body_br_cells: list[dict[str, object]] = []
    split_colspan_lines: list[int] = []
    bad_pipe_rows: list[int] = []

    for start_line, headers, body_rows in _iter_markdown_tables(lines):
        expected_cols = len(headers)
        norm_headers = [_strip_markdown_wrappers(h).lower() for h in headers]

        # Cabecera partida / colspan duplicado consecutivo o intermedia vacía
        for c_idx in range(len(norm_headers) - 1):
            if norm_headers[c_idx] and norm_headers[c_idx] == norm_headers[c_idx + 1]:
                split_colspan_lines.append(start_line)
                break
        if any(not h for h in norm_headers[1:]):
            split_colspan_lines.append(start_line)

        is_transposed_single_row = (
            len(body_rows) == 1
            and bool(norm_headers)
            and all(
                _ALPHA_WORD_RE.search(h) and not _YEAR_TOKEN_RE.search(h)
                for h in norm_headers
            )
        )

        missing_label_rows = 0
        rows_with_decimals = 0

        for offset, row in enumerate(body_rows):
            row_line_no = start_line + 2 + offset
            if len(row) != expected_cols:
                bad_pipe_rows.append(row_line_no)

            if len(row) >= 2 and not is_transposed_single_row:
                col0_clean = _strip_markdown_wrappers(row[0]).replace("€", "").strip()
                if _SPANISH_DECIMAL_RE.fullmatch(col0_clean):
                    dropped_col_lines.append(row_line_no)

                first_dec_col = next(
                    (idx for idx, cell in enumerate(row) if _SPANISH_DECIMAL_RE.search(cell)),
                    None,
                )
                if first_dec_col is not None:
                    rows_with_decimals += 1
                    has_leading_label = any(
                        _ALPHA_WORD_RE.search(_strip_markdown_wrappers(cell))
                        or _LEVEL_TOKEN_RE.fullmatch(_strip_markdown_wrappers(cell))
                        for cell in row[: max(1, first_dec_col)]
                    )
                    if not has_leading_label:
                        missing_label_rows += 1

        if (
            body_rows
            and rows_with_decimals >= 2
            and not norm_headers[0]
            and missing_label_rows > rows_with_decimals / 2
        ):
            dropped_col_lines.append(start_line)

        # Celdas fusionadas con `<br>` + decimales (error crítico en cabeceras, fila 0, multi-decimal o doc `tablas`)
        for row_idx, row in enumerate([headers, *body_rows]):
            for col_idx, cell in enumerate(row):
                if not _BR_TAG_RE.search(cell):
                    continue
                decimals_in_cell = _SPANISH_DECIMAL_RE.findall(cell)
                if not decimals_in_cell:
                    continue
                plain_cell = _BR_TAG_RE.sub(" ", cell)
                if _ALPHA_WORD_RE.search(plain_cell) or len(decimals_in_cell) >= 2:
                    sample = {
                        "line": start_line + (0 if row_idx == 0 else row_idx + 1),
                        "col": col_idx,
                        "cell": cell[:80],
                    }
                    if row_idx <= 1 or len(decimals_in_cell) >= 2 or doc_role == "tablas":
                        merged_header_cells.append(sample)
                    else:
                        merged_body_br_cells.append(sample)

    if dropped_col_lines:
        uniq = sorted(set(dropped_col_lines))
        issues.append(
            ValidationIssue(
                code="DROPPED_LABEL_COLUMN",
                severity="ERROR",
                message=f"Columna 0 numérica/vacía en tabla salarial (posible columna de categoría omitida) en líneas {uniq[:5]}.",
                details={"lines": uniq},
            )
        )

    if merged_header_cells:
        issues.append(
            ValidationIssue(
                code="MERGED_HEADER_CELL",
                severity="ERROR",
                message=f"{len(merged_header_cells)} celdas fusionadas con `<br>` e importes decimales en cabecera/fila inicial.",
                details={"samples": merged_header_cells[:5], "count": len(merged_header_cells)},
            )
        )

    if merged_body_br_cells:
        issues.append(
            ValidationIssue(
                code="MERGED_CELL_WITH_BR",
                severity="WARNING",
                message=f"{len(merged_body_br_cells)} celdas de cuerpo mezclan texto `<br>` e importe decimal.",
                details={"samples": merged_body_br_cells[:5], "count": len(merged_body_br_cells)},
            )
        )

    if split_colspan_lines:
        uniq_split = sorted(set(split_colspan_lines))
        issues.append(
            ValidationIssue(
                code="SPLIT_COLSPAN_HEADER",
                severity="WARNING",
                message=f"Cabeceras repetidas o vacías en tablas (líneas {uniq_split[:5]}).",
                details={"lines": uniq_split},
            )
        )

    if bad_pipe_rows:
        issues.append(
            ValidationIssue(
                code="INCONSISTENT_PIPE_COLUMNS",
                severity="WARNING",
                message=f"{len(bad_pipe_rows)} filas Markdown tienen distinto número de columnas que su cabecera.",
                details={"lines": bad_pipe_rows[:10], "count": len(bad_pipe_rows)},
            )
        )

    return issues


def _normalize_pdf_text_for_decimals(text: str) -> str:
    """Normaliza espacios tras punto de millar o coma errónea de millar en importes de PDFs."""
    fixed = re.sub(r"(?<!\d)(\d{1,2})(?:\.\s+|\s+\.)(\d{3},\d{2})\b", r"\1.\2", text)
    return re.sub(r"(?<!\d)(\d{1,2}),(\d{3},\d{2})\b", r"\1.\2", fixed)


def _extract_manifest_pdf_text(
    pdf_path: Path,
    manifest_entry: Optional[DocumentManifestEntry],
    *,
    sort: bool = False,
) -> str:
    """Extrae el texto del rango de páginas configurado en el manifiesto aplicando marcadores."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    start_page = manifest_entry.start_page if manifest_entry else 1
    end_page = manifest_entry.end_page if manifest_entry else total_pages
    skip_pages = set(manifest_entry.skip_pages) if manifest_entry else set()

    is_bilingual = bool(manifest_entry and manifest_entry.layout == "bilingual_right_es")
    pdf_chunks: list[str] = []
    for human_page in range(max(1, start_page), min(end_page, total_pages) + 1):
        if human_page in skip_pages:
            continue
        page_obj = doc[human_page - 1]
        if is_bilingual:
            pdf_chunks.append(_extract_bilingual_right_es_page_text(page_obj))
        else:
            pdf_chunks.append(page_obj.get_text("text", sort=sort))
    doc.close()

    extracted = "\n".join(pdf_chunks)
    if manifest_entry and (manifest_entry.start_marker or manifest_entry.end_marker):
        extracted = _apply_text_markers(
            extracted,
            start_marker=manifest_entry.start_marker,
            end_marker=manifest_entry.end_marker,
        )
    return extracted


def _check_decimal_parity(
    md_text: str,
    pdf_path: Optional[Path],
    manifest_entry: Optional[DocumentManifestEntry],
) -> tuple[list[ValidationIssue], Optional[int], int]:
    """Compara el multiconjunto de decimales españoles `X,XX` entre el rango del PDF y el `.md`."""
    md_decimals = _SPANISH_DECIMAL_RE.findall(md_text)
    md_count = len(md_decimals)
    if pdf_path is None or not pdf_path.exists():
        return [], None, md_count

    normalized_pdf_text = _normalize_pdf_text_for_decimals(
        _extract_manifest_pdf_text(pdf_path, manifest_entry, sort=False)
    )
    pdf_decimals = _SPANISH_DECIMAL_RE.findall(normalized_pdf_text)
    pdf_count = len(pdf_decimals)
    if pdf_count == 0:
        return [], 0, md_count

    pdf_counter = Counter(pdf_decimals)
    md_counter = Counter(md_decimals)
    missing_counter = pdf_counter - md_counter
    extra_counter = md_counter - pdf_counter

    missing_total = sum(missing_counter.values())
    extra_total = sum(extra_counter.values())
    recall = (pdf_count - missing_total) / pdf_count

    issues: list[ValidationIssue] = []
    doc_role = manifest_entry.doc_role if manifest_entry else "completo"

    if missing_total > 0:
        sample_missing = list(missing_counter.elements())[:10]
        is_fatal = doc_role == "tablas" or recall < 0.99 or missing_total > 5
        issues.append(
            ValidationIssue(
                code="DECIMAL_PARITY_LOSS" if is_fatal else "DECIMAL_PARITY_DIFF",
                severity="ERROR" if is_fatal else "WARNING",
                message=(
                    f"Faltan {missing_total}/{pdf_count} importes decimales del PDF "
                    f"(recall={recall:.2%}, ejemplos={sample_missing[:5]})."
                ),
                details={
                    "pdf_decimals": pdf_count,
                    "md_decimals": md_count,
                    "missing": missing_total,
                    "extra": extra_total,
                    "recall": round(recall, 4),
                    "sample_missing": sample_missing,
                },
            )
        )
    elif extra_total > 0:
        issues.append(
            ValidationIssue(
                code="DECIMAL_PARITY_EXTRA",
                severity="INFO",
                message=(
                    f"Paridad 100% de importes del PDF ({pdf_count}/{pdf_count}) con "
                    f"+{extra_total} repeticiones extra (normal en rowspan expandidos)."
                ),
                details={"pdf_decimals": pdf_count, "md_decimals": md_count, "extra": extra_total},
            )
        )

    return issues, pdf_count, md_count


_ROW_LABEL_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{5,}")


def _check_monetary_row_label_parity(
    md_text: str,
    pdf_path: Optional[Path],
    manifest_entry: Optional[DocumentManifestEntry],
) -> list[ValidationIssue]:
    """Verifica que las etiquetas descriptivas de filas salariales (>=2 decimales en PDF) permanezcan en el Markdown."""
    if pdf_path is None or not pdf_path.exists():
        return []

    raw_pdf_text = _extract_manifest_pdf_text(pdf_path, manifest_entry, sort=True)
    pdf_labels: set[str] = set()
    for line in raw_pdf_text.splitlines():
        if len(line.strip()) > 360:
            continue
        decs = list(_SPANISH_DECIMAL_RE.finditer(line))
        if len(decs) >= 2:
            prefix = line[: decs[0].start()]
            if len(prefix.strip()) > 200:
                continue
            for word in _ROW_LABEL_WORD_RE.findall(prefix.lower()):
                pdf_labels.add(word)

    if len(pdf_labels) < 10:
        return []

    md_lower = md_text.lower()
    present = {w for w in pdf_labels if w in md_lower}
    recall = len(present) / len(pdf_labels)
    if recall >= 0.90:
        return []

    missing = sorted(pdf_labels - present)
    return [
        ValidationIssue(
            code="TABLE_LABEL_PARITY_LOSS",
            severity="ERROR",
            message=(
                f"Faltan etiquetas descriptivas de filas salariales del PDF en el Markdown "
                f"(recall={recall:.2%}, ausentes={len(missing)}/{len(pdf_labels)}, ejemplos={missing[:6]})."
            ),
            details={
                "recall": round(recall, 4),
                "pdf_labels_count": len(pdf_labels),
                "missing_count": len(missing),
                "sample_missing": missing[:10],
            },
        )
    ]


def _extract_table_header_years(chunk_text: str) -> set[int]:
    """Extrae todos los años explícitos presentes en cabeceras de tablas Markdown multi-año."""
    years: set[int] = set()
    for _, headers, _ in _iter_markdown_tables(chunk_text.splitlines()):
        for cell in headers:
            for match in _YEAR_TOKEN_RE.findall(_strip_markdown_wrappers(cell)):
                years.add(int(match))
    return years


def _check_chunks_and_retriever_invariants(
    md_path: Path,
    boundaries_cache_raw: dict[str, dict],
    manifest_entry: Optional[DocumentManifestEntry],
    max_chunk_chars: int = 2600,
) -> tuple[list[ValidationIssue], int, list[str]]:
    """Ejecuta `chunk_markdown_document` e inspecciona metadatos, años y unicidad en CPU."""
    chunks = chunk_markdown_document(
        md_path,
        boundaries_cache=boundaries_cache_raw,
        max_chunk_chars=max_chunk_chars,
    )
    issues: list[ValidationIssue] = []
    if not chunks:
        issues.append(
            ValidationIssue(
                code="EMPTY_DOCUMENT_CHUNKS",
                severity="ERROR",
                message="El chunker produjo 0 fragmentos para este documento.",
            )
        )
        return issues, 0, []

    chunk_ids = [c.chunk_id for c in chunks]
    seen_ids: set[str] = set()
    dup_ids: list[str] = []
    for cid in chunk_ids:
        if cid in seen_ids:
            dup_ids.append(cid)
        seen_ids.add(cid)

    if dup_ids:
        issues.append(
            ValidationIssue(
                code="DUPLICATE_CHUNK_ID",
                severity="ERROR",
                message=f"{len(dup_ids)} chunk_ids duplicados dentro del documento.",
                details={"duplicates": dup_ids[:5]},
            )
        )

    empty_meta_chunks = [
        c.chunk_id
        for c in chunks
        if not c.territorio.strip()
        or not c.norma_titulo.strip()
        or not c.articulo.strip()
        or not c.text.strip()
    ]
    if empty_meta_chunks:
        issues.append(
            ValidationIssue(
                code="EMPTY_METADATA",
                severity="ERROR",
                message=f"{len(empty_meta_chunks)} chunks tienen metadatos vacíos o texto vacío.",
                details={"chunk_ids": empty_meta_chunks[:5]},
            )
        )

    oversized = [
        {"chunk_id": c.chunk_id, "chars": len(c.text)}
        for c in chunks
        if len(c.text) > int(max_chunk_chars * 2.5)
    ]
    if oversized:
        issues.append(
            ValidationIssue(
                code="OVERSIZED_CHUNK",
                severity="WARNING",
                message=f"{len(oversized)} chunks superan {int(max_chunk_chars * 2.5)} caracteres.",
                details={"samples": oversized[:5]},
            )
        )

    # Inspección de tablas monetarias y años mediante los helpers reales de AiHorecaVectorStore
    doc_role = manifest_entry.doc_role if manifest_entry else "completo"
    expected_years = set(manifest_entry.expected_years) if manifest_entry else set()

    monetary_chunks = 0
    yearless_monetary: list[str] = []
    observed_years: set[int] = set()

    for chunk in chunks:
        doc_view = RetrievedDocument(
            chunk_id=chunk.chunk_id,
            score=1.0,
            capa_normativa=chunk.capa_normativa,
            territorio=chunk.territorio,
            norma_titulo=chunk.norma_titulo,
            titulo_o_capitulo=chunk.titulo_o_capitulo,
            articulo=chunk.articulo,
            tipo_contenido=chunk.tipo_contenido,
            source_file=chunk.source_file,
            text=chunk.text,
        )
        is_real_table = chunk.tipo_contenido == "tabla_salarial" or (
            next(iter(_iter_markdown_tables(chunk.text.splitlines())), None) is not None
        )
        if AiHorecaVectorStore._is_monetary_table(doc_view):
            header_years = _extract_table_header_years(chunk.text)
            year = AiHorecaVectorStore._extract_chunk_year(doc_view)
            date_span = AiHorecaVectorStore._extract_chunk_date_span(doc_view)
            if year > 0:
                observed_years.add(year)
            observed_years.update(header_years)
            if (
                date_span is not None
                and not header_years
                and (date_span[1].year - date_span[0].year) <= 1
            ):
                observed_years.update({date_span[0].year, date_span[1].year})
            if is_real_table:
                monetary_chunks += 1
                if year == 0 and not header_years and date_span is None:
                    yearless_monetary.append(f"{chunk.chunk_id} ({chunk.articulo[:50]})")

    if yearless_monetary:
        if doc_role == "tablas":
            issues.append(
                ValidationIssue(
                    code="TABLE_MISSING_YEAR",
                    severity="ERROR",
                    message=(
                        f"{len(yearless_monetary)}/{monetary_chunks} tablas monetarias en documento `tablas` "
                        "carecen de año resoluble (`_extract_chunk_year == 0`)."
                    ),
                    details={"chunks": yearless_monetary[:10]},
                )
            )
        elif doc_role == "completo" and monetary_chunks > 0 and (len(yearless_monetary) / monetary_chunks) > 0.20:
            issues.append(
                ValidationIssue(
                    code="TABLE_MISSING_YEAR",
                    severity="WARNING",
                    message=(
                        f"{len(yearless_monetary)}/{monetary_chunks} tablas monetarias sin año resoluble "
                        "(>20% en documento `completo`)."
                    ),
                    details={"chunks": yearless_monetary[:10]},
                )
            )

    if expected_years:
        missing_years = sorted(expected_years - observed_years)
        if missing_years:
            issues.append(
                ValidationIssue(
                    code="MISSING_EXPECTED_YEAR",
                    severity="ERROR",
                    message=f"Faltan años esperados {missing_years} en tablas salariales (detectados: {sorted(observed_years)}).",
                    details={
                        "expected": sorted(expected_years),
                        "observed": sorted(observed_years),
                    },
                )
            )

    return issues, len(chunks), chunk_ids


def _apply_strict_mode(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """En modo `--strict`, promueve todo `WARNING` a `ERROR`."""
    promoted: list[ValidationIssue] = []
    for issue in issues:
        if issue.severity == "WARNING":
            promoted.append(issue.model_copy(update={"severity": "ERROR"}))
        else:
            promoted.append(issue)
    return promoted


def validate_document(
    md_path: Path,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    pdf_path: Optional[Path] = None,
    manifest_cache_path: Path = BOUNDARIES_CACHE_PATH,
    strict: bool = False,
    max_chunk_chars: int = 2600,
) -> DocumentValidationReport:
    """Valida deterministamente en CPU un único documento Markdown Silver."""
    md_text = md_path.read_text(encoding="utf-8")

    raw_cache: dict[str, dict] = {}
    if manifest_cache_path.exists():
        try:
            raw_cache = json.loads(manifest_cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw_cache = {}
    manifest = _load_manifest(manifest_cache_path)

    resolved_pdf = pdf_path or (raw_dir / f"{md_path.stem}.pdf")
    manifest_entry = manifest.get(resolved_pdf.name)
    doc_role = manifest_entry.doc_role if manifest_entry else "completo"

    issues: list[ValidationIssue] = []
    issues.extend(_check_markdown_formatting_and_tables(md_text, doc_role=doc_role))

    parity_issues, pdf_decimals_count, md_decimals_count = _check_decimal_parity(
        md_text=md_text,
        pdf_path=resolved_pdf if resolved_pdf.exists() else None,
        manifest_entry=manifest_entry,
    )
    issues.extend(parity_issues)
    issues.extend(
        _check_monetary_row_label_parity(
            md_text=md_text,
            pdf_path=resolved_pdf if resolved_pdf.exists() else None,
            manifest_entry=manifest_entry,
        )
    )

    chunk_issues, chunks_count, _ = _check_chunks_and_retriever_invariants(
        md_path=md_path,
        boundaries_cache_raw=raw_cache,
        manifest_entry=manifest_entry,
        max_chunk_chars=max_chunk_chars,
    )
    issues.extend(chunk_issues)

    if strict:
        issues = _apply_strict_mode(issues)

    has_error = any(issue.severity == "ERROR" for issue in issues)
    return DocumentValidationReport(
        md_file=md_path.name,
        pdf_file=resolved_pdf.name if resolved_pdf.exists() else None,
        doc_role=doc_role,
        passed=not has_error,
        chunks_count=chunks_count,
        pdf_decimals_count=pdf_decimals_count,
        md_decimals_count=md_decimals_count,
        issues=issues,
    )


def validate_corpus(
    processed_dir: Path = PROCESSED_DATA_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    manifest_cache_path: Path = BOUNDARIES_CACHE_PATH,
    strict: bool = False,
    max_chunk_chars: int = 2600,
) -> CorpusValidationReport:
    """Valida todos los `.md` en `processed_dir` y comprueba unicidad global de `chunk_id`."""
    raw_cache: dict[str, dict] = {}
    if manifest_cache_path.exists():
        try:
            raw_cache = json.loads(manifest_cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw_cache = {}

    reports: list[DocumentValidationReport] = []
    global_chunk_owners: dict[str, str] = {}

    for md_path in sorted(processed_dir.glob("*.md")):
        doc_report = validate_document(
            md_path,
            raw_dir=raw_dir,
            manifest_cache_path=manifest_cache_path,
            strict=strict,
            max_chunk_chars=max_chunk_chars,
        )
        # Verificación cruzada de colisiones de chunk_id entre documentos distintos
        chunks = chunk_markdown_document(
            md_path,
            boundaries_cache=raw_cache,
            max_chunk_chars=max_chunk_chars,
        )
        cross_dups: list[str] = []
        for chunk in chunks:
            prev_owner = global_chunk_owners.get(chunk.chunk_id)
            if prev_owner and prev_owner != md_path.name:
                cross_dups.append(f"{chunk.chunk_id} (ya en {prev_owner})")
            else:
                global_chunk_owners[chunk.chunk_id] = md_path.name
        if cross_dups:
            doc_report.issues.append(
                ValidationIssue(
                    code="GLOBAL_DUPLICATE_CHUNK_ID",
                    severity="ERROR",
                    message=f"Colisión de chunk_id entre documentos distintos: {cross_dups[:5]}.",
                    details={"collisions": cross_dups[:10]},
                )
            )
            doc_report.passed = False

        reports.append(doc_report)

    total_errors = sum(rep.error_count for rep in reports)
    total_warnings = sum(rep.warning_count for rep in reports)
    total_chunks = sum(rep.chunks_count for rep in reports)

    return CorpusValidationReport(
        strict=strict,
        passed=total_errors == 0,
        total_documents=len(reports),
        total_chunks=total_chunks,
        total_errors=total_errors,
        total_warnings=total_warnings,
        documents=reports,
    )


def format_validation_summary(report: DocumentValidationReport | CorpusValidationReport) -> str:
    """Formatea un informe legible para consola."""
    if isinstance(report, DocumentValidationReport):
        status = "PASS" if report.passed else "FAIL"
        parity = (
            f"dec={report.md_decimals_count}/{report.pdf_decimals_count}"
            if report.pdf_decimals_count is not None
            else f"dec={report.md_decimals_count} (sin PDF)"
        )
        lines = [
            f"[{status}] {report.md_file} | rol={report.doc_role} | chunks={report.chunks_count} | "
            f"{parity} | errors={report.error_count} | warnings={report.warning_count}"
        ]
        for issue in report.issues:
            lines.append(f"  - [{issue.severity}] {issue.code}: {issue.message}")
        return "\n".join(lines)

    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"=== DATA QUALITY GATE [{status}] (strict={report.strict}) ===",
        f"Documentos: {report.total_documents} | Chunks totales: {report.total_chunks} | "
        f"Errores: {report.total_errors} | Warnings: {report.total_warnings}",
        "",
    ]
    for doc_rep in report.documents:
        lines.append(format_validation_summary(doc_rep))
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Gate de Calidad determinista en CPU sobre Markdown Silver (data/processed)."
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Ruta o nombre de archivo .md concreto dentro de data/processed/ a validar.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Modo estricto: promueve cualquier WARNING a ERROR (bloqueante).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Imprime el informe completo en formato JSON.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=2600,
        help="Umbral base de caracteres de chunk (por defecto 2600).",
    )
    args = parser.parse_args()

    if args.file:
        candidate = Path(args.file)
        if not candidate.exists():
            candidate = PROCESSED_DATA_DIR / args.file
        if not candidate.exists() and not candidate.name.endswith(".md"):
            candidate = PROCESSED_DATA_DIR / f"{args.file}.md"
        if not candidate.exists():
            print(f"ERROR: No se encontró el documento '{args.file}'.", file=sys.stderr)
            sys.exit(2)
        report: DocumentValidationReport | CorpusValidationReport = validate_document(
            candidate,
            strict=args.strict,
            max_chunk_chars=args.max_chunk_chars,
        )
    else:
        report = validate_corpus(
            strict=args.strict,
            max_chunk_chars=args.max_chunk_chars,
        )

    if args.json_output:
        print(report.model_dump_json(indent=2))
    else:
        print(format_validation_summary(report))

    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    _cli()
