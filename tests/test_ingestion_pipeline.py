"""Suite de tests de regresión y canarios en CPU para el pipeline de ingesta y el Gate de Calidad.

Coste de tokens LLM = 0. Todos los tests operan determinísticamente sobre:
- heurísticas unitarias y adaptadores PDF en memoria (`TestPdfIngestorHeuristics`)
- páginas canario de los PDFs reales si están presentes en `data/raw/` (`TestRealPdfCanaries`)
- invariantes del manifiesto y del corpus Silver ya indexado (`TestManifestAndSilverCorpusInvariants`)
- trampas sintéticas de regresión para el validador (`TestValidatorDefectTraps`)
"""

from __future__ import annotations

from datetime import date
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf

from src.config import BOUNDARIES_CACHE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.evaluation.evaluator import (
    EvaluationCaseResult,
    _compute_skippable_case_ids,
    evaluate_golden_dataset,
)
from src.evaluation.golden_dataset import GoldenTestCase
from src.graph.nodes import (
    AiHorecaGraphNodes,
    _canonicalize_llm_province,
    _format_supported_territories_display,
    _pretty_territory_label,
)
from src.graph.state import LegalCitation, LegalSynthesisSchema, QueryAnalysisSchema
from src.ingestion.pdf_ingestor import (
    _apply_text_markers,
    _clean_gemini_markdown_output,
    _count_missing_spanish_decimals,
    _extract_double_column_page_text,
    _extract_vector_tables_page_cpu,
    _format_plain_page_to_markdown,
    _has_clean_vector_table_grid,
    _is_coofficial_non_spanish_text,
    _is_scanned_page,
    _normalize_spanish_thousands,
    _page_has_complex_table,
)
from src.ingestion.schemas import DocumentManifestEntry
from src.ingestion.validator import _extract_table_header_years, validate_corpus, validate_document
from src.retrieval.chunker import (
    _canonicalize_territory,
    chunk_all_processed_documents,
    chunk_markdown_document,
)
from src.retrieval.schemas import LegalChunk, RetrievedDocument
from src.retrieval.vector_store import AiHorecaVectorStore


def _build_two_column_pdf_in_memory() -> pymupdf.Document:
    """Construye un PDF vectorial sintético de 1 página con cabecera ancha + 2 columnas."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Cabecera de ancho completo
    page.insert_textbox(
        pymupdf.Rect(40, 40, 555, 80),
        "RESOLUCION BOP PROVINCIAL CONVENIO HOSTELERIA",
        fontsize=12,
    )
    # Columna izquierda
    page.insert_textbox(
        pymupdf.Rect(40, 110, 270, 260),
        "COLUMNA IZQUIERDA BLOQUE UNO\nArtículo 1. Ámbito territorial.",
        fontsize=10,
    )
    page.insert_textbox(
        pymupdf.Rect(40, 280, 270, 420),
        "COLUMNA IZQUIERDA BLOQUE DOS\nTexto continuo de la primera columna.",
        fontsize=10,
    )
    # Columna derecha (misma coordenada Y que la izquierda)
    page.insert_textbox(
        pymupdf.Rect(320, 110, 555, 260),
        "COLUMNA DERECHA BLOQUE UNO\nArtículo 2. Jornada laboral anual.",
        fontsize=10,
    )
    page.insert_textbox(
        pymupdf.Rect(320, 280, 555, 420),
        "COLUMNA DERECHA BLOQUE DOS\nTexto final de la segunda columna.",
        fontsize=10,
    )
    return doc


def _build_scanned_image_pdf_in_memory() -> pymupdf.Document:
    """Construye un PDF sintético de 1 página que contiene únicamente una imagen raster."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), 0)
    pix.clear_with(240)
    page.insert_image(pymupdf.Rect(20, 20, 380, 380), pixmap=pix)
    return doc


class TestPdfIngestorHeuristics(unittest.TestCase):
    """Pruebas unitarias sobre adaptadores CPU, normalización numérica y detección de formato."""

    def test_normalize_spanish_thousands_fixes_spaced_dots_and_preserves_valid(self) -> None:
        raw = "Camarero: 1. 623,07 €; Jefe: 2 .105,50 €; Ayudante: 12. 345,67 €; Año 2025; Normal 1.623,07 €"
        normalized = _normalize_spanish_thousands(raw)
        self.assertIn("1.623,07", normalized)
        self.assertIn("2.105,50", normalized)
        self.assertIn("12.345,67", normalized)
        self.assertNotIn("1. 623,07", normalized)
        self.assertNotIn("2 .105,50", normalized)
        self.assertIn("Año 2025", normalized)

    def test_clean_gemini_markdown_output_strips_fences_and_normalizes(self) -> None:
        raw = "```markdown\n| Categoría | 2025 |\n|---|---|\n| Cocinero | 1. 850,00 |\n```"
        cleaned = _clean_gemini_markdown_output(raw)
        self.assertFalse(cleaned.startswith("```"))
        self.assertFalse(cleaned.endswith("```"))
        self.assertIn("1.850,00", cleaned)

        raw_period = "## TABLA DE RETRIBUCIONES\nDel 1 de enero a 31 de diciembre de 2026\n\n### Horas extra"
        merged = _clean_gemini_markdown_output(raw_period)
        self.assertIn(
            "## TABLA DE RETRIBUCIONES — Del 1 de enero a 31 de diciembre de 2026",
            merged,
        )

    def test_format_plain_page_promotes_real_articles_and_rejects_inline_citations(self) -> None:
        sample_page = "\n".join(
            [
                "42",
                "Boletín Oficial de la Provincia de Málaga Núm. 70 — 14/04/2026",
                "CVE: BOP-SE-2025-102031",
                "Nº 102 - viernes 30 de mayo de 2025",
                "Butlletí Oficial de la Província d´Alacant",
                "edita excma. diputació provincial d´alacant",
                "Nº 200 de 18/10/2023",
                "8879 / 2023",
                "Real Decreto 713/2010 de 28 de mayo sobre registro.",
                "CAPÍTULO II",
                "Artículo 15. Salario base.",
                "Artículo 31º. Vacaciones.",
                "Artículo11º: Manutención.",
                "Artículo 15 del vigente Estatuto de los Trabajadores regula este punto.",
                "artículo 34.8 ET permite solicitar adaptación.",
                "Capítulo décimo sobre disposiciones varias.",
                "Anexo I. Clasificación profesional",
                "categorías establecidas en el",
                "Anexo 1 del vigente Convenio, en concordancia con las funciones ALEH.",
            ]
        )
        md = _format_plain_page_to_markdown(sample_page)
        self.assertIn("## CAPÍTULO II", md)
        self.assertIn("### Artículo 15. Salario base.", md)
        self.assertIn("### Artículo 31º. Vacaciones.", md)
        self.assertIn("### Artículo11º: Manutención.", md)
        self.assertIn("## Anexo I. Clasificación profesional", md)
        self.assertNotIn("## Anexo 1 del vigente Convenio", md)
        self.assertNotIn("### Artículo 15 del vigente Estatuto", md)
        self.assertNotIn("### artículo 34.8 ET", md)
        self.assertNotIn("## Capítulo décimo", md)
        self.assertNotIn("Boletín Oficial de la Provincia", md)
        self.assertNotIn("CVE: BOP-SE-2025-102031", md)
        self.assertNotIn("Nº 102 - viernes 30 de mayo de 2025", md)
        self.assertNotIn("Butlletí Oficial de la Província d´Alacant", md)
        self.assertNotIn("edita excma. diputació provincial", md)
        self.assertNotIn("Nº 200 de 18/10/2023", md)
        self.assertNotIn("8879 / 2023", md)
        self.assertIn("Real Decreto 713/2010 de 28 de mayo sobre registro.", md)
        self.assertNotRegex(md, r"(?m)^42$")

    def test_double_column_extractor_preserves_left_then_right_reading_order(self) -> None:
        doc = _build_two_column_pdf_in_memory()
        try:
            extracted = _extract_double_column_page_text(doc[0])
        finally:
            doc.close()

        pos_banner = extracted.find("RESOLUCION BOP PROVINCIAL")
        pos_l1 = extracted.find("COLUMNA IZQUIERDA BLOQUE UNO")
        pos_l2 = extracted.find("COLUMNA IZQUIERDA BLOQUE DOS")
        pos_r1 = extracted.find("COLUMNA DERECHA BLOQUE UNO")
        pos_r2 = extracted.find("COLUMNA DERECHA BLOQUE DOS")

        self.assertNotIn(-1, [pos_banner, pos_l1, pos_l2, pos_r1, pos_r2])
        self.assertLess(pos_banner, pos_l1)
        self.assertLess(pos_l1, pos_l2)
        self.assertLess(pos_l2, pos_r1)
        self.assertLess(pos_r1, pos_r2)

    def test_scanned_page_detector_distinguishes_text_from_raster_only(self) -> None:
        text_doc = _build_two_column_pdf_in_memory()
        scanned_doc = _build_scanned_image_pdf_in_memory()
        try:
            self.assertFalse(_is_scanned_page(text_doc[0]))
            self.assertTrue(_is_scanned_page(scanned_doc[0]))
        finally:
            text_doc.close()
            scanned_doc.close()

    def test_apply_text_markers_trims_prefix_and_trailing_local_notices(self) -> None:
        raw = (
            "Resolución administrativa previa\n"
            "EL DIRECTOR GENERAL DE TRABAJO.\n"
            "ACUERDO PARA LA MEJORA DE LAS CONDICIONES DE TRABAJO\n"
            "PRIMERO. Incremento del 4,00% en 2026.\n"
            "III. ADMINISTRACIÓN LOCAL\n"
            "EXCMO. CABILDO INSULAR DE FUERTEVENTURA\n"
            "12.345,67 €\n"
        )
        trimmed = _apply_text_markers(
            raw,
            start_marker="ACUERDO PARA LA MEJORA DE LAS CONDICIONES DE TRABAJO",
            end_marker="III. ADMINISTRACIÓN LOCAL",
        )
        self.assertTrue(trimmed.startswith("ACUERDO PARA LA MEJORA"))
        self.assertIn("4,00%", trimmed)
        self.assertNotIn("Resolución administrativa previa", trimmed)
        self.assertNotIn("CABILDO", trimmed)
        self.assertNotIn("12.345,67", trimmed)

    def test_format_plain_page_rejoins_split_article_headers(self) -> None:
        raw = (
            "Texto previo del artículo 10.\n"
            "ARTÍCULO\n"
            "11.\n"
            "GRATIFICACIONES\n"
            "EXTRAORDINARIAS.\n"
            "Las empresas abonarán dos pagas extras.\n"
            "ARTÍCULO\n"
            "6.\n"
            "COMPENSACIÓN\n"
        )
        md = _format_plain_page_to_markdown(raw)
        self.assertIn("### ARTÍCULO 11. GRATIFICACIONES", md)
        self.assertIn("### ARTÍCULO 6. COMPENSACIÓN", md)

    def test_las_palmas_island_aliases_canonicalize_uniformly(self) -> None:
        for alias in ("Las Palmas", "Las Palmas de Gran Canaria", "Gran Canaria", "Lanzarote", "Fuerteventura"):
            self.assertEqual(_canonicalize_territory(alias), "las_palmas", msg=alias)
            self.assertEqual(_canonicalize_llm_province(alias), "las_palmas", msg=alias)

    def test_santa_cruz_tenerife_island_aliases_canonicalize_uniformly(self) -> None:
        for alias in ("Santa Cruz de Tenerife", "Tenerife", "La Palma", "La Gomera", "El Hierro"):
            self.assertEqual(_canonicalize_territory(alias), "santa_cruz_tenerife", msg=alias)
            self.assertEqual(_canonicalize_llm_province(alias), "santa_cruz_tenerife", msg=alias)

    def test_count_missing_spanish_decimals_detects_dropped_amounts(self) -> None:
        pdf_txt = "Jefe cocina 1.711,30 € 1.689,95 € 1.689,95 € 1.643,30 €"
        md_ok = "| Jefe cocina | 1.711,30 € | 1.689,95 € | 1.689,95 € | 1.643,30 € |"
        missing_ok, total_ok, _ = _count_missing_spanish_decimals(pdf_txt, md_ok)
        self.assertEqual((missing_ok, total_ok), (0, 4))

        md_bad = "| Jefe cocina | 1.711,30 € | 1.689,95 € | 1.643,30 € |"
        missing_bad, total_bad, sample_bad = _count_missing_spanish_decimals(pdf_txt, md_bad)
        self.assertEqual((missing_bad, total_bad), (1, 4))
        self.assertEqual(sample_bad, ["1.689,95"])


class TestRealPdfCanaries(unittest.TestCase):
    """Páginas canario reales en `data/raw/` para vigilar regresiones de clasificación vectorial/Gemini."""

    @unittest.skipUnless(
        (RAW_DATA_DIR / "convenio_malaga_tablas_2025_2026.pdf").exists(),
        "PDF real convenio_malaga_tablas_2025_2026.pdf no presente en data/raw/",
    )
    def test_malaga_tablas_canary_pages_route_to_gemini_vision(self) -> None:
        doc = pymupdf.open(RAW_DATA_DIR / "convenio_malaga_tablas_2025_2026.pdf")
        try:
            # Pág. 1 es texto introductorio sin tabla compleja; págs. 16-17 son tablas sin rejilla cerrada
            self.assertFalse(_page_has_complex_table(doc[0]))
            for page_1idx in (2, 16, 17):
                page = doc[page_1idx - 1]
                self.assertTrue(_page_has_complex_table(page))
                self.assertFalse(_has_clean_vector_table_grid(page))
        finally:
            doc.close()

    @unittest.skipUnless(
        (RAW_DATA_DIR / "convenio_catalunya_bcn.pdf").exists(),
        "PDF real convenio_catalunya_bcn.pdf no presente en data/raw/",
    )
    def test_catalunya_clean_vector_table_grid_routes_to_cpu(self) -> None:
        doc = pymupdf.open(RAW_DATA_DIR / "convenio_catalunya_bcn.pdf")
        try:
            page_70 = doc[69]
            self.assertTrue(_page_has_complex_table(page_70))
            self.assertTrue(_has_clean_vector_table_grid(page_70))
        finally:
            doc.close()

    @unittest.skipUnless(
        (RAW_DATA_DIR / "convenio_madrid.pdf").exists()
        and (RAW_DATA_DIR / "convenio_leon_texto.pdf").exists(),
        "PDFs reales de Madrid/León no presentes en data/raw/",
    )
    def test_madrid_and_leon_table_pages_detected_as_complex(self) -> None:
        madrid_doc = pymupdf.open(RAW_DATA_DIR / "convenio_madrid.pdf")
        leon_doc = pymupdf.open(RAW_DATA_DIR / "convenio_leon_texto.pdf")
        try:
            self.assertTrue(_page_has_complex_table(madrid_doc[49]))
            self.assertTrue(_page_has_complex_table(leon_doc[71]))
        finally:
            madrid_doc.close()
            leon_doc.close()


class TestManifestAndSilverCorpusInvariants(unittest.TestCase):
    """Invariantes sobre `boundaries_cache.json`, los 942 chunks Silver y el Gate de Calidad."""

    def test_boundaries_cache_conforms_to_manifest_schema(self) -> None:
        raw = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(raw), 10)
        for pdf_name, payload in raw.items():
            entry = DocumentManifestEntry.model_validate(payload)
            self.assertLessEqual(entry.start_page, entry.end_page, msg=pdf_name)
        malaga_tablas = DocumentManifestEntry.model_validate(
            raw["convenio_malaga_tablas_2025_2026.pdf"]
        )
        self.assertEqual(malaga_tablas.doc_role, "tablas")
        self.assertEqual(malaga_tablas.expected_years, [2025, 2026])

    def test_silver_corpus_chunks_are_unique_and_non_empty(self) -> None:
        chunks = chunk_all_processed_documents(PROCESSED_DATA_DIR)
        self.assertGreaterEqual(len(chunks), 942)
        chunk_ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(set(chunk_ids)), len(chunks))
        for chunk in chunks:
            self.assertTrue(chunk.territorio.strip(), msg=chunk.chunk_id)
            self.assertTrue(chunk.norma_titulo.strip(), msg=chunk.chunk_id)
            self.assertTrue(chunk.articulo.strip(), msg=chunk.chunk_id)
            self.assertTrue(chunk.text.strip(), msg=chunk.chunk_id)

    def test_malaga_monetary_and_classification_tables_invariants(self) -> None:
        base_chunks = chunk_markdown_document(PROCESSED_DATA_DIR / "convenio_malaga.md")
        tablas_chunks = chunk_markdown_document(
            PROCESSED_DATA_DIR / "convenio_malaga_tablas_2025_2026.md"
        )
        self.assertEqual(len(base_chunks), 116)
        self.assertEqual(len(tablas_chunks), 38)

        base_level_count = 0
        for chunk in base_chunks:
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
            if AiHorecaVectorStore._is_classification_level_table(doc_view):
                base_level_count += 1
        self.assertEqual(base_level_count, 2)

        monetary_years: set[int] = set()
        monetary_count = 0
        for chunk in tablas_chunks:
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
            if AiHorecaVectorStore._is_monetary_table(doc_view):
                monetary_count += 1
                yr = AiHorecaVectorStore._extract_chunk_year(doc_view)
                self.assertIn(yr, {2025, 2026}, msg=chunk.chunk_id)
                monetary_years.add(yr)
        self.assertEqual(monetary_count, 37)
        self.assertEqual(monetary_years, {2025, 2026})

    def test_validator_corpus_passes_and_malaga_tablas_passes_strict(self) -> None:
        corpus_report = validate_corpus(strict=False)
        self.assertTrue(corpus_report.passed, msg=corpus_report.model_dump_json(indent=2))
        self.assertEqual(corpus_report.total_errors, 0)

        malaga_report = validate_document(
            PROCESSED_DATA_DIR / "convenio_malaga_tablas_2025_2026.md",
            strict=True,
        )
        self.assertTrue(malaga_report.passed)
        self.assertEqual(malaga_report.error_count, 0)
        self.assertEqual(malaga_report.warning_count, 0)

    def test_sevilla_classification_strict_and_multiyear_tables(self) -> None:
        sevilla_path = PROCESSED_DATA_DIR / "convenio_sevilla.md"
        sevilla_report = validate_document(sevilla_path, strict=True)
        self.assertTrue(sevilla_report.passed, msg=sevilla_report.model_dump_json(indent=2))
        self.assertEqual(sevilla_report.error_count, 0)
        self.assertEqual(sevilla_report.warning_count, 0)

        chunks = chunk_markdown_document(sevilla_path)
        level_count = 0
        years_found: set[int] = set()
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
            if AiHorecaVectorStore._is_classification_level_table(doc_view):
                level_count += 1
            if chunk.tipo_contenido == "tabla_salarial":
                years_found.update(_extract_table_header_years(chunk.text))
        self.assertEqual(level_count, 1)
        self.assertEqual(years_found, {2025, 2026, 2027, 2028})

    def test_alicante_classification_strict_and_2026_tables(self) -> None:
        base_path = PROCESSED_DATA_DIR / "convenio_alicante.md"
        tablas_path = PROCESSED_DATA_DIR / "convenio_alicante_tablas_2026.md"

        tablas_report = validate_document(tablas_path, strict=True)
        self.assertTrue(tablas_report.passed, msg=tablas_report.model_dump_json(indent=2))
        self.assertEqual(tablas_report.error_count, 0)
        self.assertEqual(tablas_report.warning_count, 0)

        base_chunks = chunk_markdown_document(base_path)
        level_count = 0
        for chunk in base_chunks:
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
            if AiHorecaVectorStore._is_classification_level_table(doc_view):
                level_count += 1
        self.assertGreaterEqual(level_count, 2)

        tablas_chunks = chunk_markdown_document(tablas_path)
        years_found: set[int] = set()
        for chunk in tablas_chunks:
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
            if AiHorecaVectorStore._is_monetary_table(doc_view):
                yr = AiHorecaVectorStore._extract_chunk_year(doc_view)
                self.assertEqual(yr, 2026, msg=chunk.chunk_id)
                years_found.add(yr)
        self.assertEqual(years_found, {2026})

    def test_las_palmas_classification_strict_and_2025_tables(self) -> None:
        base_path = PROCESSED_DATA_DIR / "convenio_las_palmas.md"
        tablas_path = PROCESSED_DATA_DIR / "convenio_las_palmas_acuerdo_tablas_2025_2026.md"
        self.assertTrue(base_path.exists())
        self.assertTrue(tablas_path.exists())

        base_report = validate_document(base_path, strict=True)
        tablas_report = validate_document(tablas_path, strict=True)
        self.assertTrue(base_report.passed, msg=base_report.model_dump_json(indent=2))
        self.assertEqual(base_report.error_count, 0)
        self.assertTrue(tablas_report.passed, msg=tablas_report.model_dump_json(indent=2))
        self.assertEqual(tablas_report.error_count, 0)

        base_chunks = chunk_markdown_document(base_path)
        level_count = 0
        for chunk in base_chunks:
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
            if AiHorecaVectorStore._is_classification_level_table(doc_view):
                level_count += 1
        self.assertGreaterEqual(level_count, 2)

        tablas_chunks = chunk_markdown_document(tablas_path)
        years_found: set[int] = set()
        for chunk in tablas_chunks:
            if chunk.tipo_contenido != "tabla_salarial":
                continue
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
            if AiHorecaVectorStore._is_monetary_table(doc_view):
                yr = AiHorecaVectorStore._extract_chunk_year(doc_view)
                self.assertEqual(yr, 2025, msg=chunk.chunk_id)
                years_found.add(yr)
        self.assertEqual(years_found, {2025})

    def test_santa_cruz_tenerife_strict_tables_and_anexo2_classification(self) -> None:
        base_path = PROCESSED_DATA_DIR / "convenio_santa_cruz_tenerife.md"
        tablas_path = PROCESSED_DATA_DIR / "convenio_santa_cruz_tenerife_tablas_2025_2028.md"
        rev3d_path = PROCESSED_DATA_DIR / "convenio_santa_cruz_tenerife_revision_3d_2025_2027.md"
        for p in (base_path, tablas_path, rev3d_path):
            self.assertTrue(p.exists(), msg=str(p))
            rep = validate_document(p, strict=True)
            self.assertTrue(rep.passed, msg=rep.model_dump_json(indent=2))
            self.assertEqual(rep.error_count, 0, msg=p.name)
            self.assertEqual(rep.warning_count, 0, msg=p.name)

        tablas_txt = tablas_path.read_text(encoding="utf-8")
        rev3d_txt = rev3d_path.read_text(encoding="utf-8")
        self.assertIn("1.448,63", tablas_txt)
        self.assertIn("1.583,97", tablas_txt)
        self.assertNotIn("1.302,18", tablas_txt)
        self.assertIn("1.307,24", rev3d_txt)
        self.assertIn("1.320,04", rev3d_txt)

        tablas_chunks = chunk_markdown_document(tablas_path)
        anexo2_cls = [
            c
            for c in tablas_chunks
            if AiHorecaVectorStore._is_classification_level_table(
                RetrievedDocument(
                    chunk_id=c.chunk_id,
                    score=1.0,
                    capa_normativa=c.capa_normativa,
                    territorio=c.territorio,
                    norma_titulo=c.norma_titulo,
                    titulo_o_capitulo=c.titulo_o_capitulo,
                    articulo=c.articulo,
                    tipo_contenido=c.tipo_contenido,
                    source_file=c.source_file,
                    text=c.text,
                )
            )
        ]
        self.assertGreaterEqual(len(anexo2_cls), 1)
        self.assertTrue(any("ANEXO II" in c.titulo_o_capitulo for c in anexo2_cls))

        base_chunks = chunk_markdown_document(base_path)
        art34_chunks = [c for c in base_chunks if "Artículo 34" in c.articulo]
        self.assertTrue(art34_chunks)
        self.assertTrue(
            all(
                c.tipo_contenido == "articulo"
                and c.titulo_o_capitulo == "CAPÍTULO VI. ESTRUCTURA SALARIAL"
                for c in art34_chunks
            ),
            msg=[(c.tipo_contenido, c.titulo_o_capitulo, c.articulo) for c in art34_chunks],
        )

        rev3d_chunks = chunk_markdown_document(rev3d_path)
        rev3d_camarero_26_27 = [
            c for c in rev3d_chunks if "1.320,04" in c.text and "2026" in c.titulo_o_capitulo
        ]
        self.assertEqual(len(rev3d_camarero_26_27), 1)
        self.assertIn("CLASIFICACIÓN 3D. RESTAURACIÓN Y OCIO", rev3d_camarero_26_27[0].titulo_o_capitulo)


class TestValidatorDefectTraps(unittest.TestCase):
    """Trampas de defectos sintéticos para garantizar que el Gate de Calidad bloquea cada fallo conocido."""

    def _validate_synthetic(
        self,
        md_content: str,
        *,
        filename: str = "convenio_prueba.md",
        manifest_entry: dict | None = None,
        pdf_text: str | None = None,
        strict: bool = False,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_dir = tmp / "raw"
            proc_dir = tmp / "processed"
            raw_dir.mkdir()
            proc_dir.mkdir()

            md_path = proc_dir / filename
            md_path.write_text(md_content, encoding="utf-8")

            pdf_name = f"{md_path.stem}.pdf"
            if pdf_text is not None:
                pdf_doc = pymupdf.open()
                page = pdf_doc.new_page()
                page.insert_textbox(pymupdf.Rect(40, 40, 550, 780), pdf_text, fontsize=10)
                pdf_doc.save(raw_dir / pdf_name)
                pdf_doc.close()

            cache_path = proc_dir / "boundaries_cache.json"
            if manifest_entry is not None:
                cache_path.write_text(
                    json.dumps({pdf_name: manifest_entry}, ensure_ascii=False),
                    encoding="utf-8",
                )

            return validate_document(
                md_path,
                raw_dir=raw_dir,
                manifest_cache_path=cache_path,
                strict=strict,
            )

    def test_trap_residual_code_fence(self) -> None:
        rep = self._validate_synthetic("```markdown\n## Artículo 1\nContenido.\n```\n")
        self.assertFalse(rep.passed)
        self.assertIn("RESIDUAL_CODE_FENCE", {i.code for i in rep.issues})

    def test_trap_html_table_leak(self) -> None:
        rep = self._validate_synthetic("## Artículo 1\n<table><tr><td>100,00</td></tr></table>\n")
        self.assertFalse(rep.passed)
        self.assertIn("HTML_TABLE_LEAK", {i.code for i in rep.issues})

    def test_trap_thousands_space_artifact(self) -> None:
        md = "## Tablas 2025\n\n| Categoría | 2025 |\n|---|---|\n| Camarero | 1. 623,07 |\n"
        rep = self._validate_synthetic(md)
        self.assertFalse(rep.passed)
        self.assertIn("THOUSANDS_SPACE_ARTIFACT", {i.code for i in rep.issues})

    def test_trap_dropped_label_column(self) -> None:
        md = "## Tablas salariales 2025\n\n| 2025 | 2026 |\n|---|---|\n| 1.850,00 | 1.920,00 |\n| 1.400,00 | 1.450,00 |\n"
        rep = self._validate_synthetic(md)
        self.assertFalse(rep.passed)
        self.assertIn("DROPPED_LABEL_COLUMN", {i.code for i in rep.issues})

    def test_trap_merged_header_cell_with_br(self) -> None:
        md = (
            "## Tablas salariales 2026\n\n"
            "| Categoría<br>1.850,00 | 2026 |\n"
            "|---|---|\n"
            "| Jefe de cocina | 1.920,00 |\n"
        )
        rep = self._validate_synthetic(md)
        self.assertFalse(rep.passed)
        self.assertIn("MERGED_HEADER_CELL", {i.code for i in rep.issues})

    def test_trap_inconsistent_pipe_columns_promoted_under_strict(self) -> None:
        md = (
            "## Tablas salariales 2025\n\n"
            "| Categoría | Salario Base | Plus |\n"
            "|---|---|---|\n"
            "| Cocinero | 1.600,00 |\n"
        )
        non_strict = self._validate_synthetic(md, strict=False)
        strict_rep = self._validate_synthetic(md, strict=True)
        self.assertTrue(non_strict.passed)
        self.assertIn("INCONSISTENT_PIPE_COLUMNS", {i.code for i in non_strict.issues})
        self.assertFalse(strict_rep.passed)

    def test_trap_table_missing_year_and_missing_expected_year(self) -> None:
        manifest = {
            "doc_type": "convenio_hosteleria",
            "doc_role": "tablas",
            "expected_years": [2025, 2026],
            "territorio": "prueba",
            "start_page": 1,
            "end_page": 1,
            "justificacion": "Test sintético de tablas salariales.",
        }
        md = (
            "## Tablas salariales definitivas\n\n"
            "| Categoría | Salario | Plus | Manutención |\n"
            "|---|---|---|---|\n"
            "| Camarero | 1.500,00 | 120,00 | 55,00 |\n"
            "| Cocinero | 1.750,00 | 120,00 | 55,00 |\n"
        )
        rep = self._validate_synthetic(md, manifest_entry=manifest)
        codes = {i.code for i in rep.issues if i.severity == "ERROR"}
        self.assertFalse(rep.passed)
        self.assertIn("TABLE_MISSING_YEAR", codes)
        self.assertIn("MISSING_EXPECTED_YEAR", codes)

    def test_trap_decimal_parity_loss_against_pdf(self) -> None:
        pdf_text = "Tablas 2026\nCategoría A 1.500,00\nCategoría B 1.600,00\nCategoría C 1.700,00\n"
        md = (
            "## Tablas 2026\n\n"
            "| Categoría | Salario 2026 |\n|---|---|\n"
            "| Categoría A | 1.500,00 |\n"
            "| Categoría C | 1.700,00 |\n"
        )
        manifest = {
            "doc_type": "convenio_hosteleria",
            "doc_role": "tablas",
            "territorio": "prueba",
            "start_page": 1,
            "end_page": 1,
            "justificacion": "Test de paridad numérica.",
        }
        rep = self._validate_synthetic(md, manifest_entry=manifest, pdf_text=pdf_text)
        self.assertFalse(rep.passed)
        self.assertIn("DECIMAL_PARITY_LOSS", {i.code for i in rep.issues})

    def test_multi_year_columnar_table_satisfies_expected_years_and_detects_missing_column(
        self,
    ) -> None:
        manifest = {
            "doc_type": "convenio_hosteleria",
            "doc_role": "completo",
            "expected_years": [2025, 2026, 2027, 2028],
            "territorio": "prueba",
            "start_page": 1,
            "end_page": 1,
            "justificacion": "Test de tabla salarial multi-año en columnas.",
        }
        md_ok = (
            "## TABLAS SALARIALES 2025-2028\n\n"
            "### Hoteles 5 y 4 estrellas\n\n"
            "| Niveles | 2025 | 2026 | 2027 | 2028 |\n"
            "|---|---|---|---|---|\n"
            "| Nivel 1 | 1.571,85 € | 1.650,44 € | 1.716,46 € | 1.785,11 € |\n"
        )
        rep_ok = self._validate_synthetic(md_ok, manifest_entry=manifest)
        self.assertTrue(rep_ok.passed)
        self.assertNotIn("MISSING_EXPECTED_YEAR", {i.code for i in rep_ok.issues})

        md_missing = (
            "## TABLAS SALARIALES 2025-2028\n\n"
            "### Hoteles 5 y 4 estrellas\n\n"
            "| Niveles | 2025 | 2026 | 2028 |\n"
            "|---|---|---|---|\n"
            "| Nivel 1 | 1.571,85 € | 1.650,44 € | 1.785,11 € |\n"
        )
        rep_missing = self._validate_synthetic(md_missing, manifest_entry=manifest)
        self.assertFalse(rep_missing.passed)
        self.assertIn("MISSING_EXPECTED_YEAR", {i.code for i in rep_missing.issues})

    def test_prose_monetary_article_without_table_does_not_trigger_strict_missing_year(
        self,
    ) -> None:
        manifest = {
            "doc_type": "convenio_hosteleria",
            "doc_role": "completo",
            "territorio": "prueba",
            "start_page": 1,
            "end_page": 1,
            "justificacion": "Artículo en prosa con cuantías fijas en euros sin tabla.",
        }
        md = (
            "## CAPÍTULO VII\n\n"
            "### Artículo 38. Seguro colectivo.\n\n"
            "Las empresas concertarán un seguro con indemnización de 18.000,00 € por muerte "
            "y 30.000,00 € por invalidez permanente absoluta.\n"
        )
        rep = self._validate_synthetic(md, manifest_entry=manifest, strict=True)
        self.assertTrue(rep.passed, msg=rep.model_dump_json(indent=2))
        self.assertNotIn("TABLE_MISSING_YEAR", {i.code for i in rep.issues})

    def test_transposed_single_row_category_headers_not_flagged_as_dropped_column(
        self,
    ) -> None:
        md_transposed = (
            "## Uniformidad 2026\n\n"
            "| Personal con esmoquin | Personal cocina | Resto personal |\n"
            "|---|---|---|\n"
            "| 25,00 € | 18,42 € | 10,86 € |\n"
        )
        rep_ok = self._validate_synthetic(md_transposed, strict=True)
        self.assertTrue(rep_ok.passed, msg=rep_ok.model_dump_json(indent=2))
        self.assertNotIn("DROPPED_LABEL_COLUMN", {i.code for i in rep_ok.issues})

        md_years_bad = (
            "## Tablas 2025-2026\n\n"
            "| 2025 | 2026 |\n"
            "|---|---|\n"
            "| 1.850,00 | 1.920,00 |\n"
        )
        rep_bad = self._validate_synthetic(md_years_bad)
        self.assertFalse(rep_bad.passed)
        self.assertIn("DROPPED_LABEL_COLUMN", {i.code for i in rep_bad.issues})

    def test_table_label_parity_loss_catches_dropped_job_title_column(self) -> None:
        pdf_text = "\n".join(
            [
                "ANEXO I Tablas salariales 2026",
                "Jefe/a recepción 1º 1 1.514,51 € 1.514,51 €",
                "Recepcionista 2º 3 1.338,56 € 1.338,56 €",
                "Administrativo/a 2º 3 1.332,74 € 1.332,74 €",
                "Cocinero/a 2º 4 1.307,24 € 1.307,24 €",
                "Repostero/a 2º 2 1.411,48 € 1.411,48 €",
                "Encargado economato 2º 3 1.332,74 € 1.332,74 €",
                "Camarero/a 2º 4 1.320,04 € 1.320,04 €",
                "Sumiller 2º 4 1.300,86 € 1.300,86 €",
                "Gobernante/a 1º 1 1.436,65 € 1.436,65 €",
                "Mantenimiento especialista 2º 4 1.300,85 € 1.300,85 €",
            ]
        )
        md_dropped_jobs = (
            "## ANEXO I Tablas salariales 2026\n\n"
            "| Gr. Prof | Gr. sal | Bares | Restaurantes |\n"
            "|---|---|---|---|\n"
            "| 1º | 1 | 1.514,51 € | 1.514,51 € |\n"
            "| 2º | 3 | 1.338,56 € | 1.338,56 € |\n"
            "| 2º | 3 | 1.332,74 € | 1.332,74 € |\n"
            "| 2º | 4 | 1.307,24 € | 1.307,24 € |\n"
            "| 2º | 2 | 1.411,48 € | 1.411,48 € |\n"
            "| 2º | 3 | 1.332,74 € | 1.332,74 € |\n"
            "| 2º | 4 | 1.320,04 € | 1.320,04 € |\n"
            "| 2º | 4 | 1.300,86 € | 1.300,86 € |\n"
            "| 1º | 1 | 1.436,65 € | 1.436,65 € |\n"
            "| 2º | 4 | 1.300,85 € | 1.300,85 € |\n"
        )
        rep = self._validate_synthetic(
            md_dropped_jobs,
            pdf_text=pdf_text,
            manifest_entry={
                "doc_type": "convenio_hosteleria",
                "doc_role": "tablas",
                "expected_years": [2026],
                "territorio": "prueba",
                "start_page": 1,
                "end_page": 1,
                "justificacion": "Trampa sintética por omisión de columna de puestos.",
            },
        )
        self.assertFalse(rep.passed)
        self.assertIn("TABLE_LABEL_PARITY_LOSS", {i.code for i in rep.issues})


class TestRetrievalSelectionLogic(unittest.TestCase):
    """Tests deterministas en CPU para cuotas y selección territorial en rerank_and_select_by_territory."""

    @staticmethod
    def _doc(
        chunk_id: str,
        *,
        capa: str = "convenio_provincial",
        territorio: str = "alicante",
        titulo: str = "CAPITULO V",
        articulo: str = "",
        tipo: str = "articulo",
        text: str = "",
        rerank_score: float = 0.1,
    ) -> RetrievedDocument:
        return RetrievedDocument(
            chunk_id=chunk_id,
            score=rerank_score,
            rerank_score=rerank_score,
            capa_normativa=capa,
            territorio=territorio,
            matched_territorio=territorio if capa == "convenio_provincial" else None,
            norma_titulo="Convenio Colectivo de Hostelería de Alicante",
            titulo_o_capitulo=titulo,
            articulo=articulo,
            tipo_contenido=tipo,
            source_file="convenio_alicante.md",
            text=text,
        )

    def test_extra_pay_title_first_and_separate_2026_extra_pay_table(self) -> None:
        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        candidates = [
            # Art. 32 tiene mayor score pero solo menciona "gratificaciones extraordinarias de verano, navidad y octubre"
            self._doc(
                "art32",
                articulo="Artículo 32. ESTRUCTURA DEL SALARIO",
                tipo="articulo",
                rerank_score=0.95,
                text=(
                    "Además del salario base, están las gratificaciones extraordinarias de verano, "
                    "navidad y octubre."
                ),
            ),
            self._doc(
                "art39",
                articulo="Artículo 39. REVISIÓN SALARIAL DURANTE LA VIGENCIA DEL CONVENIO",
                tipo="articulo",
                rerank_score=0.90,
                text="Para el año 2026 el incremento será de un 2%.",
            ),
            # Art. 38 lleva la rúbrica específica pero menor score inicial que Art. 32
            self._doc(
                "art38",
                articulo="Artículo 38. GRATIFICACIONES EXTRAORDINARIAS",
                tipo="articulo",
                rerank_score=0.60,
                text=(
                    "a) VERANO Y NAVIDAD: 10 de julio y 10 de diciembre.\n"
                    "b) OCTUBRE: cuantía fija el 10 de octubre."
                ),
            ),
            self._doc(
                "tbl_base_2026",
                titulo="TABLA DE RETRIBUCIONES — Del 1 de enero a 31 de diciembre de 2026",
                articulo="Salarios base mensuales",
                tipo="tabla_salarial",
                rerank_score=0.92,
                text=(
                    "| NIVELES | GRUPOS DE ESTABLECIMIENTOS A | B | C |\n"
                    "|---|---|---|---|\n"
                    "| 1 | 1.727,77 | 1.644,83 | 1.499,70 |\n"
                    "| 4 | 1.461,45 | 1.388,30 | 1.314,25 |"
                ),
            ),
            self._doc(
                "tbl_extra_2026",
                titulo="TABLA DE RETRIBUCIONES — Del 1 de enero a 31 de diciembre de 2026",
                articulo="Valor del plus de nocturnidad por hora trabajada (Art. 34)",
                tipo="tabla_salarial",
                rerank_score=0.35,
                text=(
                    "| NIVELES | A | B | C |\n|---|---|---|---|\n| 4 | 1,21 | 1,14 | 1,10 |\n\n"
                    "### Conceptos económicos adicionales\n"
                    "| Concepto | Importe |\n|---|---|\n"
                    "| Gratificación Extraordinaria Octubre (Art. 38 letra b) | 1.235,82 |"
                ),
            ),
            self._doc(
                "anexo1",
                titulo="ANEXO I",
                articulo="ANEXO I (Parte 1/2)",
                tipo="anexo_o_disposicion",
                rerank_score=0.50,
                text="GRUPO A: Hoteles y hoteles-apartamentos de 4 estrellas, 4 estrellas superior y 5 estrellas.",
            ),
            self._doc(
                "anexo2",
                titulo="ANEXO II",
                articulo="ANEXO II (Parte 1/2)",
                tipo="tabla_salarial",
                rerank_score=0.55,
                text="| NIVEL 4 |\n|---|\n| Cocinero/a |",
            ),
            self._doc("aleh1", capa="aleh_estatal", territorio="estatal", articulo="Art. 16 ALEH", rerank_score=0.70, text="Área Funcional 2ª Cocina: Cocinero/a."),
            self._doc("aleh2", capa="aleh_estatal", territorio="estatal", articulo="Preámbulo ALEH", rerank_score=0.65, text="VI ALEH."),
            self._doc("et1", capa="estatuto_trabajadores", territorio="estatal", articulo="Art. 26 ET", rerank_score=0.68, text="Del salario."),
            self._doc("et2", capa="estatuto_trabajadores", territorio="estatal", articulo="Art. 31 ET", rerank_score=0.64, text="Gratificaciones extraordinarias."),
        ]

        selected = store.rerank_and_select_by_territory(
            candidates=candidates,
            query=(
                "Soy cocinero en un hotel de 4 estrellas en Benidorm (Alicante), "
                "¿cuál es mi salario base mensual vigente en 2026 y qué pagas extraordinarias me corresponden?"
            ),
            territorios=["alicante"],
            use_cohere_rerank=False,
            role_context="Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupo Profesional II)",
        )
        selected_ids = {d.chunk_id for d in selected}
        self.assertLessEqual(len(selected), 10)
        self.assertIn("art38", selected_ids)
        self.assertIn("tbl_base_2026", selected_ids)
        self.assertIn("tbl_extra_2026", selected_ids)
        self.assertIn("anexo1", selected_ids)
        self.assertIn("anexo2", selected_ids)

    def test_build_filter_preserves_multiword_territories_and_aliases(self) -> None:
        f_lp = AiHorecaVectorStore._build_filter(provincia="las_palmas")
        self.assertEqual(f_lp.must[0].match.value, "las_palmas")

        f_lanz = AiHorecaVectorStore._build_filter(provincia="Lanzarote")
        self.assertEqual(f_lanz.must[0].match.value, "las_palmas")

        f_pair = AiHorecaVectorStore._build_filter(provincia="madrid,_leon")
        self.assertEqual(f_pair.must[0].match.any, ["madrid", "leon"])

        f_canarias = AiHorecaVectorStore._build_filter(provincia="las_palmas,_santa_cruz_tenerife")
        self.assertEqual(f_canarias.must[0].match.any, ["las_palmas", "santa_cruz_tenerife"])

        for isla in ("La Palma", "La Gomera", "El Hierro"):
            f_isla = AiHorecaVectorStore._build_filter(provincia=isla)
            self.assertEqual(f_isla.must[0].match.value, "santa_cruz_tenerife", msg=isla)

    def test_date_span_and_preferred_monetary_table_selection(self) -> None:
        ref = date(2026, 11, 22)
        d_25_26 = self._doc(
            "t25_26",
            territorio="santa_cruz_tenerife",
            titulo="ANEXO I. Tablas salariales 1 de julio 2025 - 30 de junio 2026",
            articulo="AF 2ª — Cocina y Economato",
            tipo="tabla_salarial",
            text="| Cocinero/a | 1.403,03 € | 1.382,34 € |",
        )
        d_26_27 = self._doc(
            "t26_27",
            territorio="santa_cruz_tenerife",
            titulo="ANEXO I. Tablas salariales 1 de julio 2026 - 30 de junio 2027",
            articulo="AF 2ª — Cocina y Economato",
            tipo="tabla_salarial",
            text="| Cocinero/a | 1.448,63 € | 1.427,27 € |",
        )
        d_27_28 = self._doc(
            "t27_28",
            territorio="santa_cruz_tenerife",
            titulo="ANEXO I. Tablas salariales 1 de julio 2027 - 30 de junio 2028",
            articulo="AF 2ª — Cocina y Economato",
            tipo="tabla_salarial",
            text="| Cocinero/a | 1.495,71 € | 1.473,66 € |",
        )

        self.assertEqual(
            AiHorecaVectorStore._extract_chunk_date_span(d_25_26),
            (date(2025, 7, 1), date(2026, 6, 30)),
        )
        self.assertEqual(
            AiHorecaVectorStore._extract_chunk_date_span(d_26_27),
            (date(2026, 7, 1), date(2027, 6, 30)),
        )

        tenerife_docs = [d_25_26, d_26_27, d_27_28]
        self.assertEqual(
            AiHorecaVectorStore._select_preferred_monetary_tables(tenerife_docs, [], reference_date=ref),
            {"t26_27"},
        )
        self.assertEqual(
            AiHorecaVectorStore._select_preferred_monetary_tables(tenerife_docs, [2026], reference_date=ref),
            {"t26_27"},
        )
        self.assertEqual(
            AiHorecaVectorStore._select_preferred_monetary_tables(tenerife_docs, [2025], reference_date=ref),
            {"t25_26"},
        )

        d_ultra_25 = self._doc(
            "lp_25",
            territorio="las_palmas",
            titulo="TABLAS SALARIALES AÑO 2025",
            articulo="Nivel IV",
            tipo="tabla_salarial",
            text="| Nivel IV | 1.518,51 € |",
        )
        self.assertEqual(
            AiHorecaVectorStore._select_preferred_monetary_tables([d_ultra_25], [], reference_date=ref),
            {"lp_25"},
        )

        area_pat = AiHorecaVectorStore._extract_aleh_area_pattern(
            "Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupo Profesional II)"
        )
        self.assertIsNotNone(area_pat)
        self.assertTrue(AiHorecaVectorStore._doc_matches_area(d_26_27, area_pat))

    def test_audit_accepts_annual_percentage_revision_and_prorated_and_rejects_hallucination(self) -> None:
        docs = [
            self._doc(
                "lp_tbl_2025",
                territorio="las_palmas",
                articulo="TABLAS SALARIALES AÑO 2025",
                tipo="tabla_salarial",
                text="| Nivel IV | 1.518,51 |",
            ),
            self._doc(
                "lp_acuerdo_2026",
                territorio="las_palmas",
                articulo="Preámbulo / Encabezado (Parte 2/4)",
                tipo="articulo",
                text="CUARTO. El incremento salarial para 2026 será de 4,00% sobre las tablas de 2025.",
            ),
        ]
        cite = LegalCitation(
            norma="Convenio Colectivo de Hostelería de Las Palmas",
            capa_normativa="convenio_provincial",
            articulo_o_tabla="Acuerdo Cuarto y Tabla 2025",
            extracto_clave="1.518,51 € en abril 2025 + 4,00 % en enero 2026.",
        )

        synth_ok = LegalSynthesisSchema(
            jerarquia_normativa_aplicada="Convenio Las Palmas sobre mínimos ET.",
            citas_normativas=[cite],
            cifras_y_datos_criticos=["1.518,51 €", "4,00%"],
            respuesta_final_markdown=(
                "Según la Tabla Salarial 2025 el salario base es 1.518,51 € mensuales; "
                "aplicando la subida obligatoria del 4,00 % prevista en el Acuerdo Cuarto para 2026, "
                "el importe actualizado asciende a 1.579,25 € brutos/mes en 14 pagas "
                "(1.842,46 € brutos/mes prorrateado en 12 pagas)."
            ),
        )
        rep_ok = AiHorecaGraphNodes._audit_numerical_and_citation_fidelity(
            synthesis=synth_ok,
            docs=docs,
            expected_territorio="las_palmas",
        )
        self.assertTrue(rep_ok.passed, msg=rep_ok.model_dump_json(indent=2))
        self.assertEqual(rep_ok.unverified_numbers, [])

        synth_bad = LegalSynthesisSchema(
            jerarquia_normativa_aplicada="Convenio Las Palmas.",
            citas_normativas=[cite],
            cifras_y_datos_criticos=["1.518,51 €"],
            respuesta_final_markdown=(
                "Según la Tabla Salarial 2025 corresponde 1.518,51 € y en 2026 pasaría a 1.999,99 €."
            ),
        )
        rep_bad = AiHorecaGraphNodes._audit_numerical_and_citation_fidelity(
            synthesis=synth_bad,
            docs=docs,
            expected_territorio="las_palmas",
        )
        self.assertFalse(rep_bad.passed)
        self.assertIn("1.999,99", rep_bad.unverified_numbers)

    def test_router_rescues_supported_insular_province_reclassified_from_unsupported(self) -> None:
        inline, _ = _format_supported_territories_display(
            {"santa_cruz_tenerife", "las_palmas", "baleares"}
        )
        for token in (
            "La Palma",
            "La Gomera",
            "El Hierro",
            "Lanzarote",
            "Fuerteventura",
            "Mallorca",
            "Ibiza",
        ):
            self.assertIn(token, inline)

        nodes = AiHorecaGraphNodes.__new__(AiHorecaGraphNodes)
        nodes.indexed_territories = {"santa_cruz_tenerife", "las_palmas", "madrid"}

        fake_analysis = QueryAnalysisSchema(
            intent="UNSUPPORTED_PROVINCE",
            motivo_clasificacion="La Palma no aparece literal",
            provincias_detectadas_en_turno_actual=["la palma"],
            cambio_de_puesto_en_turno_actual=True,
            requiere_consulta_normativa_bbdd=False,
            requiere_convenio_provincial=True,
            requiere_puesto_o_categoria=True,
            oficio_coloquial_detectado="cocinero",
            categoria_profesional_aleh="Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupo Profesional II)",
            area_funcional_aleh="Área Funcional 2ª",
            consulta_autocontenida_resuelta="Salario base de cocinero en hotel de 4 estrellas en La Palma",
            query_busqueda_hibrida="salario base cocinero hotel 4 estrellas Santa Cruz de Tenerife tablas salariales",
        )

        with patch(
            "src.graph.nodes._call_structured_llm",
            return_value=(fake_analysis, "mock-lite"),
        ):
            out = nodes.analyze_and_route_node(
                {
                    "query": "Soy cocinero en un hotel de 4 estrellas en La Palma, ¿cuál es mi salario base?",
                    "conversation_history": [],
                    "session_slots": {},
                }
            )

        self.assertEqual(out["intent"], "HOSPITALITY_CORE")
        self.assertEqual(out["territorio_normalizado"], "santa_cruz_tenerife")
        self.assertTrue(out["requiere_consulta_normativa_bbdd"])
        self.assertTrue(out["datos_completos"])
        self.assertEqual(out["route_decision"], "parallel_retrieve")

    def test_multi_topic_title_first_and_area_stem_table_priority(self) -> None:
        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        candidates = [
            self._doc(
                "art23",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 23º. Jornada Nocturna",
                tipo="articulo",
                rerank_score=0.96,
                text="Las horas nocturnas entre 22:00 y 06:00 se retribuyen con un 50% de incremento.",
            ),
            self._doc(
                "art32_extra_mention",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 32º. Salario Bruto Garantizado",
                tipo="articulo",
                rerank_score=0.92,
                text="La estructura salarial contempla gratificaciones extraordinarias y complementos.",
            ),
            self._doc(
                "art19",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 19º. Clasificación de Empresas",
                tipo="articulo",
                rerank_score=0.88,
                text="Clasificación de establecimientos por grupos.",
            ),
            self._doc(
                "art24",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 24º. Horas Extraordinarias",
                tipo="articulo",
                rerank_score=0.45,
                text="Dividiendo el total devengado anual entre 1.829 horas.",
            ),
            self._doc(
                "tbl_parte2_ayudante",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO I. 1 de julio 2026 - 30 de junio 2027 | CLASIFICACIÓN 3",
                articulo="AF 3ª RESTAURANTE (Parte 2/2)",
                tipo="tabla_salarial",
                rerank_score=0.94,
                text="| Ayudante/a de camarero | 3º | 5 | 1.298,23 € |",
            ),
            self._doc(
                "tbl_3d_camarero",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO I. 1 de julio 2026 - 30 de junio 2027 | CLASIFICACIÓN 3D. RESTAURACIÓN Y OCIO",
                articulo="AF 3ª: Restaurante, Sala, Bar y Similares",
                tipo="tabla_salarial",
                rerank_score=0.72,
                text="| Camarero/a. | 2º | 4 | 1.320,04 € |",
            ),
            self._doc(
                "anexo2",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO II. CLASIFICACIÓN EMPRESAS",
                articulo="Tabla de clasificación de empresas",
                tipo="tabla_salarial",
                rerank_score=0.80,
                text="| D | Pensiones | Aptos | Restaurantes | Club |",
            ),
            self._doc(
                "art7",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 7º. Revisión Salarial",
                tipo="articulo",
                rerank_score=0.90,
                text="Cláusula de revisión salarial por IPC.",
            ),
            self._doc(
                "aleh1",
                capa="aleh_estatal",
                territorio="estatal",
                articulo="Art. 15 ALEH",
                rerank_score=0.70,
                text="Área Funcional 3ª: Camarero/a.",
            ),
            self._doc(
                "aleh2",
                capa="aleh_estatal",
                territorio="estatal",
                articulo="Art. 16 ALEH",
                rerank_score=0.66,
                text="Grupos profesionales VI ALEH.",
            ),
            self._doc(
                "et1",
                capa="estatuto_trabajadores",
                territorio="estatal",
                articulo="Art. 35 ET",
                rerank_score=0.68,
                text="Horas extraordinarias.",
            ),
            self._doc(
                "et2",
                capa="estatuto_trabajadores",
                territorio="estatal",
                articulo="Art. 36 ET",
                rerank_score=0.64,
                text="Trabajo nocturno.",
            ),
        ]

        selected = store.rerank_and_select_by_territory(
            candidates=candidates,
            query=(
                "Trabajo como camarero en un restaurante ordinario en San Sebastián de La Gomera. "
                "¿Cuál es mi salario base, cómo se retribuye la nocturnidad y cuál es el divisor anual "
                "para calcular las horas extraordinarias?"
            ),
            territorios=["santa_cruz_tenerife"],
            use_cohere_rerank=False,
            role_context="camarero | Camarero/a (Área Funcional 3ª Restauración) | Área Funcional 3ª",
        )
        selected_ids = {d.chunk_id for d in selected}
        self.assertIn("art23", selected_ids)
        self.assertIn("art24", selected_ids)
        self.assertIn("tbl_3d_camarero", selected_ids)
        self.assertNotIn("tbl_parte2_ayudante", selected_ids)

    def test_atomic_replace_source_files_fails_fast_on_per_day_quota_without_deleting(self) -> None:
        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        store.collection_name = "test_collection"
        store.qdrant = MagicMock()
        store.ensure_collection = MagicMock()
        store._embed_sparse_batch = MagicMock()
        store.genai_client = MagicMock()
        store.genai_client.models.embed_content.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED: EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"
        )

        chunk = LegalChunk(
            chunk_id="00000000-0000-0000-0000-000000000001",
            source_file="convenio_santa_cruz_tenerife.md",
            norma_titulo="Convenio Santa Cruz de Tenerife",
            capa_normativa="convenio_provincial",
            territorio="santa_cruz_tenerife",
            provincias_aplicables=["santa_cruz_tenerife"],
            titulo_o_capitulo="CAPÍTULO VI",
            articulo="Artículo 34º. Pagas Extraordinarias",
            tipo_contenido="articulo",
            chunk_index=0,
            text="Texto artículo 34º",
        )

        with self.assertRaises(RuntimeError) as ctx:
            store.index_chunks(
                [chunk],
                replace_source_files=["convenio_santa_cruz_tenerife.md"],
            )

        self.assertIn("Cuota diaria agotada", str(ctx.exception))
        # Fail-fast: exactamente 1 intento sin bucle 8x25s
        self.assertEqual(store.genai_client.models.embed_content.call_count, 1)
        # Atomicidad: nunca borra ni upserta si fallaron los embeddings
        store.qdrant.delete.assert_not_called()
        store.qdrant.upsert.assert_not_called()

    def test_establishment_subsector_disambiguates_hotel_vs_ordinary_restaurant_tables(
        self,
    ) -> None:
        self.assertEqual(
            AiHorecaVectorStore._extract_query_establishment_subsector(
                "Trabajo como camarero en un restaurante ordinario en La Gomera"
            ),
            "restauracion",
        )
        self.assertEqual(
            AiHorecaVectorStore._extract_query_establishment_subsector(
                "Soy camarero en un hotel de 4 estrellas en Alicante"
            ),
            "hospedaje",
        )

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        common_cands = [
            self._doc(
                "art23",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 23º. Jornada Nocturna",
                tipo="articulo",
                rerank_score=0.90,
                text="Trabajo nocturno del 50%.",
            ),
            self._doc(
                "art24",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 24º. Horas Extraordinarias",
                tipo="articulo",
                rerank_score=0.88,
                text="Dividiendo el total devengado anual entre 1.829 horas.",
            ),
            self._doc(
                "art32",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 32º. Salario Bruto Garantizado",
                tipo="articulo",
                rerank_score=0.85,
                text="Salarios brutos garantizados.",
            ),
            self._doc(
                "art7",
                territorio="santa_cruz_tenerife",
                articulo="Artículo 7º. Revisión Salarial",
                tipo="articulo",
                rerank_score=0.82,
                text="Revisión salarial.",
            ),
            self._doc(
                "tbl_cls1_hoteles",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO I. Tablas salariales 1 de julio 2026 - 30 de junio 2027 | CLASIFICACIÓN 1. HOTELES Y PENSIONES",
                articulo="AF 3ª Restaurante, Sala, Bar y Similares (Parte 1/2)",
                tipo="tabla_salarial",
                rerank_score=0.95,
                text=(
                    "| AF 3ª | Gr. Prof | Gr. sal | H 5* | H 4* | H 3* | Pensiones |\n"
                    "|---|---|---|---|---|---|---|\n"
                    "| Camarero/a. | 2º | 4 | 1.455,26 € | 1.455,26 € | 1.433,90 € | 1.369,76 € |"
                ),
            ),
            self._doc(
                "tbl_cls3_lujo",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO I. Tablas salariales 1 de julio 2026 - 30 de junio 2027 | CLASIFICACIÓN 3. CAMPOS DE GOLF, CASINOS... RESTAURACIÓN",
                articulo="AF 3ª RESTAURANTE (Parte 1/2)",
                tipo="tabla_salarial",
                rerank_score=0.90,
                text=(
                    "| AF 3ª | Gr. Prof | Gr. sal | Campos golf | Casinos | Rtesy cafés bares lujo |\n"
                    "|---|---|---|---|---|---|\n"
                    "| Camarero/a. | 2º | 4 | 1.455,26 € | 1.455,26 € | 1.455,26 € |"
                ),
            ),
            self._doc(
                "tbl_cls3d_restauracion",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO I. Tablas salariales 1 de julio 2026 - 30 de junio 2027 | CLASIFICACIÓN 3D. RESTAURACIÓN Y OCIO",
                articulo="AF 3ª: Restaurante, Sala, Bar y Similares",
                tipo="tabla_salarial",
                rerank_score=0.60,
                text=(
                    "| Puesto | Gr. Prof | Gr. sal | Bares y Cafet. | Rtes | Sal. recreativos |\n"
                    "|---|---|---|---|---|---|\n"
                    "| Camarero/a. | 2º | 4 | 1.320,04 € | 1.320,04 € | 1.320,04 € |"
                ),
            ),
            self._doc(
                "anexo2",
                territorio="santa_cruz_tenerife",
                titulo="ANEXO II. CLASIFICACIÓN EMPRESAS",
                articulo="Tabla de clasificación de empresas",
                tipo="tabla_salarial",
                rerank_score=0.75,
                text="| D | Pensiones | Aptos | Restaurantes | Club |",
            ),
            self._doc("aleh1", capa="aleh_estatal", territorio="estatal", articulo="Art. 15 ALEH", rerank_score=0.70, text="AF 3ª"),
            self._doc("aleh2", capa="aleh_estatal", territorio="estatal", articulo="Art. 16 ALEH", rerank_score=0.68, text="Grupo II"),
            self._doc("et1", capa="estatuto_trabajadores", territorio="estatal", articulo="Art. 26 ET", rerank_score=0.69, text="Salario"),
            self._doc("et2", capa="estatuto_trabajadores", territorio="estatal", articulo="Art. 36 ET", rerank_score=0.67, text="Nocturnidad"),
        ]

        role_ctx = "camarero | Camarero/a (Área Funcional 3ª Restauración) | Área Funcional 3ª"

        sel_rest = store.rerank_and_select_by_territory(
            candidates=common_cands,
            query="Trabajo como camarero en un restaurante ordinario en San Sebastián de La Gomera. ¿Salario, nocturnidad y divisor?",
            territorios=["santa_cruz_tenerife"],
            use_cohere_rerank=False,
            role_context=role_ctx,
        )
        rest_ids = {d.chunk_id for d in sel_rest}
        self.assertIn("tbl_cls3d_restauracion", rest_ids)
        self.assertNotIn("tbl_cls1_hoteles", rest_ids)
        self.assertNotIn("tbl_cls3_lujo", rest_ids)

        sel_hotel = store.rerank_and_select_by_territory(
            candidates=common_cands,
            query="Trabajo como camarero en un hotel de 4 estrellas en Tenerife. ¿Cuál es mi salario base?",
            territorios=["santa_cruz_tenerife"],
            use_cohere_rerank=False,
            role_context=role_ctx,
        )
        hotel_ids = {d.chunk_id for d in sel_hotel}
        self.assertIn("tbl_cls1_hoteles", hotel_ids)
        self.assertNotIn("tbl_cls3d_restauracion", hotel_ids)


class TestEvaluationCheckpointLogic(unittest.TestCase):
    """Tests deterministas en CPU para el checkpoint incremental por caso y reanudación multi-turno."""

    @staticmethod
    def _case(
        case_id: str,
        session_id: str,
        *,
        reset_session_before: bool = True,
    ) -> GoldenTestCase:
        return GoldenTestCase(
            case_id=case_id,
            category="guardrail_routing",
            session_id=session_id,
            reset_session_before=reset_session_before,
            query=f"Consulta {case_id}",
            expected_intent="HOSPITALITY_CORE",
            expected_route="ask_clarification",
            expected_territorio=None,
            ground_truth_reference=f"Referencia {case_id}",
        )

    @staticmethod
    def _cached(tc: GoldenTestCase, final_response: str = "OK") -> EvaluationCaseResult:
        return EvaluationCaseResult(
            case_id=tc.case_id,
            category=tc.category,
            session_id=tc.session_id,
            query=tc.query,
            effective_query=tc.query,
            expected_intent=tc.expected_intent,
            predicted_intent=tc.expected_intent,
            intent_match=True,
            expected_route=tc.expected_route,
            predicted_route=tc.expected_route,
            route_match=True,
            expected_territorio=tc.expected_territorio,
            predicted_territorio=tc.expected_territorio,
            territorio_match=True,
            num_retrieved_docs=0,
            retrieved_layers=[],
            retrieved_articles=[],
            layer_coverage_score=1.0,
            citation_recall_at_k=1.0,
            ground_truth_number_hit=1.0,
            zero_numerical_hallucination=True,
            verified_numbers=[],
            unverified_numbers=[],
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_recall_semantic=1.0,
            hierarchy_application_score=1.0,
            judge_critique="ok",
            judge_model_used="mock",
            node_timings_ms={"analyze_and_route": 10.0},
            total_latency_ms=10.0,
            final_response=final_response,
        )

    def test_checkpoint_skippable_ids_enforces_multiturn_session_chain_integrity(self) -> None:
        c1 = self._case("C1", "ses-1", reset_session_before=True)
        c2 = self._case("C2", "ses-m", reset_session_before=True)
        c3 = self._case("C3", "ses-m", reset_session_before=False)
        c4 = self._case("C4", "ses-m", reset_session_before=False)
        c5 = self._case("C5", "ses-2", reset_session_before=True)
        c6 = self._case("C6", "ses-2", reset_session_before=False)
        dataset = [c1, c2, c3, c4, c5, c6]

        cached = {tc.case_id: self._cached(tc) for tc in (c1, c2, c3, c5, c6)}
        skippable = _compute_skippable_case_ids(dataset, cached)
        self.assertEqual(skippable, {"C1", "C5", "C6"})

    def test_evaluate_golden_dataset_saves_atomically_and_resumes_after_midrun_error(self) -> None:
        t1 = self._case("T1", "ses-t1", reset_session_before=True)
        t2 = self._case("T2", "ses-t2", reset_session_before=True)
        t3 = self._case("T3", "ses-t3", reset_session_before=True)
        dataset = [t1, t2, t3]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "evaluation_results.json"

            calls_run1: list[str] = []

            def _run1_side_effect(*, query: str, session_id: str, **_: object) -> dict[str, object]:
                calls_run1.append(session_id)
                if session_id == "ses-t3":
                    raise RuntimeError("503 UNAVAILABLE")
                return {
                    "intent": "HOSPITALITY_CORE",
                    "route_decision": "ask_clarification",
                    "territorio_normalizado": None,
                    "retrieved_docs": [],
                    "audit_report": {"unverified_numbers": [], "verified_numbers": []},
                    "node_timings_ms": {"analyze_and_route": 12.0},
                    "final_response": f"Resp {session_id}",
                }

            with (
                patch("src.evaluation.evaluator.AiHorecaVectorStore", return_value=MagicMock()),
                patch("src.evaluation.evaluator.build_aihoreca_graph", return_value=MagicMock()),
                patch("src.evaluation.evaluator.run_legal_query", side_effect=_run1_side_effect),
            ):
                with self.assertRaises(RuntimeError):
                    evaluate_golden_dataset(
                        force_recompute=True,
                        output_path=out_path,
                        cases=dataset,
                        skip_semantic_judge=True,
                    )

            self.assertTrue(out_path.exists())
            self.assertFalse(out_path.with_name(f"{out_path.name}.tmp").exists())
            partial_data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual([c["case_id"] for c in partial_data["cases"]], ["T1", "T2"])

            calls_run2: list[str] = []

            def _run2_side_effect(*, query: str, session_id: str, **_: object) -> dict[str, object]:
                calls_run2.append(session_id)
                return {
                    "intent": "HOSPITALITY_CORE",
                    "route_decision": "ask_clarification",
                    "territorio_normalizado": None,
                    "retrieved_docs": [],
                    "audit_report": {"unverified_numbers": [], "verified_numbers": []},
                    "node_timings_ms": {"analyze_and_route": 15.0},
                    "final_response": f"Resp {session_id}",
                }

            with (
                patch("src.evaluation.evaluator.AiHorecaVectorStore", return_value=MagicMock()),
                patch("src.evaluation.evaluator.build_aihoreca_graph", return_value=MagicMock()),
                patch("src.evaluation.evaluator.run_legal_query", side_effect=_run2_side_effect),
            ):
                resumed_report = evaluate_golden_dataset(
                    force_recompute=False,
                    output_path=out_path,
                    cases=dataset,
                    skip_semantic_judge=True,
                )

            self.assertEqual(calls_run2, ["ses-t3"])
            self.assertEqual([c.case_id for c in resumed_report.cases], ["T1", "T2", "T3"])
            final_data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual([c["case_id"] for c in final_data["cases"]], ["T1", "T2", "T3"])

    def test_evaluate_golden_dataset_subset_force_preserves_other_cached_cases(self) -> None:
        t1 = self._case("T1", "ses-t1")
        t2 = self._case("T2", "ses-t2")
        t3 = self._case("T3", "ses-t3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "evaluation_results.json"
            seed = {
                "total_cases": 3,
                "generative_cases": 0,
                "guardrail_and_hitl_cases": 3,
                "intent_accuracy": 1.0,
                "route_accuracy": 1.0,
                "territorio_accuracy": 1.0,
                "mean_layer_coverage": 0.0,
                "mean_citation_recall_at_k": 0.0,
                "mean_ground_truth_number_hit": 0.0,
                "zero_hallucination_rate": 1.0,
                "mean_faithfulness": 1.0,
                "mean_answer_relevancy": 1.0,
                "mean_context_recall_semantic": 1.0,
                "mean_hierarchy_application_score": 1.0,
                "mean_routing_latency_ms": 10.0,
                "mean_retrieval_latency_ms": 0.0,
                "mean_synthesis_latency_ms": 0.0,
                "mean_total_latency_ms": 10.0,
                "cases": [self._cached(tc, final_response=f"Old {tc.case_id}").model_dump() for tc in (t1, t2, t3)],
            }
            out_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")

            with (
                patch("src.evaluation.evaluator.AiHorecaVectorStore", return_value=MagicMock()),
                patch("src.evaluation.evaluator.build_aihoreca_graph", return_value=MagicMock()),
                patch(
                    "src.evaluation.evaluator.run_legal_query",
                    return_value={
                        "intent": "HOSPITALITY_CORE",
                        "route_decision": "ask_clarification",
                        "territorio_normalizado": None,
                        "retrieved_docs": [],
                        "audit_report": {"unverified_numbers": [], "verified_numbers": []},
                        "node_timings_ms": {"analyze_and_route": 20.0},
                        "final_response": "Updated T2",
                    },
                ),
            ):
                subset_report = evaluate_golden_dataset(
                    force_recompute=True,
                    output_path=out_path,
                    cases=[t2],
                    skip_semantic_judge=True,
                    preserve_other_cached=True,
                )

            self.assertEqual([c.case_id for c in subset_report.cases], ["T2"])
            self.assertEqual(subset_report.cases[0].final_response, "Updated T2")
            persisted = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [(c["case_id"], c["final_response"]) for c in persisted["cases"]],
                [("T1", "Old T1"), ("T2", "Updated T2"), ("T3", "Old T3")],
            )


class TestCadizIngestionAndRetrieval(unittest.TestCase):
    """Invariantes CPU para el convenio de Cádiz y extracción tabular vectorial por clip."""

    def test_cadiz_territory_canonicalization_and_pretty_label(self) -> None:
        self.assertEqual(_canonicalize_llm_province("Cádiz"), "cadiz")
        self.assertEqual(_canonicalize_llm_province("Provincia de Cádiz"), "cadiz")
        self.assertEqual(_canonicalize_territory("Cádiz"), "cadiz")
        self.assertEqual(_pretty_territory_label("cadiz"), "Cádiz")

    def test_cadiz_silver_chunks_and_2026_table_selection(self) -> None:
        cadiz_md = PROCESSED_DATA_DIR / "convenio_cadiz.md"
        if not cadiz_md.exists():
            self.skipTest("convenio_cadiz.md no disponible en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = chunk_markdown_document(cadiz_md, boundaries_cache=cache)
        self.assertEqual(len(chunks), 87)

        texts_all = "\n".join(c.text for c in chunks)
        self.assertIn("ART. 7º.- NOCTURNIDAD", texts_all)
        self.assertIn("ART. 8º.- PAGAS EXTRAORDINARIAS", texts_all)
        self.assertIn("ART. 20º.- JORNADA DE TRABAJO", texts_all)
        self.assertIn("ART. 21º.- HORAS EXTRAORDINARIAS", texts_all)
        self.assertIn("Cocinero/a.", texts_all)
        self.assertIn("Camarero/a.", texts_all)

        anexo3_tables = [
            c
            for c in chunks
            if c.tipo_contenido == "tabla_salarial" and "ANEXO III" in c.titulo_o_capitulo
        ]
        self.assertEqual(len(anexo3_tables), 4)

        docs_anexo3 = [
            RetrievedDocument(
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                norma_titulo=c.norma_titulo,
                capa_normativa=c.capa_normativa,
                territorio=c.territorio,
                titulo_o_capitulo=c.titulo_o_capitulo,
                articulo=c.articulo,
                tipo_contenido=c.tipo_contenido,
                text=c.text,
                score=0.9,
            )
            for c in anexo3_tables
        ]
        selected_ids = AiHorecaVectorStore._select_preferred_monetary_tables(
            docs_anexo3,
            [],
            reference_date=date(2026, 11, 23),
        )
        selected_docs = [d for d in docs_anexo3 if d.chunk_id in selected_ids]
        self.assertEqual(len(selected_docs), 1)
        self.assertIn("TABLA SALARIAL 2026", selected_docs[0].articulo)
        self.assertIn("1.447,69", selected_docs[0].text)
        self.assertIn("1.411,04", selected_docs[0].text)
        self.assertIn("1.253,89", selected_docs[0].text)
        self.assertIn("3,02", selected_docs[0].text)

    def test_cadiz_pdf_prose_filter_and_clip_table_cpu(self) -> None:
        cadiz_pdf = RAW_DATA_DIR / "convenio_cadiz.pdf"
        if not cadiz_pdf.exists():
            self.skipTest("convenio_cadiz.pdf no disponible en data/raw/")

        doc = pymupdf.open(cadiz_pdf)
        try:
            self.assertFalse(_page_has_complex_table(doc[1]))  # P02 prosa con decimales inline
            self.assertFalse(_page_has_complex_table(doc[4]))  # P05 prosa con importes seguro/natalidad
            p10_md = _extract_vector_tables_page_cpu(doc[9], layout="double_column")
            self.assertIsNotNone(p10_md)
            assert p10_md is not None
            missing, total, _ = _count_missing_spanish_decimals(doc[9].get_text("text"), p10_md)
            self.assertGreater(total, 50)
            self.assertEqual(missing, 0)
            self.assertIn("| GRUPO 3 | 1.447,69 | 1.292,71 | 1.272,06 |", p10_md)
            self.assertIn("Encargado/a de Lavandería/Lencería", p10_md)
            self.assertIn("Dependiente/a 1ª/1º.", p10_md)
            self.assertNotIn("| g y g |", p10_md)
        finally:
            doc.close()

    def test_cadiz_dual_classification_anexo1_and_anexo2_selected_together(self) -> None:
        cadiz_md = PROCESSED_DATA_DIR / "convenio_cadiz.md"
        if not cadiz_md.exists():
            self.skipTest("convenio_cadiz.md no disponible en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = chunk_markdown_document(cadiz_md, boundaries_cache=cache)
        anexo1 = next(c for c in chunks if c.titulo_o_capitulo == "ANEXO I")
        anexo2 = next(c for c in chunks if c.titulo_o_capitulo == "ANEXO II")
        tbl25 = next(c for c in chunks if c.articulo == "TABLA SALARIAL 2025")
        tbl26 = next(c for c in chunks if c.articulo == "TABLA SALARIAL 2026")
        art7 = next(c for c in chunks if c.articulo.startswith("ART. 7º"))
        art8 = next(c for c in chunks if c.articulo.startswith("ART. 8º"))
        art14 = next(c for c in chunks if c.articulo.startswith("ART. 14º"))
        art16 = next(c for c in chunks if c.articulo.startswith("ART. 16º"))
        art20 = next(c for c in chunks if c.articulo.startswith("ART. 20º"))
        art21 = next(c for c in chunks if c.articulo.startswith("ART. 21º"))

        def _to_doc(ch: LegalChunk, rscore: float) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=rscore,
                rerank_score=rscore,
            )

        anexo1_doc = _to_doc(anexo1, 0.01)
        anexo2_doc = _to_doc(anexo2, 0.02)
        self.assertTrue(AiHorecaVectorStore._is_classification_level_table(anexo1_doc))
        self.assertTrue(AiHorecaVectorStore._is_classification_level_table(anexo2_doc))

        state_docs = [
            RetrievedDocument(
                chunk_id=f"aleh_{i}",
                source_file="aleh_vi_estatal.md",
                norma_titulo="VI ALEH",
                capa_normativa="aleh_estatal",
                territorio="estatal",
                titulo_o_capitulo="ALEH",
                articulo=f"Art. {i} ALEH",
                tipo_contenido="articulo",
                text="Clasificación profesional estatal.",
                score=0.4,
                rerank_score=0.4,
            )
            for i in range(1, 3)
        ] + [
            RetrievedDocument(
                chunk_id=f"et_{i}",
                source_file="estatuto_trabajadores_consolidado.md",
                norma_titulo="ET",
                capa_normativa="estatuto_trabajadores",
                territorio="estatal",
                titulo_o_capitulo="ET",
                articulo=f"Art. {i} ET",
                tipo_contenido="articulo",
                text="Normativa estatal supletoria.",
                score=0.35,
                rerank_score=0.35,
            )
            for i in range(1, 3)
        ]

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        pool = [
            _to_doc(art14, 0.90),
            _to_doc(art16, 0.85),
            _to_doc(tbl25, 0.82),
            _to_doc(tbl26, 0.80),
            _to_doc(art20, 0.78),
            _to_doc(art21, 0.76),
            _to_doc(art7, 0.74),
            _to_doc(art8, 0.72),
            anexo2_doc,
            anexo1_doc,
            *state_docs,
        ]

        with patch.object(
            AiHorecaVectorStore,
            "_select_preferred_monetary_tables",
            return_value={tbl26.chunk_id},
        ):
            sel_salary = store.rerank_and_select_by_territory(
                candidates=pool,
                query="Soy cocinero en un hotel de 4 estrellas en Jerez de la Frontera (Cádiz), ¿cuál es mi salario base mensual vigente y cuántas pagas extraordinarias me corresponden?",
                territorios=["cadiz"],
                use_cohere_rerank=False,
                role_context="Cocinero/a (Área Funcional 2ª Cocina y Economato, Grupo Profesional II)",
            )
            ids_salary = {d.chunk_id for d in sel_salary}
            self.assertIn(tbl26.chunk_id, ids_salary)
            self.assertIn(anexo1.chunk_id, ids_salary)
            self.assertIn(anexo2.chunk_id, ids_salary)
            self.assertIn(art8.chunk_id, ids_salary)

            sel_cond = store.rerank_and_select_by_territory(
                candidates=pool,
                query="Trabajo como camarero en un restaurante de 1 tenedor en Tarifa (Cádiz). ¿Cuál es mi salario base mensual vigente, cuál es la jornada máxima anual, a partir de qué hora y con qué recargo se paga la nocturnidad, y qué recargo tienen las horas extraordinarias?",
                territorios=["cadiz"],
                use_cohere_rerank=False,
                role_context="Camarero/a (Área Funcional 3ª Restaurante, Sala, Bar y Similares, Grupo Profesional II)",
            )
            ids_cond = {d.chunk_id for d in sel_cond}
            self.assertIn(tbl26.chunk_id, ids_cond)
            self.assertIn(anexo1.chunk_id, ids_cond)
            self.assertIn(anexo2.chunk_id, ids_cond)
            self.assertIn(art20.chunk_id, ids_cond)
            self.assertIn(art7.chunk_id, ids_cond)
            self.assertIn(art21.chunk_id, ids_cond)

    def test_part1_priority_and_backfill_excludes_stale_monetary_tables(self) -> None:
        cadiz_md = PROCESSED_DATA_DIR / "convenio_cadiz.md"
        if not cadiz_md.exists():
            self.skipTest("convenio_cadiz.md no disponible en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = chunk_markdown_document(cadiz_md, boundaries_cache=cache)
        anexo1 = next(c for c in chunks if c.titulo_o_capitulo == "ANEXO I")
        anexo2 = next(c for c in chunks if c.titulo_o_capitulo == "ANEXO II")
        tbl25 = next(c for c in chunks if c.articulo == "TABLA SALARIAL 2025")
        tbl26 = next(c for c in chunks if c.articulo == "TABLA SALARIAL 2026")
        tbl27 = next(c for c in chunks if c.articulo == "TABLA SALARIAL 2027")
        art7 = next(c for c in chunks if c.articulo.startswith("ART. 7º"))
        art21 = next(c for c in chunks if c.articulo.startswith("ART. 21º"))
        art20_parts = [c for c in chunks if c.articulo.startswith("ART. 20º")]
        self.assertGreaterEqual(len(art20_parts), 2)
        art20_p1 = next(c for c in art20_parts if "(Parte 1/" in c.articulo)
        art20_p2 = next(c for c in art20_parts if "(Parte 2/" in c.articulo)

        def _to_doc(ch: LegalChunk, rscore: float) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=rscore,
                rerank_score=rscore,
            )

        aleh_docs = [
            RetrievedDocument(
                chunk_id=f"aleh_f_{i}",
                source_file="aleh_vi_estatal.md",
                norma_titulo="VI ALEH",
                capa_normativa="aleh_estatal",
                territorio="estatal",
                titulo_o_capitulo="ALEH",
                articulo=f"Art. {i} ALEH",
                tipo_contenido="articulo",
                text="Normativa estatal.",
                score=0.3,
                rerank_score=0.3,
            )
            for i in range(1, 3)
        ]
        et_docs = [
            RetrievedDocument(
                chunk_id=f"et_f_{i}",
                source_file="estatuto_trabajadores_consolidado.md",
                norma_titulo="ET",
                capa_normativa="estatuto_trabajadores",
                territorio="estatal",
                titulo_o_capitulo="ET",
                articulo=f"Art. {i} ET",
                tipo_contenido="articulo",
                text="Normativa básica estatal.",
                score=0.28,
                rerank_score=0.28,
            )
            for i in range(1, 3)
        ]

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        base_prov = [
            _to_doc(tbl27, 0.96),
            _to_doc(tbl25, 0.94),
            _to_doc(art20_p2, 0.92),  # Parte 2 con mayor score que Parte 1
            _to_doc(art7, 0.88),
            _to_doc(art21, 0.86),
            _to_doc(art20_p1, 0.70),  # Parte 1 con "1.800" horas
            _to_doc(tbl26, 0.65),
            _to_doc(anexo2, 0.50),
            _to_doc(anexo1, 0.45),
        ]
        extra_arts = [
            _to_doc(ch, 0.25)
            for ch in chunks
            if ch.tipo_contenido == "articulo"
            and ch.chunk_id not in {art20_p1.chunk_id, art20_p2.chunk_id, art7.chunk_id, art21.chunk_id}
        ][:5]
        pool_full = [*base_prov, *aleh_docs, *et_docs]
        pool_backfill = [*base_prov, *aleh_docs, *extra_arts]
        query_str = "Camarero restaurante Tarifa Cádiz salario, jornada máxima anual, nocturnidad y horas extraordinarias"

        with patch.object(
            AiHorecaVectorStore,
            "_select_preferred_monetary_tables",
            return_value={tbl26.chunk_id},
        ):
            sel_full = store.rerank_and_select_by_territory(
                candidates=pool_full,
                query=query_str,
                territorios=["cadiz"],
                use_cohere_rerank=False,
                role_context="Camarero/a (Área Funcional 3ª)",
            )
            ids_full = {d.chunk_id for d in sel_full}
            self.assertIn(art20_p1.chunk_id, ids_full)
            self.assertNotIn(art20_p2.chunk_id, ids_full)
            self.assertIn(tbl26.chunk_id, ids_full)
            self.assertNotIn(tbl25.chunk_id, ids_full)
            self.assertNotIn(tbl27.chunk_id, ids_full)

            sel_backfill = store.rerank_and_select_by_territory(
                candidates=pool_backfill,
                query=query_str,
                territorios=["cadiz"],
                use_cohere_rerank=False,
                role_context="Camarero/a (Área Funcional 3ª)",
            )
            ids_backfill = {d.chunk_id for d in sel_backfill}
            self.assertEqual(len(sel_backfill), 10)
            self.assertIn(tbl26.chunk_id, ids_backfill)
            self.assertNotIn(tbl25.chunk_id, ids_backfill)
            self.assertNotIn(tbl27.chunk_id, ids_backfill)


class TestGranadaIngestionAndRetrieval(unittest.TestCase):
    """Tests en CPU sobre la ingesta, clasificación, corrección de errores y tablas 2026-2028 de Granada."""

    def test_clock_hours_do_not_trigger_complex_table_flag(self) -> None:
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        prose = (
            "Artículo 40.- PLUS NOCTURNIDAD\n"
            "Se considera trabajo nocturno el realizado entre las 22.00 horas y las 06.00 horas.\n"
            "1) HOTELES:\n"
            "- Entre las 22,00 h y las 23,00 h., sin incremento alguno.\n"
            "- Entre las 23,00 h. y las 6,00 h., incremento del 15 % de su salario.\n"
            "- Costa del 1 de junio al 30 de septiembre entre las 0,00 h y las 6,00 h.\n"
            "2) Resto de sectores:\n"
            "- Entre las 22,00 h y las 0,00 h., sin incremento alguno.\n"
            "- Entre las 0,00 h. y las 6,00 h., incremento del 15% de su salario.\n"
        )
        page.insert_text((40, 60), prose, fontsize=10)
        self.assertFalse(_page_has_complex_table(page))
        doc.close()

    def test_granada_strict_validation_errata_level_iv_and_year_selection(self) -> None:
        base_path = PROCESSED_DATA_DIR / "convenio_granada.md"
        tablas_path = PROCESSED_DATA_DIR / "convenio_granada_tablas_2026_2028.md"
        if not base_path.exists() or not tablas_path.exists():
            self.skipTest("Convenio de Granada no disponible en data/processed/")

        base_report = validate_document(base_path, strict=True)
        tablas_report = validate_document(tablas_path, strict=True)
        self.assertTrue(base_report.passed, msg=base_report.model_dump_json(indent=2))
        self.assertEqual(base_report.error_count, 0)
        self.assertEqual(base_report.warning_count, 0)
        self.assertTrue(tablas_report.passed, msg=tablas_report.model_dump_json(indent=2))
        self.assertEqual(tablas_report.error_count, 0)
        self.assertEqual(tablas_report.warning_count, 0)

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        base_chunks = chunk_markdown_document(base_path, boundaries_cache=cache)
        cls_chunks = []
        for ch in base_chunks:
            doc_view = RetrievedDocument(
                chunk_id=ch.chunk_id,
                score=1.0,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                norma_titulo=ch.norma_titulo,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                source_file=ch.source_file,
                text=ch.text,
            )
            if AiHorecaVectorStore._is_classification_level_table(doc_view):
                cls_chunks.append(ch)
        self.assertTrue(
            any("59" in c.articulo and "Hoteles de 4 y 5 estrellas" in c.text for c in cls_chunks),
            msg=[c.articulo for c in cls_chunks],
        )

        tablas_chunks = chunk_markdown_document(tablas_path, boundaries_cache=cache)
        tablas_docs = [
            RetrievedDocument(
                chunk_id=ch.chunk_id,
                score=1.0,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                norma_titulo=ch.norma_titulo,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                source_file=ch.source_file,
                text=ch.text,
            )
            for ch in tablas_chunks
        ]

        # Verificar corrección de errores del BOP n.º 214: CAMARERA DE PISOS en NIVEL IV (no en NIVEL V)
        chunk_2026 = next(
            d for d in tablas_docs if d.tipo_contenido == "tabla_salarial" and "ANEXO I" in d.titulo_o_capitulo and "OTROS CONCEPTOS" not in d.articulo
        )
        nivel_iv_block = chunk_2026.text.split("| NIVEL IV |", 1)[1].split("### NIVEL V", 1)[0]
        nivel_v_block = chunk_2026.text.split("### NIVEL V", 1)[1]
        self.assertIn("1.737,50", nivel_iv_block)
        self.assertIn("1.489,03", nivel_iv_block)
        self.assertIn("CAMARERA DE PISOS", nivel_iv_block)
        self.assertNotIn("CAMARERA DE PISOS", nivel_v_block)

        otros_years = {
            AiHorecaVectorStore._extract_chunk_year(d)
            for d in tablas_docs
            if "OTROS CONCEPTOS RETRIBUTIVOS" in d.articulo
        }
        self.assertEqual(otros_years, {2026, 2027, 2028})

        mon_docs = [d for d in tablas_docs if d.tipo_contenido == "tabla_salarial" and AiHorecaVectorStore._is_monetary_table(d)]
        preferred_2026 = AiHorecaVectorStore._select_preferred_monetary_tables(
            mon_docs,
            [],
            reference_date=date(2026, 11, 23),
        )
        self.assertEqual(len(preferred_2026), 2)
        self.assertEqual(
            {AiHorecaVectorStore._extract_chunk_year(d) for d in mon_docs if d.chunk_id in preferred_2026},
            {2026},
        )

    def test_granada_retrieval_preserves_main_salary_table_with_aux_concepts_and_art59(self) -> None:
        base_path = PROCESSED_DATA_DIR / "convenio_granada.md"
        tablas_path = PROCESSED_DATA_DIR / "convenio_granada_tablas_2026_2028.md"
        if not base_path.exists() or not tablas_path.exists():
            self.skipTest("Convenio de Granada no disponible en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        base_chunks = chunk_markdown_document(base_path, boundaries_cache=cache)
        tablas_chunks = chunk_markdown_document(tablas_path, boundaries_cache=cache)

        def _to_doc(ch: LegalChunk, rscore: float) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=rscore,
                rerank_score=rscore,
            )

        art40 = next(c for c in base_chunks if c.articulo.startswith("Artículo 40"))
        art15 = next(c for c in base_chunks if c.articulo.startswith("Artículo 15"))
        art18 = next(c for c in base_chunks if c.articulo.startswith("Artículo 18"))
        art33 = next(c for c in base_chunks if c.articulo.startswith("Artículo 33"))
        art34 = next(c for c in base_chunks if c.articulo.startswith("Artículo 34"))
        art59 = next(c for c in base_chunks if "59" in c.articulo)
        main_2026 = next(c for c in tablas_chunks if c.tipo_contenido == "tabla_salarial" and "ANEXO I" in c.titulo_o_capitulo and "OTROS CONCEPTOS" not in c.articulo)
        otros_2026 = next(c for c in tablas_chunks if c.tipo_contenido == "tabla_salarial" and "ANEXO I" in c.titulo_o_capitulo and "OTROS CONCEPTOS" in c.articulo)
        main_2027 = next(c for c in tablas_chunks if c.tipo_contenido == "tabla_salarial" and "ANEXO II" in c.titulo_o_capitulo and "OTROS CONCEPTOS" not in c.articulo)

        # Artículo 40 con horarios 22,00 h / 23,00 h / 0,00 h / 6,00 h NO es tabla monetaria
        self.assertFalse(AiHorecaVectorStore._is_monetary_table(_to_doc(art40, 0.5)))
        self.assertTrue(AiHorecaVectorStore._is_auxiliary_concepts_table(_to_doc(otros_2026, 0.5)))
        self.assertFalse(AiHorecaVectorStore._is_auxiliary_concepts_table(_to_doc(main_2026, 0.5)))

        aleh_docs = [
            RetrievedDocument(
                chunk_id=f"aleh_gr_{i}",
                source_file="aleh_vi_estatal.md",
                norma_titulo="VI ALEH",
                capa_normativa="aleh_estatal",
                territorio="estatal",
                titulo_o_capitulo="ALEH",
                articulo=f"Art. {i} ALEH",
                tipo_contenido="articulo",
                text="Normativa estatal.",
                score=0.2,
                rerank_score=0.2,
            )
            for i in range(1, 3)
        ]
        et_docs = [
            RetrievedDocument(
                chunk_id=f"et_gr_{i}",
                source_file="estatuto_trabajadores_consolidado.md",
                norma_titulo="ET",
                capa_normativa="estatuto_trabajadores",
                territorio="estatal",
                titulo_o_capitulo="ET",
                articulo=f"Art. {i} ET",
                tipo_contenido="articulo",
                text="Normativa básica estatal.",
                score=0.18,
                rerank_score=0.18,
            )
            for i in range(1, 3)
        ]
        extra_arts = [
            _to_doc(c, 0.25)
            for c in base_chunks
            if c.tipo_contenido == "articulo"
            and c.chunk_id not in {art40.chunk_id, art15.chunk_id, art18.chunk_id, art33.chunk_id, art34.chunk_id, art59.chunk_id}
        ][:3]

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)
        # Escenario GD_37: Art. 59 y OTROS CONCEPTOS puntúan por encima de TABLAS SALARIALES 2026
        pool_37 = [
            _to_doc(art59, 0.98),
            _to_doc(otros_2026, 0.96),
            _to_doc(main_2027, 0.94),
            _to_doc(art34, 0.89),
            _to_doc(art33, 0.85),
            _to_doc(main_2026, 0.55),
            *extra_arts,
            *aleh_docs,
            *et_docs,
        ]
        with patch.object(
            AiHorecaVectorStore,
            "_select_preferred_monetary_tables",
            return_value={main_2026.chunk_id, otros_2026.chunk_id},
        ):
            sel_37 = store.rerank_and_select_by_territory(
                candidates=pool_37,
                query="Trabajo como camarera de pisos en un hotel de 4 estrellas en Granada capital. ¿Qué salario base mensual y nivel retributivo me corresponden, cuántas pagas extraordinarias tengo al año y cuánto es el plus cultural?",
                territorios=["granada"],
                use_cohere_rerank=False,
                role_context="Camarera de pisos (Área Funcional 4ª Pisos y Limpieza, Grupo Profesional II)",
            )
            ids_37 = {d.chunk_id for d in sel_37}
            self.assertIn(main_2026.chunk_id, ids_37)
            self.assertIn(otros_2026.chunk_id, ids_37)
            self.assertIn(art59.chunk_id, ids_37)
            self.assertIn(art33.chunk_id, ids_37)
            self.assertIn(art34.chunk_id, ids_37)
            self.assertNotIn(main_2027.chunk_id, ids_37)

            # Escenario GD_38: Art. 15, Art. 18, Art. 40, Art. 59 y TABLAS SALARIALES 2026
            pool_38 = [
                _to_doc(otros_2026, 0.90),
                _to_doc(art40, 0.88),
                _to_doc(art18, 0.84),
                _to_doc(art15, 0.82),
                _to_doc(main_2026, 0.62),
                _to_doc(art59, 0.45),
                *extra_arts,
                *aleh_docs,
                *et_docs,
            ]
            sel_38 = store.rerank_and_select_by_territory(
                candidates=pool_38,
                query="Soy camarero en un restaurante de 2 tenedores en Motril (Granada) en 2026. ¿Cuál es mi salario base mensual, cuál es la jornada laboral y el descanso semanal, cómo se retribuye el plus de nocturnidad si trabajo hasta las 02:00 y cómo se calculan y abonan las horas extraordinarias?",
                territorios=["granada"],
                use_cohere_rerank=False,
                role_context="Camarero/a (Área Funcional 3ª Restaurante, Sala, Bar y Similares, Grupo Profesional II)",
            )
            ids_38 = {d.chunk_id for d in sel_38}
            self.assertIn(main_2026.chunk_id, ids_38)
            self.assertIn(art59.chunk_id, ids_38)
            self.assertIn(art15.chunk_id, ids_38)
            self.assertIn(art18.chunk_id, ids_38)
            self.assertIn(art40.chunk_id, ids_38)


class TestGuipuzcoaIngestionAndRetrieval(unittest.TestCase):
    """Tests deterministas en CPU para Guipúzcoa (bilingüe BOG y doble convenio subsectorial)."""

    def test_coofficial_language_filter_distinguishes_basque_from_spanish(self) -> None:
        self.assertTrue(_is_coofficial_non_spanish_text("V. ERANSKINA"))
        self.assertTrue(_is_coofficial_non_spanish_text("LEHEN EREMU FUNTZIONALA"))
        self.assertTrue(_is_coofficial_non_spanish_text("2026KO SOLDATA-TAULAK"))
        self.assertFalse(_is_coofficial_non_spanish_text("ANEXO II TABLAS SALARIALES AÑO 2026"))
        self.assertFalse(_is_coofficial_non_spanish_text("Camarero/a de pisos"))

    def test_guipuzcoa_territory_and_subsector_routing_filters(self) -> None:
        for alias in ("Donostia", "San Sebastián", "Donostia-San Sebastián", "Gipuzkoa", "Guipúzcoa"):
            self.assertEqual(_canonicalize_territory(alias), "guipuzcoa")
            self.assertEqual(_canonicalize_llm_province(alias), "guipuzcoa")

        self.assertIn("guipuzcoa", AiHorecaVectorStore.get_split_subsector_territories())
        self.assertEqual(
            AiHorecaVectorStore._extract_query_agreement_subsector(
                "Trabajo como camarera de pisos en un hotel de 4 estrellas en Donostia."
            ),
            "alojamiento",
        )
        self.assertEqual(
            AiHorecaVectorStore._extract_query_agreement_subsector(
                "Soy cocinero en un restaurante en Irún (Gipuzkoa)."
            ),
            "restauracion",
        )
        self.assertIsNone(
            AiHorecaVectorStore._extract_query_agreement_subsector(
                "¿Cuántos días de vacaciones tengo en hostelería en Gipuzkoa?"
            )
        )

        flt_aloj = AiHorecaVectorStore._build_filter(
            capa_normativa="convenio_provincial",
            provincia="guipuzcoa",
            subsector="alojamiento",
        )
        self.assertIsNotNone(flt_aloj)
        keys_vals = {
            cond.key: getattr(cond.match, "value", None)
            for cond in (flt_aloj.must or [])
        }
        self.assertEqual(keys_vals.get("provincias_aplicables"), "guipuzcoa")
        self.assertEqual(keys_vals.get("subsector"), "alojamiento")

        flt_rest = AiHorecaVectorStore._build_filter(
            capa_normativa="convenio_provincial",
            provincia="Donostia",
            subsector="restauracion",
        )
        self.assertIsNotNone(flt_rest)
        keys_vals_rest = {
            cond.key: getattr(cond.match, "value", None)
            for cond in (flt_rest.must or [])
        }
        self.assertEqual(keys_vals_rest.get("provincias_aplicables"), "guipuzcoa")
        self.assertEqual(keys_vals_rest.get("subsector"), "restauracion")

    def test_guipuzcoa_strict_validation_and_chunk_metadata(self) -> None:
        files_expected = [
            (
                "convenio_guipuzcoa_alojamientos.md",
                "alojamiento",
                "Convenio Colectivo de Alojamientos de Guipúzcoa",
            ),
            (
                "convenio_guipuzcoa_alojamientos_tablas_2026.md",
                "alojamiento",
                "Convenio Colectivo de Alojamientos de Guipúzcoa",
            ),
            (
                "convenio_guipuzcoa_restauracion.md",
                "restauracion",
                "Convenio Colectivo de Empresas de Restauración de Guipúzcoa",
            ),
            (
                "convenio_guipuzcoa_restauracion_tablas_2026.md",
                "restauracion",
                "Convenio Colectivo de Empresas de Restauración de Guipúzcoa",
            ),
        ]
        if not all((PROCESSED_DATA_DIR / fn).exists() for fn, _, _ in files_expected):
            self.skipTest("Convenios de Guipúzcoa no disponibles en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        for fname, expected_sub, expected_title in files_expected:
            report = validate_document(PROCESSED_DATA_DIR / fname, strict=True)
            self.assertTrue(report.passed, msg=f"{fname} falló validación estricta: {report.issues}")
            self.assertEqual(report.error_count, 0, msg=f"{fname} tiene errores: {report.issues}")
            self.assertEqual(report.warning_count, 0, msg=f"{fname} tiene warnings: {report.issues}")

            chunks = chunk_markdown_document(PROCESSED_DATA_DIR / fname, boundaries_cache=cache)
            self.assertGreater(len(chunks), 0)
            self.assertTrue(all(c.territorio == "guipuzcoa" for c in chunks))
            self.assertTrue(all(c.subsector == expected_sub for c in chunks))
            self.assertTrue(all(c.norma_titulo == expected_title for c in chunks))

        aloj_t_chunks = chunk_markdown_document(
            PROCESSED_DATA_DIR / "convenio_guipuzcoa_alojamientos_tablas_2026.md",
            boundaries_cache=cache,
        )
        for ch in aloj_t_chunks:
            if ch.tipo_contenido == "tabla_salarial":
                self.assertIn("2026", ch.titulo_o_capitulo)

        rest_t_chunks = chunk_markdown_document(
            PROCESSED_DATA_DIR / "convenio_guipuzcoa_restauracion_tablas_2026.md",
            boundaries_cache=cache,
        )
        for ch in rest_t_chunks:
            if ch.tipo_contenido == "tabla_salarial":
                self.assertIn("2026", ch.titulo_o_capitulo)

    def test_guipuzcoa_retrieval_selection_for_alojamientos_and_restauracion(self) -> None:
        required_files = [
            "convenio_guipuzcoa_alojamientos.md",
            "convenio_guipuzcoa_alojamientos_tablas_2026.md",
            "convenio_guipuzcoa_restauracion.md",
            "convenio_guipuzcoa_restauracion_tablas_2026.md",
        ]
        if not all((PROCESSED_DATA_DIR / fn).exists() for fn in required_files):
            self.skipTest("Convenios de Guipúzcoa no disponibles en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        aloj_base = chunk_markdown_document(PROCESSED_DATA_DIR / "convenio_guipuzcoa_alojamientos.md", boundaries_cache=cache)
        aloj_2026 = chunk_markdown_document(PROCESSED_DATA_DIR / "convenio_guipuzcoa_alojamientos_tablas_2026.md", boundaries_cache=cache)
        rest_base = chunk_markdown_document(PROCESSED_DATA_DIR / "convenio_guipuzcoa_restauracion.md", boundaries_cache=cache)
        rest_2026 = chunk_markdown_document(PROCESSED_DATA_DIR / "convenio_guipuzcoa_restauracion_tablas_2026.md", boundaries_cache=cache)

        def _to_doc(ch: LegalChunk, rscore: float) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                subsector=ch.subsector,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=rscore,
                rerank_score=rscore,
            )

        aleh_docs = [
            RetrievedDocument(
                chunk_id=f"aleh_gi_{i}",
                source_file="aleh_vi_estatal.md",
                norma_titulo="VI ALEH",
                capa_normativa="aleh_estatal",
                territorio="estatal",
                titulo_o_capitulo="ALEH",
                articulo=f"Art. {i} ALEH",
                tipo_contenido="articulo",
                text="Normativa estatal.",
                score=0.2,
                rerank_score=0.2,
            )
            for i in range(1, 3)
        ]
        et_docs = [
            RetrievedDocument(
                chunk_id=f"et_gi_{i}",
                source_file="estatuto_trabajadores_consolidado.md",
                norma_titulo="ET",
                capa_normativa="estatuto_trabajadores",
                territorio="estatal",
                titulo_o_capitulo="ET",
                articulo=f"Art. {i} ET",
                tipo_contenido="articulo",
                text="Normativa básica estatal.",
                score=0.18,
                rerank_score=0.18,
            )
            for i in range(1, 3)
        ]

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)

        # Escenario GD_39 (Alojamientos)
        aloj_anexo1 = next(c for c in aloj_base if c.articulo.startswith("ANEXO I (Parte 1"))
        aloj_art7 = next(c for c in aloj_base if c.articulo.startswith("Artículo 7"))
        aloj_art13 = next(c for c in aloj_base if c.articulo.startswith("Artículo 13") and "1.705" in c.text)
        aloj_art18 = next(c for c in aloj_base if c.articulo.startswith("Artículo 18"))
        aloj_main_2026 = next(c for c in aloj_2026 if c.titulo_o_capitulo == "ANEXO II | 2026")
        aloj_anexo6_2026 = next(c for c in aloj_2026 if c.titulo_o_capitulo == "ANEXO VI | 2026")
        self.assertTrue(AiHorecaVectorStore._is_auxiliary_concepts_table(_to_doc(aloj_anexo6_2026, 0.98)))
        self.assertFalse(AiHorecaVectorStore._is_auxiliary_concepts_table(_to_doc(aloj_main_2026, 0.60)))
        aloj_extra = [
            _to_doc(c, 0.25)
            for c in aloj_base
            if c.tipo_contenido == "articulo"
            and c.chunk_id not in {aloj_art7.chunk_id, aloj_art13.chunk_id, aloj_art18.chunk_id}
        ][:3]
        pool_39 = [
            _to_doc(aloj_anexo6_2026, 0.98),
            _to_doc(aloj_anexo1, 0.96),
            _to_doc(aloj_art7, 0.90),
            _to_doc(aloj_art13, 0.89),
            _to_doc(aloj_art18, 0.88),
            _to_doc(aloj_main_2026, 0.60),
            *aloj_extra,
            *aleh_docs,
            *et_docs,
        ]
        sel_39 = store.rerank_and_select_by_territory(
            candidates=pool_39,
            query="Trabajo como camarera de pisos en un hotel de 4 estrellas en Donostia - San Sebastián (Guipúzcoa). ¿Qué nivel salarial, salario mensual y anual me corresponden en 2026, cuál es mi jornada anual ordinaria, cuántas pagas extraordinarias tengo y cómo se retribuye la nocturnidad?",
            territorios=["guipuzcoa"],
            use_cohere_rerank=False,
            role_context="Camarera de pisos (Área Funcional 4ª)",
        )
        ids_39 = {d.chunk_id for d in sel_39}
        self.assertIn(aloj_anexo1.chunk_id, ids_39)
        self.assertIn(aloj_main_2026.chunk_id, ids_39)
        self.assertIn(aloj_art7.chunk_id, ids_39)
        self.assertIn(aloj_art13.chunk_id, ids_39)
        self.assertIn(aloj_art18.chunk_id, ids_39)

        # Escenario GD_40 (Restauración)
        rest_anexo2 = next(c for c in rest_base if c.articulo.startswith("ANEXO II (Parte 1"))
        rest_art14 = next(c for c in rest_base if c.articulo.startswith("Artículo 14"))
        rest_art18 = next(c for c in rest_base if c.articulo.startswith("Artículo 18"))
        rest_art28 = next(c for c in rest_base if c.articulo.startswith("Artículo 28") and "1.723" in c.text)
        rest_main_2026 = next(c for c in rest_2026 if "ANEXO III" in c.titulo_o_capitulo)
        rest_anexo5_2026 = next(c for c in rest_2026 if "ANEXO V" in c.titulo_o_capitulo)
        rest_extra = [
            _to_doc(c, 0.25)
            for c in rest_base
            if c.tipo_contenido == "articulo"
            and c.chunk_id not in {rest_art14.chunk_id, rest_art18.chunk_id, rest_art28.chunk_id}
        ][:2]
        pool_40 = [
            _to_doc(rest_anexo2, 0.96),
            _to_doc(rest_main_2026, 0.95),
            _to_doc(rest_anexo5_2026, 0.93),
            _to_doc(rest_art14, 0.90),
            _to_doc(rest_art18, 0.89),
            _to_doc(rest_art28, 0.88),
            *rest_extra,
            *aleh_docs,
            *et_docs,
        ]
        sel_40 = store.rerank_and_select_by_territory(
            candidates=pool_40,
            query="Soy cocinero en un restaurante en Irún (Gipuzkoa). ¿Qué nivel salarial y salario base mensual y anual me corresponden en 2026, cuál es la jornada anual ordinaria, cómo se regulan las pagas extraordinarias y cuánto es el plus de nocturnidad?",
            territorios=["guipuzcoa"],
            use_cohere_rerank=False,
            role_context="Cocinero/a (Área Funcional 2ª Cocina)",
        )
        ids_40 = {d.chunk_id for d in sel_40}
        self.assertIn(rest_anexo2.chunk_id, ids_40)
        self.assertIn(rest_main_2026.chunk_id, ids_40)
        self.assertIn(rest_anexo5_2026.chunk_id, ids_40)
        self.assertIn(rest_art14.chunk_id, ids_40)
        self.assertIn(rest_art18.chunk_id, ids_40)
        self.assertIn(rest_art28.chunk_id, ids_40)


class TestAlavaIngestionAndRetrieval(unittest.TestCase):
    """Tests deterministas en CPU para el Convenio Colectivo de Hostelería de Álava (BOTHA n.º 20/2026)."""

    def test_alava_territory_aliases_and_pretty_label(self) -> None:
        aliases = (
            "Álava",
            "alava",
            "Araba",
            "Araba/Álava",
            "Araba / Álava",
            "Álava/Araba",
            "Vitoria-Gasteiz",
            "Vitoria",
            "Gasteiz",
            "Llodio",
            "Laudio",
            "Amurrio",
        )
        for alias in aliases:
            self.assertEqual(_canonicalize_territory(alias), "alava")
            self.assertEqual(_canonicalize_llm_province(alias), "alava")
        self.assertEqual(_pretty_territory_label("alava"), "Álava")

    def test_alava_strict_validation_and_2025_vs_2026_table_selection(self) -> None:
        md_path = PROCESSED_DATA_DIR / "convenio_alava.md"
        if not md_path.exists():
            self.skipTest("Convenio de Álava no disponible en data/processed/")

        report = validate_document(md_path, strict=True)
        self.assertTrue(report.passed, msg=f"convenio_alava.md falló validación estricta: {report.issues}")
        self.assertEqual(report.error_count, 0, msg=f"Errores en convenio_alava.md: {report.issues}")
        self.assertEqual(report.warning_count, 0, msg=f"Warnings en convenio_alava.md: {report.issues}")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = chunk_markdown_document(md_path, boundaries_cache=cache)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.territorio == "alava" for c in chunks))
        self.assertTrue(all(c.subsector == "general" for c in chunks))

        t_2025 = next(c for c in chunks if "1 de diciembre de 2025" in c.articulo)
        t_2026 = next(c for c in chunks if "Tabla salarial año 2026" in c.articulo)
        self.assertNotEqual(t_2025.chunk_id, t_2026.chunk_id)

        def _to_doc(ch: LegalChunk) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                subsector=ch.subsector,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=0.8,
                rerank_score=0.8,
            )

        d25 = _to_doc(t_2025)
        d26 = _to_doc(t_2026)
        self.assertEqual(AiHorecaVectorStore._extract_chunk_year(d25), 2025)
        self.assertEqual(AiHorecaVectorStore._extract_chunk_year(d26), 2026)
        preferred = AiHorecaVectorStore._select_preferred_monetary_tables(
            [d25, d26],
            [],
            reference_date=date(2026, 11, 24),
        )
        self.assertEqual(preferred, {d26.chunk_id})

    def test_alava_retrieval_selection_gd41_and_gd42(self) -> None:
        md_path = PROCESSED_DATA_DIR / "convenio_alava.md"
        if not md_path.exists():
            self.skipTest("Convenio de Álava no disponible en data/processed/")

        cache = json.loads(BOUNDARIES_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = chunk_markdown_document(md_path, boundaries_cache=cache)

        def _to_doc(ch: LegalChunk, rscore: float) -> RetrievedDocument:
            return RetrievedDocument(
                chunk_id=ch.chunk_id,
                source_file=ch.source_file,
                norma_titulo=ch.norma_titulo,
                capa_normativa=ch.capa_normativa,
                territorio=ch.territorio,
                subsector=ch.subsector,
                titulo_o_capitulo=ch.titulo_o_capitulo,
                articulo=ch.articulo,
                tipo_contenido=ch.tipo_contenido,
                text=ch.text,
                score=rscore,
                rerank_score=rscore,
            )

        aleh_docs = [
            RetrievedDocument(
                chunk_id=f"aleh_vi_{i}",
                source_file="aleh_vi_estatal.md",
                norma_titulo="VI ALEH",
                capa_normativa="aleh_estatal",
                territorio="estatal",
                titulo_o_capitulo="ALEH",
                articulo=f"Art. {i} ALEH",
                tipo_contenido="articulo",
                text="Normativa estatal.",
                score=0.2,
                rerank_score=0.2,
            )
            for i in range(1, 3)
        ]
        et_docs = [
            RetrievedDocument(
                chunk_id=f"et_vi_{i}",
                source_file="estatuto_trabajadores_consolidado.md",
                norma_titulo="ET",
                capa_normativa="estatuto_trabajadores",
                territorio="estatal",
                titulo_o_capitulo="ET",
                articulo=f"Art. {i} ET",
                tipo_contenido="articulo",
                text="Normativa básica estatal.",
                score=0.18,
                rerank_score=0.18,
            )
            for i in range(1, 3)
        ]

        anexo2 = next(c for c in chunks if c.articulo == "Anexo II (Parte 2/4)")
        anexo2_part3 = next(c for c in chunks if c.articulo == "Anexo II (Parte 3/4)")
        t_2025 = next(c for c in chunks if "1 de diciembre de 2025" in c.articulo)
        t_2026 = next(c for c in chunks if "Tabla salarial año 2026" in c.articulo)
        art6 = next(c for c in chunks if c.articulo.startswith("Artículo 6º"))
        art8 = next(c for c in chunks if c.articulo.startswith("Artículo 8º"))
        art11 = next(c for c in chunks if c.articulo.startswith("Artículo 11º"))
        art14 = next(c for c in chunks if c.articulo.startswith("Artículo 14º"))
        art15 = next(c for c in chunks if c.articulo.startswith("Artículo 15º"))
        art16 = next(c for c in chunks if c.articulo.startswith("Artículo 16º"))
        art17 = next(c for c in chunks if c.articulo.startswith("Artículo 17º"))
        art18 = next(c for c in chunks if c.articulo.startswith("Artículo 18."))
        extra = [
            _to_doc(c, 0.22)
            for c in chunks
            if c.tipo_contenido == "articulo"
            and c.chunk_id
            not in {
                art6.chunk_id,
                art8.chunk_id,
                art11.chunk_id,
                art14.chunk_id,
                art15.chunk_id,
                art16.chunk_id,
                art17.chunk_id,
                art18.chunk_id,
            }
        ][:2]

        store = AiHorecaVectorStore.__new__(AiHorecaVectorStore)

        self.assertTrue(store._matches_primary_role_stem(_to_doc(anexo2, 0.5), ["camarer"]))
        self.assertFalse(store._matches_primary_role_stem(_to_doc(anexo2_part3, 0.5), ["camarer"]))
        self.assertTrue(store._matches_primary_role_stem(_to_doc(anexo2_part3, 0.5), ["camarer", "pis"]))

        # Escenario GD_41: art11 y art6 con score superior a art8 para validar rescate por descanso semanal
        pool_41 = [
            _to_doc(t_2025, 0.99),
            _to_doc(art11, 0.98),
            _to_doc(anexo2, 0.96),
            _to_doc(art18, 0.90),
            _to_doc(art17, 0.88),
            _to_doc(art6, 0.85),
            _to_doc(t_2026, 0.65),
            _to_doc(art8, 0.60),
            *extra,
            *aleh_docs,
            *et_docs,
        ]
        sel_41 = store.rerank_and_select_by_territory(
            candidates=pool_41,
            query="Trabajo como cocinero en un hotel de 4 estrellas en Vitoria-Gasteiz (Álava). ¿Cuál es mi nivel salarial, salario base, complemento en especie y bruto mensual en 2026, cuántas pagas extraordinarias tengo, cuánto es el Bonus 31 de octubre y cuántos días de descanso semanal me corresponden?",
            territorios=["alava"],
            use_cohere_rerank=False,
            role_context="Cocinero/a (Área Funcional 2ª Cocina, Grupo Profesional II)",
        )
        ids_41 = {d.chunk_id for d in sel_41}
        self.assertIn(anexo2.chunk_id, ids_41)
        self.assertIn(t_2026.chunk_id, ids_41)
        self.assertNotIn(t_2025.chunk_id, ids_41)
        self.assertIn(art8.chunk_id, ids_41)
        self.assertIn(art17.chunk_id, ids_41)
        self.assertIn(art18.chunk_id, ids_41)

        # Escenario GD_42: anexo2_part3 con score superior a anexo2 para validar prioridad de puesto principal
        pool_42 = [
            _to_doc(t_2025, 0.995),
            _to_doc(anexo2_part3, 0.99),
            _to_doc(art6, 0.92),
            _to_doc(art14, 0.90),
            _to_doc(art15, 0.89),
            _to_doc(art16, 0.87),
            _to_doc(anexo2, 0.70),
            _to_doc(t_2026, 0.65),
            *extra,
            *aleh_docs,
            *et_docs,
        ]
        sel_42 = store.rerank_and_select_by_territory(
            candidates=pool_42,
            query="Soy camarero en un restaurante en Llodio (Araba) en 2026. ¿Qué nivel salarial y salario base mensual me corresponden, cuál es la jornada laboral anual, cómo se retribuye la nocturnidad si trabajo hasta las 02:00, cuánto vale cada hora extraordinaria y cuál es el plus de domingos y festivos?",
            territorios=["alava"],
            use_cohere_rerank=False,
            role_context="Camarero/a (Área Funcional 3ª Restaurante, Grupo Profesional II)",
        )
        ids_42 = {d.chunk_id for d in sel_42}
        self.assertIn(anexo2.chunk_id, ids_42)
        self.assertNotIn(anexo2_part3.chunk_id, ids_42)
        self.assertIn(t_2026.chunk_id, ids_42)
        self.assertNotIn(t_2025.chunk_id, ids_42)
        self.assertIn(art6.chunk_id, ids_42)
        self.assertIn(art14.chunk_id, ids_42)
        self.assertIn(art15.chunk_id, ids_42)
        self.assertIn(art16.chunk_id, ids_42)


if __name__ == "__main__":
    unittest.main()
