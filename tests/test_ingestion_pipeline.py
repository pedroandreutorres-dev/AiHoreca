"""Suite de tests de regresión y canarios en CPU para el pipeline de ingesta y el Gate de Calidad.

Coste de tokens LLM = 0. Todos los tests operan determinísticamente sobre:
- heurísticas unitarias y adaptadores PDF en memoria (`TestPdfIngestorHeuristics`)
- páginas canario de los PDFs reales si están presentes en `data/raw/` (`TestRealPdfCanaries`)
- invariantes del manifiesto y del corpus Silver ya indexado (`TestManifestAndSilverCorpusInvariants`)
- trampas sintéticas de regresión para el validador (`TestValidatorDefectTraps`)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.config import BOUNDARIES_CACHE_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.ingestion.pdf_ingestor import (
    _clean_gemini_markdown_output,
    _extract_double_column_page_text,
    _format_plain_page_to_markdown,
    _has_clean_vector_table_grid,
    _is_scanned_page,
    _normalize_spanish_thousands,
    _page_has_complex_table,
)
from src.ingestion.schemas import DocumentManifestEntry
from src.ingestion.validator import validate_corpus, validate_document
from src.retrieval.chunker import chunk_all_processed_documents, chunk_markdown_document
from src.retrieval.schemas import RetrievedDocument
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

    def test_format_plain_page_promotes_real_articles_and_rejects_inline_citations(self) -> None:
        sample_page = "\n".join(
            [
                "42",
                "Boletín Oficial de la Provincia de Málaga Núm. 70 — 14/04/2026",
                "CAPÍTULO II",
                "Artículo 15. Salario base.",
                "Artículo 15 del vigente Estatuto de los Trabajadores regula este punto.",
                "artículo 34.8 ET permite solicitar adaptación.",
                "Capítulo décimo sobre disposiciones varias.",
                "Anexo I. Clasificación profesional",
            ]
        )
        md = _format_plain_page_to_markdown(sample_page)
        self.assertIn("## CAPÍTULO II", md)
        self.assertIn("### Artículo 15. Salario base.", md)
        self.assertIn("## Anexo I. Clasificación profesional", md)
        self.assertNotIn("### Artículo 15 del vigente Estatuto", md)
        self.assertNotIn("### artículo 34.8 ET", md)
        self.assertNotIn("## Capítulo décimo", md)
        self.assertNotIn("Boletín Oficial de la Provincia", md)
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


if __name__ == "__main__":
    unittest.main()
