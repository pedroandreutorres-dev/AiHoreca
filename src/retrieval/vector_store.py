"""Motor de Base de Datos Vectorial Híbrida en Qdrant (Vectores Densos Gemini + BM25 FastEmbed + Cohere Rerank)."""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cohere
import numpy as np
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from src.config import (
    COHERE_RERANK_MODEL,
    EMBEDDING_MODEL,
    QDRANT_STORAGE_DIR,
    QUOTA_ALEH_CHUNKS,
    QUOTA_ESTATUTO_CHUNKS,
    QUOTA_PROVINCIAL_ARTICLES,
    QUOTA_PROVINCIAL_TABLES,
    RETRIEVAL_FINAL_TOP_K,
    RETRIEVAL_TOP_K_PER_LAYER,
    get_genai_client,
    load_environment,
)
from src.retrieval.chunker import chunk_all_processed_documents
from src.retrieval.schemas import LegalChunk, RetrievedDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "aihoreca_legal_corpus"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
DENSE_DIMENSION = 3072


class AiHorecaVectorStore:
    """Gestiona la indexación y búsqueda híbrida jerárquica en Qdrant local (100% CPU)."""

    def __init__(
        self,
        storage_dir: Path = QDRANT_STORAGE_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        load_environment()
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Registro global a nivel de proceso (`sys`) para reutilizar la instancia de QdrantLocal
        # ante recargas en caliente de Streamlit (@st.cache_resource / importlib.reload) sin colisión de portalocker.
        qdrant_registry: dict[str, QdrantClient] = getattr(sys, "_aihoreca_qdrant_clients", {})
        setattr(sys, "_aihoreca_qdrant_clients", qdrant_registry)
        storage_key = str(self.storage_dir.resolve())

        if storage_key in qdrant_registry and qdrant_registry[storage_key] is not None:
            self.qdrant = qdrant_registry[storage_key]
        else:
            self.qdrant = QdrantClient(path=str(self.storage_dir))
            qdrant_registry[storage_key] = self.qdrant

        self.genai_client: genai.Client = get_genai_client()
        self._sparse_model: Optional[SparseTextEmbedding] = getattr(
            sys, "_aihoreca_sparse_model", None
        )

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """Inicializa perezosamente en CPU el modelo léxico BM25 de FastEmbed y lo conserva en `sys`."""
        if self._sparse_model is None:
            cached = getattr(sys, "_aihoreca_sparse_model", None)
            if cached is not None:
                self._sparse_model = cached
            else:
                logger.info("Cargando modelo disperso FastEmbed 'Qdrant/bm25' en CPU...")
                self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                setattr(sys, "_aihoreca_sparse_model", self._sparse_model)
        return self._sparse_model

    def ensure_collection(self, recreate: bool = False) -> None:
        """Crea o verifica la colección híbrida (Dense 3072 + Sparse BM25 IDF) e índices de payload."""
        exists = self.qdrant.collection_exists(self.collection_name)
        if exists and recreate:
            logger.info("Eliminando colección previa '%s' para recrearla...", self.collection_name)
            self.qdrant.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creando colección híbrida '%s' (dense=%d Cosine + sparse=BM25 IDF)...",
                self.collection_name,
                DENSE_DIMENSION,
            )
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=DENSE_DIMENSION,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )

    def _embed_dense_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Genera embeddings densos en lote con gemini-embedding-001 respetando el límite de 30.000 TPM del Free Tier."""
        last_err: Optional[Exception] = None
        for attempt in range(8):
            try:
                resp = self.genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors = [list(e.values) for e in resp.embeddings]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Discrepancia de cardinalidad en embeddings: esperados={len(texts)}, recibidos={len(vectors)}"
                    )
                return vectors
            except Exception as exc:
                last_err = exc
                err_str = str(exc)
                wait_s = 22.0 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 5.0
                logger.warning(
                    "Ventana TPM (30k tokens/min) de Free Tier alcanzada en %s (intento %d/8). Esperando %.0fs...",
                    EMBEDDING_MODEL,
                    attempt + 1,
                    wait_s,
                )
                time.sleep(wait_s)
        raise RuntimeError(f"Error al generar embeddings densos: {last_err}") from last_err

    def _embed_sparse_batch(self, texts: list[str]) -> list[models.SparseVector]:
        """Genera vectores dispersos BM25 en CPU mediante FastEmbed."""
        sparse_embeddings = list(self.sparse_model.embed(texts))
        return [
            models.SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            for emb in sparse_embeddings
        ]

    def index_chunks(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 96,
        recreate: bool = False,
    ) -> int:
        """Indexa de forma incremental todos los LegalChunk en Qdrant combinando Dense (gemini-embedding-001) y BM25 en CPU."""
        self.ensure_collection(recreate=recreate)

        existing_points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(pt.id) for pt in existing_points}

        pending_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not pending_chunks:
            logger.info(
                "La colección '%s' ya contiene los %d chunks indexados.",
                self.collection_name,
                len(existing_ids),
            )
            return len(existing_ids)

        logger.info(
            "Chunks ya presentes en Qdrant: %d | Chunks pendientes de indexar: %d",
            len(existing_ids),
            len(pending_chunks),
        )

        # Agrupar en lotes acotados tanto por número de ítems (<=40) como por caracteres (<=55.000 chars ~ 14k tokens < 30k TPM)
        batches: list[list[LegalChunk]] = []
        current_batch: list[LegalChunk] = []
        current_chars = 0
        max_batch_chars = 55000
        max_items = min(batch_size, 40)

        for ch in pending_chunks:
            ch_len = len(ch.text)
            if current_batch and (
                len(current_batch) >= max_items or current_chars + ch_len > max_batch_chars
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(ch)
            current_chars += ch_len
        if current_batch:
            batches.append(current_batch)

        processed_so_far = 0
        total_pending = len(pending_chunks)
        for b_idx, batch in enumerate(batches, start=1):
            batch_texts = [c.text for c in batch]
            logger.info(
                "Indexando lote %d/%d (%d chunks, %d chars | %d/%d pendientes)...",
                b_idx,
                len(batches),
                len(batch),
                sum(len(t) for t in batch_texts),
                processed_so_far + len(batch),
                total_pending,
            )
            dense_vecs = self._embed_dense_batch(
                batch_texts, task_type="RETRIEVAL_DOCUMENT"
            )
            sparse_vecs = self._embed_sparse_batch(batch_texts)

            points: list[models.PointStruct] = []
            for chunk, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs, strict=True):
                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            DENSE_VECTOR_NAME: d_vec,
                            SPARSE_VECTOR_NAME: s_vec,
                        },
                        payload=chunk.model_dump(),
                    )
                )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            processed_so_far += len(batch)

        final_count = self.qdrant.count(collection_name=self.collection_name).count
        logger.info(
            "Indexación completada: %d puntos almacenados en '%s'.",
            final_count,
            self.collection_name,
        )
        return final_count

    def close(self) -> None:
        """Cierra limpiamente el cliente local de Qdrant y limpia el registro de proceso."""
        qdrant_registry: dict[str, QdrantClient] = getattr(sys, "_aihoreca_qdrant_clients", {})
        storage_key = str(self.storage_dir.resolve())
        qdrant_registry.pop(storage_key, None)
        try:
            self.qdrant.close()
        except Exception:
            pass

    @staticmethod
    def _build_filter(
        capa_normativa: Optional[str] = None,
        provincia: Optional[str] = None,
        tipo_contenido: Optional[str] = None,
    ) -> Optional[models.Filter]:
        """Construye el filtro determinista de metadatos (payload) para Qdrant."""
        must_conditions: list[models.Condition] = []
        if capa_normativa:
            must_conditions.append(
                models.FieldCondition(
                    key="capa_normativa",
                    match=models.MatchValue(value=capa_normativa),
                )
            )
        if provincia:
            norm_prov = (
                provincia.lower()
                .strip()
                .replace("á", "a")
                .replace("é", "e")
                .replace("í", "i")
                .replace("ó", "o")
                .replace("ú", "u")
                .replace("ñ", "n")
            )
            must_conditions.append(
                models.FieldCondition(
                    key="provincias_aplicables",
                    match=models.MatchValue(value=norm_prov),
                )
            )
        if tipo_contenido:
            must_conditions.append(
                models.FieldCondition(
                    key="tipo_contenido",
                    match=models.MatchValue(value=tipo_contenido),
                )
            )
        return models.Filter(must=must_conditions) if must_conditions else None

    def hybrid_search(
        self,
        query: str,
        capa_normativa: Optional[str] = None,
        provincia: Optional[str] = None,
        tipo_contenido: Optional[str] = None,
        top_k: int = 6,
        dense_query_vec: Optional[list[float]] = None,
        sparse_query_vec: Optional[models.SparseVector] = None,
    ) -> list[RetrievedDocument]:
        """Ejecuta búsqueda híbrida nativa en Qdrant (Prefetch Dense + Prefetch BM25 -> Fusión RRF)."""
        d_vec = (
            dense_query_vec
            if dense_query_vec is not None
            else self._embed_dense_batch([query], task_type="RETRIEVAL_QUERY")[0]
        )
        s_vec = (
            sparse_query_vec
            if sparse_query_vec is not None
            else self._embed_sparse_batch([query])[0]
        )
        q_filter = self._build_filter(
            capa_normativa=capa_normativa,
            provincia=provincia,
            tipo_contenido=tipo_contenido,
        )

        q_dense = np.asarray(d_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_dense)) or 1.0

        prefetch_limit = max(top_k * 3, 15)
        result = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=d_vec,
                    using=DENSE_VECTOR_NAME,
                    filter=q_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=s_vec,
                    using=SPARSE_VECTOR_NAME,
                    filter=q_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=[DENSE_VECTOR_NAME],
        )

        docs: list[RetrievedDocument] = []
        for pt in result.points:
            payload = pt.payload or {}
            rrf_score = float(pt.score or 0.0)
            cos_score = rrf_score
            if isinstance(pt.vector, dict) and DENSE_VECTOR_NAME in pt.vector:
                d_arr = np.asarray(pt.vector[DENSE_VECTOR_NAME], dtype=np.float32)
                d_norm = float(np.linalg.norm(d_arr)) or 1.0
                cos_score = float(np.dot(q_dense, d_arr) / (q_norm * d_norm))
            calibrated = round(0.75 * cos_score + 0.25 * rrf_score, 4)
            docs.append(
                RetrievedDocument(
                    chunk_id=str(pt.id),
                    score=rrf_score,
                    rerank_score=calibrated,
                    capa_normativa=payload.get("capa_normativa", "convenio_provincial"),
                    territorio=payload.get("territorio", ""),
                    norma_titulo=payload.get("norma_titulo", ""),
                    titulo_o_capitulo=payload.get("titulo_o_capitulo", ""),
                    articulo=payload.get("articulo", ""),
                    tipo_contenido=payload.get("tipo_contenido", "articulo"),
                    source_file=payload.get("source_file", ""),
                    text=payload.get("text", ""),
                )
            )
        return docs

    @staticmethod
    def _extract_chunk_year(doc: RetrievedDocument) -> int:
        """Extrae el año específico de una tabla salarial descartando rangos generales de anexo (ej. '2023-2025')."""
        header_surface = f"{doc.titulo_o_capitulo} | {doc.articulo}"
        # Eliminamos rangos generales tipo '2023-2025' o '2023 - 2025' para quedarnos con el año concreto de la sub-tabla
        cleaned = re.sub(r"\b2\.?02[2-9]\s*[-–—/]\s*2\.?02[2-9]\b", "", header_surface)
        years: list[int] = []
        for m in re.finditer(r"\b2\.?0(2[3-9])\b", cleaned):
            years.append(2000 + int(m.group(1)))
        if years:
            return years[-1]
        # Si no estaba en el título/artículo, inspeccionamos las primeras líneas del cuerpo tras la cabecera [NORMA:...]
        body_start = doc.text.split("\n\n", 1)[-1][:250]
        cleaned_body = re.sub(r"\b2\.?02[2-9]\s*[-–—/]\s*2\.?02[2-9]\b", "", body_start)
        for m in re.finditer(r"\b2\.?0(2[3-9])\b", cleaned_body):
            years.append(2000 + int(m.group(1)))
        return years[-1] if years else 0

    def parallel_hierarchical_retrieve(
        self,
        query: str,
        provincia: Optional[str] = None,
        top_k_per_layer: int = RETRIEVAL_TOP_K_PER_LAYER,
        final_top_k: int = RETRIEVAL_FINAL_TOP_K,
        use_cohere_rerank: bool = True,
        rerank_query: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """Recupera en paralelo de las 3 capas normativas (ET + ALEH + Convenio Provincial estratificado) y reordena con Cohere."""
        # Calculamos el vector de consulta una sola vez (1 llamada API densa + 1 cálculo BM25 en CPU)
        d_vec = self._embed_dense_batch([query], task_type="RETRIEVAL_QUERY")[0]
        s_vec = self._embed_sparse_batch([query])[0]

        # Detectar si el usuario solicitó explícitamente un año histórico concreto en la consulta
        explicit_years = [2000 + int(m.group(1)) for m in re.finditer(r"\b2\.?0(2[3-9])\b", query)]
        is_remuneration_query = bool(
            re.search(
                r"(?i)\b(?:sueldo|salario|cobrar|cobro|n[óo]mina|retribuci[óo]n|pagas?|horas?\s+extras?|extraordinarias?|nocturnidad|plus)\b",
                query,
            )
        )

        candidates: list[RetrievedDocument] = []
        seen_ids: set[str] = set()

        def _append_unique(items: list[RetrievedDocument]) -> None:
            for item in items:
                if item.chunk_id not in seen_ids:
                    candidates.append(item)
                    seen_ids.add(item.chunk_id)

        # Capa 1: Estatuto de los Trabajadores (Mínimo legal estatal)
        _append_unique(
            self.hybrid_search(
                query=query,
                capa_normativa="estatuto_trabajadores",
                top_k=top_k_per_layer,
                dense_query_vec=d_vec,
                sparse_query_vec=s_vec,
            )
        )

        # Capa 2: VI Acuerdo Laboral Estatal de Hostelería (ALEH)
        _append_unique(
            self.hybrid_search(
                query=query,
                capa_normativa="aleh_estatal",
                top_k=top_k_per_layer,
                dense_query_vec=d_vec,
                sparse_query_vec=s_vec,
            )
        )

        # Capa 3: Convenio Colectivo Provincial / Autonómico estratificado (articulado + tablas salariales + disposiciones)
        # Reutiliza d_vec y s_vec en memoria (0 llamadas extra a la API de embeddings)
        if provincia:
            _append_unique(
                self.hybrid_search(
                    query=query,
                    capa_normativa="convenio_provincial",
                    provincia=provincia,
                    tipo_contenido="articulo",
                    top_k=max(top_k_per_layer + 2, 8),
                    dense_query_vec=d_vec,
                    sparse_query_vec=s_vec,
                )
            )
            if is_remuneration_query:
                # Búsqueda léxica local en CPU (0 llamadas API) para asegurar que tanto el artículo de Gratificaciones Extraordinarias / Pagas Extras
                # como los artículos de Horas Extraordinarias / Nocturnidad / Jornada estén siempre entre los candidatos antes del Rerank
                pay_sparse_vec = self._embed_sparse_batch(
                    [f"{query} gratificaciones extraordinarias pagas extras horas extraordinarias nocturnidad jornada anual"]
                )[0]
                _append_unique(
                    self.hybrid_search(
                        query=query,
                        capa_normativa="convenio_provincial",
                        provincia=provincia,
                        tipo_contenido="articulo",
                        top_k=5,
                        dense_query_vec=d_vec,
                        sparse_query_vec=pay_sparse_vec,
                    )
                )
            raw_tables = self.hybrid_search(
                query=query,
                capa_normativa="convenio_provincial",
                provincia=provincia,
                tipo_contenido="tabla_salarial",
                top_k=max(top_k_per_layer * 3, 12),
                dense_query_vec=d_vec,
                sparse_query_vec=s_vec,
            )
            # Si es un convenio multiprovincial (ej. Cataluña: barcelona, girona, tarragona, lleida),
            # descartamos tablas salariales exclusivas de otra provincia hermana para asegurar que entren
            # tanto la tabla de niveles como la tabla de importes en € de la provincia consultada.
            norm_p = provincia.lower().strip()
            sibling_provs = {"barcelona", "girona", "tarragona", "lleida"}
            other_siblings = sibling_provs - {norm_p} if norm_p in sibling_provs else set()
            filtered_tables = [
                t
                for t in raw_tables
                if not any(
                    sib in f"{t.titulo_o_capitulo} {t.articulo}".lower()
                    for sib in other_siblings
                )
                and (
                    norm_p != "girona"
                    or "a) barcelona" not in f"{t.titulo_o_capitulo} {t.articulo}".lower()
                )
            ]
            # En el DOGC del Convenio de Cataluña existe una errata tipográfica oficial ("A) Girona" en lugar de "C) Girona"
            # para la tabla salarial de 2025). Como no contiene ningún nombre de otra provincia hermana, pasa el filtro anterior.
            # Prioridad Temporal General: si el usuario NO pide un año concreto, priorizamos las tablas con el año vigente más reciente
            # (ej. 2025/2026 sobre 2023/2024) manteniendo su orden de relevancia semántica dentro del mismo año.
            euro_tables = [t for t in filtered_tables if "€" in t.text or "euros" in t.text.lower()]
            level_tables = [t for t in filtered_tables if t not in euro_tables]
            if not explicit_years and euro_tables:
                max_avail_year = max((self._extract_chunk_year(t) for t in euro_tables), default=0)
                if max_avail_year > 0:
                    latest_euro = [t for t in euro_tables if self._extract_chunk_year(t) == max_avail_year]
                    older_euro = [t for t in euro_tables if t not in latest_euro]
                    euro_tables = latest_euro + older_euro

            balanced_tables: list[RetrievedDocument] = []
            for seq in (euro_tables[:3], level_tables[:2], filtered_tables):
                for item in seq:
                    if item not in balanced_tables:
                        balanced_tables.append(item)
            _append_unique((balanced_tables or raw_tables)[: top_k_per_layer + 2])
            _append_unique(
                self.hybrid_search(
                    query=query,
                    capa_normativa="convenio_provincial",
                    provincia=provincia,
                    tipo_contenido="anexo_o_disposicion",
                    top_k=2,
                    dense_query_vec=d_vec,
                    sparse_query_vec=s_vec,
                )
            )

        if not candidates:
            return []

        cohere_key = os.getenv("COHERE_API_KEY", "").strip()
        effective_rerank_query = (rerank_query or query).strip()
        if use_cohere_rerank and cohere_key and not cohere_key.startswith("..."):
            try:
                co_client = cohere.ClientV2(api_key=cohere_key)
                rerank_resp = co_client.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=effective_rerank_query,
                    documents=[c.text for c in candidates],
                    top_n=len(candidates),
                )
                reranked_all: list[RetrievedDocument] = []
                for item in rerank_resp.results:
                    doc = candidates[item.index].model_copy()
                    doc.rerank_score = float(item.relevance_score)
                    reranked_all.append(doc)
                candidates = reranked_all
            except Exception as exc:
                logger.warning(
                    "Aviso: fallo o clave ausente en Cohere Rerank (%s). Usando ordenación RRF híbrida.",
                    exc,
                )
                candidates.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        else:
            # Respaldo matemático calibrado entre capas (0.75 * Cosine global + 0.25 * RRF local)
            candidates.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)

        # Selección estratificada por capa normativa para evitar que el nombre de la provincia
        # monopolice el Top-K y deje sin representación al Estatuto o al VI ALEH
        if not provincia or len(candidates) <= final_top_k:
            return candidates[:final_top_k]

        selected: list[RetrievedDocument] = []
        selected_ids: set[str] = set()

        def _pick(predicate, limit: int) -> None:
            count = 0
            for doc in candidates:
                if count >= limit or len(selected) >= final_top_k:
                    break
                if doc.chunk_id not in selected_ids and predicate(doc):
                    selected.append(doc)
                    selected_ids.add(doc.chunk_id)
                    count += 1

        # Cuotas mínimas garantizadas de contexto jerárquico (parametrizadas desde src/config.py)
        if is_remuneration_query:
            # 1) Garantizar artículo de Gratificaciones Extraordinarias / Pagas Extras del convenio
            _pick(
                lambda d: d.capa_normativa == "convenio_provincial"
                and d.tipo_contenido == "articulo"
                and bool(re.search(r"(?i)gratificaciones\s+extraordinarias|pagas\s+extraordinarias", f"{d.articulo} {d.text}")),
                1,
            )
            # 2) Si la consulta pregunta además por un concepto específico (horas extras, nocturnidad, jornada, vacaciones),
            # garantizar que su artículo regulador específico también tenga plaza asegurada antes de llenar el resto de la cuota
            for specific_pat in (
                r"(?i)horas\s+extra(?:ordinaria)?s?",
                r"(?i)nocturnidad|horas\s+nocturnas",
                r"(?i)vacaciones",
                r"(?i)jornada\s+de\s+trabajo|tiempo\s+parcial",
            ):
                if re.search(specific_pat, f"{query} {effective_rerank_query}"):
                    _pick(
                        lambda d, pat=specific_pat: d.capa_normativa == "convenio_provincial"
                        and d.tipo_contenido == "articulo"
                        and bool(re.search(pat, f"{d.articulo} {d.text[:500]}")),
                        1,
                    )
        _pick(
            lambda d: d.capa_normativa == "convenio_provincial" and d.tipo_contenido == "articulo",
            max(0, QUOTA_PROVINCIAL_ARTICLES - sum(1 for s in selected if s.capa_normativa == "convenio_provincial" and s.tipo_contenido == "articulo")),
        )
        # En tablas salariales: si el usuario no pidió un año histórico específico, garantizamos primero 1 tabla en €
        # del año vigente más reciente disponible en los candidatos (ej. 2025/2026) y 1 segunda tabla complementaria
        prov_euro_tables = [
            d
            for d in candidates
            if d.capa_normativa == "convenio_provincial"
            and d.tipo_contenido == "tabla_salarial"
            and ("€" in d.text or "euros" in d.text.lower())
        ]
        max_cand_year = max((self._extract_chunk_year(d) for d in prov_euro_tables), default=0)
        if not explicit_years and max_cand_year > 0:
            _pick(
                lambda d: d.capa_normativa == "convenio_provincial"
                and d.tipo_contenido == "tabla_salarial"
                and ("€" in d.text or "euros" in d.text.lower())
                and self._extract_chunk_year(d) == max_cand_year,
                1,
            )
        _pick(
            lambda d: d.capa_normativa == "convenio_provincial"
            and d.tipo_contenido == "tabla_salarial"
            and ("€" in d.text or "euros" in d.text.lower()),
            1,
        )
        _pick(
            lambda d: d.capa_normativa == "convenio_provincial" and d.tipo_contenido == "tabla_salarial",
            max(0, QUOTA_PROVINCIAL_TABLES - sum(1 for s in selected if s.tipo_contenido == "tabla_salarial")),
        )
        _pick(lambda d: d.capa_normativa == "aleh_estatal", QUOTA_ALEH_CHUNKS)
        _pick(lambda d: d.capa_normativa == "estatuto_trabajadores", QUOTA_ESTATUTO_CHUNKS)
        # Rellenar huecos restantes hasta final_top_k con los mejores candidatos globales
        _pick(lambda _d: True, max(0, final_top_k - len(selected)))

        selected.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        return selected


def build_and_index_corpus(recreate: bool = False) -> tuple[AiHorecaVectorStore, list[LegalChunk]]:
    """Función principal para fragmentar los 5 Markdowns e indexarlos en Qdrant local."""
    chunks = chunk_all_processed_documents()
    store = AiHorecaVectorStore()
    store.index_chunks(chunks, recreate=recreate)
    return store, chunks
