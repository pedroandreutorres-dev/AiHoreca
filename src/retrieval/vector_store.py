"""Motor de Base de Datos Vectorial Híbrida en Qdrant (Vectores Densos Gemini + BM25 FastEmbed + Cohere Rerank)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import cohere
import numpy as np
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from src.config import (
    BOUNDARIES_CACHE_PATH,
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
    get_qdrant_config,
    load_environment,
)
from src.retrieval.chunker import _canonicalize_territory, chunk_all_processed_documents
from src.retrieval.schemas import LegalChunk, RetrievedDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "aihoreca_legal_corpus"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
DENSE_DIMENSION = 3072


class AiHorecaVectorStore:
    """Gestiona la indexación y búsqueda híbrida jerárquica en Qdrant (Local CPU o Qdrant Cloud)."""

    def __init__(
        self,
        storage_dir: Path = QDRANT_STORAGE_DIR,
        collection_name: Optional[str] = None,
        force_local: bool = False,
    ) -> None:
        load_environment()
        qdrant_url, qdrant_api_key, default_collection = get_qdrant_config()
        self.storage_dir = storage_dir
        self.collection_name = collection_name or default_collection
        self.is_cloud = bool(qdrant_url and not force_local)

        # Registro global a nivel de proceso (`sys`) para reutilizar la instancia de Qdrant
        # ante recargas en caliente de Streamlit (@st.cache_resource / importlib.reload) sin colisión de portalocker.
        qdrant_registry: dict[str, QdrantClient] = getattr(sys, "_aihoreca_qdrant_clients", {})
        setattr(sys, "_aihoreca_qdrant_clients", qdrant_registry)

        if self.is_cloud:
            storage_key = f"cloud::{qdrant_url}::{self.collection_name}"
        else:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            storage_key = str(self.storage_dir.resolve())

        self._storage_key = storage_key
        if storage_key in qdrant_registry and qdrant_registry[storage_key] is not None:
            self.qdrant = qdrant_registry[storage_key]
        else:
            if self.is_cloud:
                logger.info("Conectando a Qdrant Cloud (%s)...", qdrant_url)
                self.qdrant = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=60,
                )
            else:
                self.qdrant = QdrantClient(path=str(self.storage_dir))
            qdrant_registry[storage_key] = self.qdrant

        self.genai_client: genai.Client = get_genai_client()
        self._sparse_model: Optional[SparseTextEmbedding] = getattr(
            sys, "_aihoreca_sparse_model", None
        )
        self._embed_lock = threading.Lock()
        self._sparse_init_lock = threading.Lock()
        self._dense_inflight: dict[str, threading.Event] = {}
        self._sparse_inflight: dict[str, threading.Event] = {}
        self._dense_query_cache: dict[str, list[float]] = {}
        self._sparse_query_cache: dict[str, models.SparseVector] = {}

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """Inicializa perezosamente en CPU el modelo léxico BM25 de FastEmbed de forma thread-safe."""
        if self._sparse_model is None:
            with self._sparse_init_lock:
                if self._sparse_model is None:
                    cached = getattr(sys, "_aihoreca_sparse_model", None)
                    if cached is not None:
                        self._sparse_model = cached
                    else:
                        logger.info("Cargando modelo disperso FastEmbed 'Qdrant/bm25' en CPU...")
                        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                        setattr(sys, "_aihoreca_sparse_model", self._sparse_model)
        return self._sparse_model

    def _ensure_payload_indexes(self) -> None:
        """Crea índices de payload (KEYWORD) necesarios en Qdrant Cloud para filtros eficientes."""
        for field_name in (
            "capa_normativa",
            "provincias_aplicables",
            "tipo_contenido",
            "source_file",
            "territorio",
            "subsector",
        ):
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug("Índice de payload '%s' omitido/existente: %s", field_name, exc)

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
        self._ensure_payload_indexes()

    def _get_all_existing_ids(self) -> set[str]:
        """Recupera todos los IDs de la colección mediante scroll paginado (sin límite de 10k)."""
        existing_ids: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            for pt in points:
                existing_ids.add(str(pt.id))
            if next_offset is None or not points:
                break
        return existing_ids

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
                if "PerDay" in err_str or "RequestsPerDay" in err_str:
                    raise RuntimeError(
                        f"Cuota diaria agotada en {EMBEDDING_MODEL}: {err_str}"
                    ) from exc
                wait_s = 25.0 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 5.0
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

    def _singleflight_query_vector(
        self,
        key: str,
        cache: dict[str, Any],
        inflight: dict[str, threading.Event],
        compute_fn: Callable[[], Any],
    ) -> tuple[Any, bool]:
        """Caché single-flight thread-safe para compartir embeddings entre ramas `Send()` concurrentes."""
        while True:
            with self._embed_lock:
                if key in cache:
                    return cache[key], True
                wait_evt = inflight.get(key)
                if wait_evt is None:
                    own_evt = threading.Event()
                    inflight[key] = own_evt
                    break
            wait_evt.wait()

        try:
            vec = compute_fn()
            with self._embed_lock:
                if len(cache) >= 64:
                    oldest_key = next(iter(cache))
                    cache.pop(oldest_key, None)
                cache[key] = vec
            return vec, False
        finally:
            with self._embed_lock:
                inflight.pop(key, None)
                own_evt.set()

    def _get_query_dense_vec(
        self,
        query: str,
        usage_out: Optional[list[dict[str, Any]]] = None,
        step_name: str = "embed_query",
    ) -> list[float]:
        """Obtiene o calcula el embedding denso de una consulta evitando llamadas duplicadas en paralelo."""
        t0 = time.perf_counter()
        vec, from_cache = self._singleflight_query_vector(
            key=query,
            cache=self._dense_query_cache,
            inflight=self._dense_inflight,
            compute_fn=lambda: self._embed_dense_batch([query], task_type="RETRIEVAL_QUERY")[0],
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        if usage_out is not None:
            usage_out.append(
                {
                    "step": step_name,
                    "provider": "gemini_embedding",
                    "model": EMBEDDING_MODEL,
                    "status": "cache_hit" if from_cache else "ok",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0 if from_cache else 1,
                    "cache_hit": from_cache,
                    "chars": len(query or ""),
                    "latency_ms": elapsed_ms,
                }
            )
        return vec

    def _get_query_sparse_vec(self, query: str) -> models.SparseVector:
        """Obtiene o calcula el vector BM25 disperso de una consulta con caché single-flight en CPU."""
        vec, _ = self._singleflight_query_vector(
            key=query,
            cache=self._sparse_query_cache,
            inflight=self._sparse_inflight,
            compute_fn=lambda: self._embed_sparse_batch([query])[0],
        )
        return vec

    def index_chunks(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 96,
        recreate: bool = False,
        replace_source_files: Optional[list[str]] = None,
    ) -> int:
        """Indexa de forma incremental los LegalChunk en Qdrant (permite actualizar solo documentos específicos)."""
        self.ensure_collection(recreate=recreate)

        replace_mode = bool(replace_source_files and not recreate)
        if replace_mode:
            target_set = set(replace_source_files or [])
            pending_chunks = [c for c in chunks if c.source_file in target_set]
            existing_count = 0
        else:
            existing_ids = self._get_all_existing_ids()
            pending_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            existing_count = len(existing_ids)

        if not pending_chunks:
            count_now = (
                self.qdrant.count(collection_name=self.collection_name).count
                if replace_mode
                else existing_count
            )
            logger.info(
                "La colección '%s' ya contiene los %d chunks indexados.",
                self.collection_name,
                count_now,
            )
            return count_now

        logger.info(
            "Chunks ya presentes en Qdrant: %s | Chunks pendientes de indexar: %d",
            "reemplazo atómico" if replace_mode else str(existing_count),
            len(pending_chunks),
        )

        # Agrupar en lotes acotados por número de ítems (<=25) y caracteres (<=24.000 chars ~ <=14k tokens incluso en tablas numéricas densas < 30k TPM)
        batches: list[list[LegalChunk]] = []
        current_batch: list[LegalChunk] = []
        current_chars = 0
        max_batch_chars = 24000
        max_items = min(batch_size, 25)

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
        staged_points: list[models.PointStruct] = []
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

            if replace_mode:
                staged_points.extend(points)
            else:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            processed_so_far += len(batch)

        if replace_mode and staged_points:
            logger.info(
                "Actualización atómica: eliminando puntos previos de %s y subiendo %d puntos...",
                replace_source_files,
                len(staged_points),
            )
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchAny(any=replace_source_files),
                            )
                        ]
                    )
                ),
            )
            for idx in range(0, len(staged_points), 30):
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=staged_points[idx : idx + 30],
                )

        final_count = self.qdrant.count(collection_name=self.collection_name).count
        logger.info(
            "Indexación completada: %d puntos almacenados en '%s'.",
            final_count,
            self.collection_name,
        )
        return final_count

    def close(self) -> None:
        """Cierra limpiamente el cliente Qdrant (Local o Cloud) y limpia el registro de proceso."""
        qdrant_registry: dict[str, QdrantClient] = getattr(sys, "_aihoreca_qdrant_clients", {})
        storage_key = getattr(self, "_storage_key", str(self.storage_dir.resolve()))
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
        subsector: Optional[str] = None,
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
            prov_tokens = [
                p.strip(" _")
                for p in re.split(r"[,/|]|\s+y\s+", provincia)
                if p.strip(" _")
            ]
            norm_provincias = list(
                dict.fromkeys(
                    canon
                    for tok in prov_tokens
                    if (canon := _canonicalize_territory(tok))
                )
            )
            if len(norm_provincias) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="provincias_aplicables",
                        match=models.MatchValue(value=norm_provincias[0]),
                    )
                )
            elif len(norm_provincias) > 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="provincias_aplicables",
                        match=models.MatchAny(any=norm_provincias),
                    )
                )
        if tipo_contenido:
            must_conditions.append(
                models.FieldCondition(
                    key="tipo_contenido",
                    match=models.MatchValue(value=tipo_contenido),
                )
            )
        if subsector:
            must_conditions.append(
                models.FieldCondition(
                    key="subsector",
                    match=models.MatchValue(value=subsector),
                )
            )
        return models.Filter(must=must_conditions) if must_conditions else None

    def hybrid_search(
        self,
        query: str,
        capa_normativa: Optional[str] = None,
        provincia: Optional[str] = None,
        tipo_contenido: Optional[str] = None,
        subsector: Optional[str] = None,
        top_k: int = 6,
        dense_query_vec: Optional[list[float]] = None,
        sparse_query_vec: Optional[models.SparseVector] = None,
        include_vectors: bool = True,
        usage_out: Optional[list[dict[str, Any]]] = None,
        step_name: str = "embed_hybrid_search",
    ) -> list[RetrievedDocument]:
        """Ejecuta búsqueda híbrida nativa en Qdrant (Prefetch Dense + Prefetch BM25 -> Fusión RRF)."""
        d_vec = (
            dense_query_vec
            if dense_query_vec is not None
            else self._get_query_dense_vec(query, usage_out=usage_out, step_name=step_name)
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
            subsector=subsector,
        )

        q_dense = np.asarray(d_vec, dtype=np.float32) if include_vectors else None
        q_norm = (float(np.linalg.norm(q_dense)) or 1.0) if q_dense is not None else 1.0

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
            with_vectors=[DENSE_VECTOR_NAME] if include_vectors else False,
        )

        docs: list[RetrievedDocument] = []
        for pt in result.points:
            rrf_score = float(pt.score or 0.0)
            cos_score = rrf_score
            if q_dense is not None and isinstance(pt.vector, dict) and DENSE_VECTOR_NAME in pt.vector:
                d_arr = np.asarray(pt.vector[DENSE_VECTOR_NAME], dtype=np.float32)
                d_norm = float(np.linalg.norm(d_arr)) or 1.0
                cos_score = float(np.dot(q_dense, d_arr) / (q_norm * d_norm))
            calibrated = round(0.75 * cos_score + 0.25 * rrf_score, 4) if include_vectors else round(rrf_score, 4)
            docs.append(
                self._point_to_retrieved_doc(
                    pt,
                    score=rrf_score,
                    calibrated_score=calibrated,
                )
            )
        return docs

    @staticmethod
    def _point_to_retrieved_doc(
        pt: Any,
        *,
        score: float,
        calibrated_score: float,
    ) -> RetrievedDocument:
        """Convierte un punto devuelto por Qdrant en un `RetrievedDocument` normalizado."""
        payload = pt.payload or {}
        return RetrievedDocument(
            chunk_id=str(pt.id),
            score=score,
            rerank_score=calibrated_score,
            capa_normativa=payload.get("capa_normativa", "convenio_provincial"),
            territorio=payload.get("territorio", ""),
            subsector=payload.get("subsector") or "general",
            norma_titulo=payload.get("norma_titulo", ""),
            titulo_o_capitulo=payload.get("titulo_o_capitulo", ""),
            articulo=payload.get("articulo", ""),
            tipo_contenido=payload.get("tipo_contenido", "articulo"),
            source_file=payload.get("source_file", ""),
            text=payload.get("text", ""),
        )

    def sparse_topic_search(
        self,
        topic_query: str,
        *,
        capa_normativa: str,
        provincia: str,
        tipo_contenido: str = "articulo",
        subsector: Optional[str] = None,
        top_k: int = 2,
    ) -> list[RetrievedDocument]:
        """Ejecuta búsqueda dispersa BM25 pura por tema en Qdrant sin llamadas a embeddings densos."""
        if getattr(self, "qdrant", None) is None:
            return []
        sparse_vec = self._get_query_sparse_vec(topic_query)
        q_filter = self._build_filter(
            capa_normativa=capa_normativa,
            provincia=provincia,
            tipo_contenido=tipo_contenido,
            subsector=subsector,
        )
        result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            self._point_to_retrieved_doc(
                pt,
                score=float(pt.score or 0.0),
                calibrated_score=round(float(pt.score or 0.0), 4),
            )
            for pt in result.points
        ]

    @staticmethod
    def _extract_chunk_year(doc: RetrievedDocument) -> int:
        """Extrae el año específico de una tabla salarial descartando rangos generales de anexo (ej. '2023-2025') e importes en euros."""
        header_surface = f"{doc.titulo_o_capitulo} | {doc.articulo}"
        # Eliminamos rangos generales tipo '2023-2025' o '2023 - 2025' para quedarnos con el año concreto de la sub-tabla
        cleaned = re.sub(r"\b2\.?02[2-9]\s*[-–—/]\s*2\.?02[2-9]\b(?!,\d)", "", header_surface)
        years: list[int] = []
        for m in re.finditer(r"\b2\.?0(2[3-9])\b(?!,\d)", cleaned):
            years.append(2000 + int(m.group(1)))
        if years:
            return years[-1]
        # Si no estaba en el título/artículo, inspeccionamos todo el cuerpo del fragmento tras la cabecera [NORMA:...]
        body = doc.text.split("\n\n", 1)[-1]
        cleaned_body = re.sub(r"\b2\.?02[2-9]\s*[-–—/]\s*2\.?02[2-9]\b(?!,\d)", "", body)
        for m in re.finditer(r"\b2\.?0(2[3-9])\b(?!,\d)", cleaned_body):
            years.append(2000 + int(m.group(1)))
        return max(years) if years else 0

    _SPANISH_MONTHS: dict[str, int] = {
        "ene": 1, "enero": 1, "xaneiro": 1,
        "feb": 2, "febrero": 2, "febreiro": 2,
        "mar": 3, "marzo": 3,
        "abr": 4, "abril": 4,
        "may": 5, "mayo": 5, "maio": 5,
        "jun": 6, "junio": 6, "xuño": 6, "xuno": 6,
        "jul": 7, "julio": 7, "xullo": 7,
        "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "septiembre": 9, "setiembre": 9, "setembro": 9,
        "oct": 10, "octubre": 10, "outubro": 10,
        "nov": 11, "noviembre": 11, "novembro": 11,
        "dic": 12, "diciembre": 12, "decembro": 12,
    }

    @classmethod
    def _extract_chunk_date_span(cls, doc: RetrievedDocument) -> Optional[tuple[date, date]]:
        """Resuelve el rango exacto de fechas de vigencia [inicio, fin] de una tabla o fragmento."""
        month_alts = "|".join(sorted(cls._SPANISH_MONTHS.keys(), key=len, reverse=True))
        month_range_re = re.compile(
            rf"(?i)\b(\d{{1,2}})\s*(?:de\s+)?({month_alts})\b(?:\s*(?:de\s+)?(202[2-9]|2[2-9]))?"
            rf"\s*(?:[-–—]|\b(?:al?|hasta(?:\s+el)?)\b)\s*"
            rf"(?:el\s+)?(\d{{1,2}})\s*(?:de\s+)?({month_alts})\b\s+(?:de\s+)?(202[2-9]|2[2-9])\b"
        )
        num_range_re = re.compile(
            r"(?i)\b(\d{1,2})[/.-](\d{1,2})[/.-](202[2-9]|2[2-9])\b"
            r"\s*(?:[-–—]|\b(?:al?|hasta(?:\s+el)?)\b)\s*"
            r"(?:el\s+)?(\d{1,2})[/.-](\d{1,2})[/.-](202[2-9]|2[2-9])\b"
        )

        def _norm_year(raw_y: str) -> int:
            y = int(raw_y)
            return y + 2000 if y < 100 else y

        def _collect_ranges(surface: str) -> list[tuple[date, date]]:
            found: list[tuple[date, date]] = []
            for m in month_range_re.finditer(surface):
                d1, m1_s, y1_s, d2, m2_s, y2_s = m.groups()
                m1 = cls._SPANISH_MONTHS.get(m1_s.lower(), 0)
                m2 = cls._SPANISH_MONTHS.get(m2_s.lower(), 0)
                y2 = _norm_year(y2_s)
                y1 = _norm_year(y1_s) if y1_s else (y2 - 1 if m1 > m2 else y2)
                try:
                    dt1, dt2 = date(y1, m1, int(d1)), date(y2, m2, int(d2))
                    if dt1 <= dt2:
                        found.append((dt1, dt2))
                except ValueError:
                    continue
            for m in num_range_re.finditer(surface):
                d1, m1_s, y1_s, d2, m2_s, y2_s = m.groups()
                try:
                    dt1 = date(_norm_year(y1_s), int(m1_s), int(d1))
                    dt2 = date(_norm_year(y2_s), int(m2_s), int(d2))
                    if dt1 <= dt2:
                        found.append((dt1, dt2))
                except ValueError:
                    continue
            return found

        header_surface = f"{doc.titulo_o_capitulo} | {doc.articulo}"
        body = doc.text.split("\n\n", 1)[-1] if "\n\n" in doc.text else doc.text

        ranges = _collect_ranges(header_surface) or _collect_ranges(body)
        if ranges:
            return min(r[0] for r in ranges), max(r[1] for r in ranges)

        pipe_header_years: list[int] = []
        for line in body.splitlines()[:8]:
            if "|" in line:
                for ym in re.finditer(r"(?<!\d)(2\.?02[2-9])(?!\d|,\d)", line):
                    pipe_header_years.append(int(ym.group(1).replace(".", "")))
        if len(set(pipe_header_years)) >= 2:
            return date(min(pipe_header_years), 1, 1), date(max(pipe_header_years), 12, 31)

        yr = cls._extract_chunk_year(doc)
        if yr > 0:
            return date(yr, 1, 1), date(yr, 12, 31)

        year_range = re.search(
            r"\b(202[2-9])\s*[-–—/]\s*(202[2-9])\b",
            f"{header_surface}\n{body[:350]}",
        )
        if year_range:
            y1, y2 = int(year_range.group(1)), int(year_range.group(2))
            if y1 <= y2:
                return date(y1, 1, 1), date(y2, 12, 31)
        return None

    @classmethod
    def _select_preferred_monetary_tables(
        cls,
        docs: list[RetrievedDocument],
        explicit_years: list[int],
        reference_date: Optional[date] = None,
    ) -> set[str]:
        """Selecciona los IDs de tablas monetarias preferentes según la fecha de vigencia o años consultados."""
        if not docs:
            return set()
        today = reference_date or date.today()
        raw_spans = [(d, cls._extract_chunk_date_span(d)) for d in docs]

        # Normaliza erratas tipográficas de boletín del tipo `01/01/YYYY - 31/01/YYYY` cuando
        # no existe ninguna tabla posterior dentro de ese mismo año natural YYYY.
        spans: list[tuple[RetrievedDocument, Optional[tuple[date, date]]]] = []
        for d, sp in raw_spans:
            if (
                sp is not None
                and sp[0] == date(sp[0].year, 1, 1)
                and sp[1] == date(sp[0].year, 1, 31)
                and not any(
                    other_sp is not None
                    and other_sp[0].year == sp[0].year
                    and other_sp[0] > sp[1]
                    for _, other_sp in raw_spans
                )
            ):
                spans.append((d, (sp[0], date(sp[0].year, 12, 31))))
            else:
                spans.append((d, sp))

        if not explicit_years:
            active_today = {
                d.chunk_id
                for d, sp in spans
                if sp is not None and sp[0] <= today <= sp[1]
            }
            if active_today:
                return active_today

            past_spans = [
                (d, sp)
                for d, sp in spans
                if sp is not None and sp[0].year <= today.year
            ]
            if past_spans:
                max_end = max(sp[1] for _, sp in past_spans)
                return {
                    d.chunk_id
                    for d, sp in past_spans
                    if sp[1].year == max_end.year
                }

            max_yr = max((cls._extract_chunk_year(d) for d in docs), default=0)
            return (
                {d.chunk_id for d in docs if cls._extract_chunk_year(d) == max_yr}
                if max_yr > 0
                else set()
            )

        req_years = set(explicit_years)
        matching = [
            (d, sp)
            for d, sp in spans
            if sp is not None
            and any(
                sp[0] <= date(y, 12, 31) and sp[1] >= date(y, 1, 1)
                for y in req_years
            )
        ]
        if not matching:
            return set()
        if today.year in req_years and len(req_years) == 1:
            active_in_req = {
                d.chunk_id
                for d, sp in matching
                if sp[0] <= today <= sp[1]
            }
            if active_in_req:
                return active_in_req
        return {d.chunk_id for d, _ in matching}

    @staticmethod
    def _extract_aleh_area_pattern(role_context: Optional[str]) -> Optional[re.Pattern]:
        """Extrae un patrón regex general para el Área Funcional oficial del VI ALEH (1ª a 6ª) si consta en role_context."""
        if not role_context:
            return None
        area_map = {
            "1": r"1\.?[ªa]?|primera",
            "primera": r"1\.?[ªa]?|primera",
            "2": r"2\.?[ªa]?|segunda",
            "segunda": r"2\.?[ªa]?|segunda",
            "3": r"3\.?[ªa]?|tercera",
            "tercera": r"3\.?[ªa]?|tercera",
            "4": r"4\.?[ªa]?|cuarta",
            "cuarta": r"4\.?[ªa]?|cuarta",
            "5": r"5\.?[ªa]?|quinta",
            "quinta": r"5\.?[ªa]?|quinta",
            "6": r"6\.?[ªa]?|sexta",
            "sexta": r"6\.?[ªa]?|sexta",
        }
        m = re.search(
            r"(?i)(?:[áa]rea\s+funcional\s+|(?:\b|^))([1-6])(?:\.?[ªa]|\b)|(?:\b)(primera|segunda|tercera|cuarta|quinta|sexta)(?:\b)",
            role_context,
        )
        if not m:
            return None
        key = (m.group(1) or m.group(2) or "").lower()
        if key not in area_map:
            return None
        return re.compile(rf"(?i)(?:[áa]rea\s+funcional|A\.?F\.?)\s+(?:{area_map[key]})\b")

    @staticmethod
    def _doc_matches_area(doc: RetrievedDocument, area_pattern: Optional[re.Pattern]) -> bool:
        """Comprueba si el encabezado o referencia propia del chunk pertenece al Área Funcional buscada."""
        if not area_pattern:
            return False
        body_head = doc.text.split("\n\n", 1)[-1][:250]
        surface = f"{doc.articulo} | {body_head}"
        return bool(area_pattern.search(surface))

    @staticmethod
    def _will_use_cohere(use_cohere_rerank: bool) -> bool:
        """Determina si Cohere Rerank está habilitado y dispone de clave configurada."""
        cohere_key = os.getenv("COHERE_API_KEY", "").strip()
        return bool(use_cohere_rerank and cohere_key and not cohere_key.startswith("..."))

    @staticmethod
    def _build_search_query(query: str, role_context: Optional[str]) -> str:
        """Combina la consulta técnica con el contexto del oficio/categoría si aún no figura en ella."""
        return (
            f"{query} ({role_context})"
            if role_context and role_context.lower() not in query.lower()
            else query
        )

    @staticmethod
    def _is_remuneration_query(query: str) -> bool:
        """Detecta si la consulta involucra retribución, salarios, tablas, pagas extras, pluses o liquidaciones."""
        return bool(
            re.search(
                r"(?i)\b(?:sueldos?|salarios?|salarial(?:es)?|tablas?|cobrar|cobro|n[óo]minas?|retribuci[óo]n(?:es)?|pagas?|horas?\s+extras?|extraordinarias?|nocturnidad|plus(?:es)?|finiquitos?|liquidaci[óo]n|indemnizaci[óo]n|antig[üu]edad|trienios?|manutenci[óo]n|dietas?)\b",
                query,
            )
        )

    @classmethod
    def _clean_intent_query(
        cls,
        query: str,
        rerank_query: Optional[str] = None,
        role_context: Optional[str] = None,
    ) -> str:
        """Devuelve la consulta limpia del turno (sin la etiqueta automática de role_context ni el keyword-stuffing)."""
        target = (rerank_query or query or "").strip()
        if role_context:
            target = target.replace(f"({role_context})", "").replace(role_context, "").strip()
        return target

    @classmethod
    def _needs_table_retrieval(
        cls,
        query: str,
        rerank_query: Optional[str] = None,
        role_context: Optional[str] = None,
    ) -> bool:
        """Determina de forma generalista si la consulta requiere tablas salariales o anexos de encuadramiento/niveles."""
        clean_target = cls._clean_intent_query(query, rerank_query, role_context)
        if cls._is_remuneration_query(clean_target):
            return True
        return bool(
            re.search(
                r"(?i)\b(?:tablas?|nivel(?:es)?\s+(?:salarial|retributivo)|encuadramiento|clasificaci[óo]n\s+en\s+grupos|[áa]reas?\s+funcional(?:es)?|categor[íi]as?\s+del?\s+establecimiento)\b",
                clean_target,
            )
        )

    _COOFFICIAL_STEM_EQUIVALENTS: dict[str, tuple[str, ...]] = {
        "cociner": ("cociner", "cociñeir", "cuiner"),
        "cociñeir": ("cociñeir", "cociner", "cuiner"),
        "cuiner": ("cuiner", "cociner", "cociñeir"),
        "cocin": ("cocin", "cociñ", "cuin"),
        "cociñ": ("cociñ", "cocin", "cuin"),
        "camarer": ("camarer", "camareir", "cambrer"),
        "camareir": ("camareir", "camarer", "cambrer"),
        "cambrer": ("cambrer", "camarer", "camareir"),
        "ayudant": ("ayudant", "axudant", "ajudant"),
        "axudant": ("axudant", "ayudant", "ajudant"),
        "ajudant": ("ajudant", "ayudant", "axudant"),
        "jef": ("jef", "xef"),
        "xef": ("xef", "jef"),
        "conserj": ("conserj", "conserx", "conserg"),
        "conserx": ("conserx", "conserj", "conserg"),
        "reposter": ("reposter", "reposteir"),
        "reposteir": ("reposteir", "reposter"),
        "limpiez": ("limpiez", "limpez", "netej"),
        "limpez": ("limpez", "limpiez", "netej"),
        "mantenimient": ("mantenimient", "mantement", "manteniment"),
        "mantement": ("mantement", "mantenimient", "manteniment"),
        "encargad": ("encargad", "encarregat"),
    }

    @classmethod
    def _stem_variants(cls, stem: str) -> tuple[str, ...]:
        """Devuelve variantes morfológicas equivalentes en lenguas cooficiales para una raíz léxica."""
        for pref, variants in cls._COOFFICIAL_STEM_EQUIVALENTS.items():
            if stem.startswith(pref):
                return variants
        return (stem,)

    @staticmethod
    def _build_role_stems(role_context: Optional[str]) -> list[str]:
        """Extrae raíces léxicas morfológicas (eliminando solo sufijo de género/número) del puesto detectado."""
        raw_role = role_context.split("|")[0].strip().lower() if role_context else ""
        cleaned_role = re.sub(r"\([^)]*\)", " ", raw_role).strip() or raw_role
        stems: list[str] = []
        for w in re.findall(r"[a-záéíóúüñ]{4,}", cleaned_role):
            stem = re.sub(r"[oa]s?$", "", w)
            stems.append(stem if len(stem) >= 4 else w)
        return stems

    @classmethod
    def _line_matches_all_stems(cls, surface: str, role_stems: list[str]) -> bool:
        """Comprueba si cada raíz léxica (o su equivalente cooficial) aparece en la cadena."""
        return all(
            any(var in surface for var in cls._stem_variants(stem))
            for stem in role_stems
        )

    @classmethod
    def _matches_role_stem(cls, doc: RetrievedDocument, role_stems: list[str]) -> bool:
        """Comprueba si todas las raíces léxicas del oficio coexisten en una misma línea del fragmento."""
        if not role_stems:
            return False
        return any(
            cls._line_matches_all_stems(line, role_stems)
            for line in doc.text.lower().splitlines()
        )

    _SUBORDINATE_ROLE_RE = re.compile(
        r"(?i)\b(?:ayudante(?:/a)?|axudante(?:/a)?|ajudant|auxiliar|aprendiz|segundo|segunda|2\.?[ºª]|subgobernante|subgob\.?/a|subdirector|sub)\b"
    )
    _SUBORDINATE_STEMS = ("ayudant", "axudant", "ajudant", "auxiliar", "aprendiz", "segund", "pinch")

    @classmethod
    def _matches_primary_role_stem(cls, doc: RetrievedDocument, role_stems: list[str]) -> bool:
        """Comprueba coincidencia del puesto principal evitando falsos positivos en filas subordinadas o de pisos."""
        if not role_stems:
            return False
        allow_subordinate = any(
            s.startswith(cls._SUBORDINATE_STEMS) for s in role_stems
        )
        asks_pisos = any(s.startswith("pis") for s in role_stems)
        for raw_line in doc.text.lower().splitlines():
            if not cls._line_matches_all_stems(raw_line, role_stems):
                continue
            cells = (
                [c.strip() for c in raw_line.strip("|").split("|")]
                if "|" in raw_line
                else [raw_line]
            )
            matching_cells = [
                c for c in cells if cls._line_matches_all_stems(c, role_stems)
            ]
            targets = matching_cells or [raw_line]
            for target in targets:
                sub_items = [
                    p.strip()
                    for p in re.split(r"[,;]", target)
                    if p.strip() and cls._line_matches_all_stems(p, role_stems)
                ] or [target]
                for item in sub_items:
                    if not allow_subordinate and cls._SUBORDINATE_ROLE_RE.search(item):
                        continue
                    if not asks_pisos and re.search(r"(?i)\bde\s+pisos?\b", item):
                        continue
                    return True
        return False

    _QUERY_SUBSECTOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "hospedaje",
            re.compile(
                r"(?i)\b(?:hotel(?:es)?|hostal(?:es)?|pensi[óo]n(?:es)?|parador(?:es)?|balneario|hospedaje|alojamiento)\b"
            ),
        ),
        (
            "apartamentos",
            re.compile(
                r"(?i)\b(?:apartamento(?:s)?|aparthotel|camping(?:s)?|vivienda(?:s)?\s+(?:vacacional(?:es)?|(?:de\s+(?:alquiler|uso)\s+)?tur[íi]stic[ao]s?)|vivendas?\s+de\s+aluguer\s+tur[íi]stico)\b"
            ),
        ),
        (
            "colectividades",
            re.compile(
                r"(?i)\b(?:catering|colectividades|comedor(?:es)?\s+(?:escolar(?:es)?|hospital|empresa)|club(?:es)?\s+privado|cafeter[íi]a\s+(?:de\s+(?:un\s+)?)?hospital|cafeter[íi]a\s+hospitalaria)\b"
            ),
        ),
        (
            "lujo_ocio_especial",
            re.compile(
                r"(?i)\b(?:casino(?:s)?|campo(?:s)?\s+de\s+golf|sal[óo]n(?:es)?\s+de\s+juegos?|bingo|sala(?:s)?\s+de\s+fiestas?|lujo|5\s+tenedores|barra\s+americana)\b"
            ),
        ),
        (
            "restauracion",
            re.compile(
                r"(?i)\b(?:restaurante(?:s)?|bar(?:es)?|cafeter[íi]a(?:s)?|taberna|tasca|mes[óo]n|chiringuito|pizzer[íi]a|hamburgueser[íi]a|helader[íi]a|churrer[íi]a|pub|discoteca|ocio\s+nocturno|restauraci[óo]n)\b"
            ),
        ),
    )

    _DOC_SUBSECTOR_INCLUDE: dict[str, re.Pattern[str]] = {
        "hospedaje": re.compile(
            r"\b(?:hote(?:is|l(?:es)?)|hosta(?:is|l(?:es)?)|pensi[óo]n(?:es|s)?|pousadas?|hospedaje|alojamiento|h\s*[1-5]\*)\b"
        ),
        "apartamentos": re.compile(
            r"\b(?:apartamento(?:s)?|aptos?|camping(?:s)?|vivienda\s+vacacional|hoteis\s+apartamentos)\b"
        ),
        "colectividades": re.compile(
            r"\b(?:colectividades|catering|club(?:es)?\s+privado)\b"
        ),
        "lujo_ocio_especial": re.compile(
            r"\b(?:campos?\s+(?:de\s+)?golf|casinos?|salas?\s+(?:de\s+)?fiestas?|salas?\s+de\s+festa|barras?\s+americanas?|bares?\s+lujo|lujo)\b"
        ),
        "restauracion": re.compile(
            r"\b(?:restauraci[óo]n(?:\s+y\s+ocio)?|restaurantes?|rtes\b|casas?\s+de\s+comidas|bares?\b|cafet(?:ar[íi]as?|er[íi]as?|\.)?|tabernas?|chiringuitos?|ocio\s+nocturno|clasificaci[óo]n\s+3d)\b"
        ),
    }

    _DOC_RESTAURACION_EXCLUDE_RE = re.compile(
        r"\b(?:campos?\s+(?:de\s+)?golf|casinos?|salas?\s+(?:de\s+)?fiestas?|salas?\s+de\s+festa|barras?\s+americanas?|bares?\s+lujo|hoteles?\s+y\s+pensiones|tabernas?\s+que\s+non\s+serven\s+comidas|apartamentos|colectividades|catering)\b"
    )

    _SPLIT_TERRITORIES_CACHE: Optional[tuple[tuple[int, int], set[str]]] = None

    @classmethod
    def get_split_subsector_territories(cls) -> set[str]:
        """Descubre dinámicamente territorios con convenios separados (alojamiento vs restauración)."""
        cache_path = BOUNDARIES_CACHE_PATH
        if not cache_path.exists():
            return set()
        try:
            st = cache_path.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return set()
        cached = cls._SPLIT_TERRITORIES_CACHE
        if cached is not None and cached[0] == sig:
            return set(cached[1])
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        subsectors_by_terr: dict[str, set[str]] = {}
        for info in data.values():
            if not isinstance(info, dict):
                continue
            raw_t = str(info.get("territorio", "")).strip()
            sub = str(info.get("subsector") or "general").lower().strip()
            if raw_t and raw_t != "estatal" and sub in ("alojamiento", "restauracion"):
                canon = _canonicalize_territory(raw_t)
                subsectors_by_terr.setdefault(canon, set()).add(sub)
        split_terrs = {t for t, subs in subsectors_by_terr.items() if len(subs) >= 2}
        cls._SPLIT_TERRITORIES_CACHE = (sig, split_terrs)
        return set(split_terrs)

    _AGREEMENT_ALOJAMIENTO_EST_RE = re.compile(
        r"(?i)\b(?:hotel(?:es)?|hostal(?:es)?|pensi[óo]n(?:es)?|parador(?:es)?|balneario|hospedaje|alojamiento(?:s)?|apartahotel(?:es)?|aparthotel(?:es)?|apartamentos?\s+tur[íi]sticos?|campings?|casa\s+rural|agroturismo|albergue(?:s)?)\b"
    )
    _AGREEMENT_RESTAURACION_EST_RE = re.compile(
        r"(?i)\b(?:restaurante(?:s)?|bar(?:es)?|cafeter[íi]a(?:s)?|taberna(?:s)?|tasca|sidrer[íi]a(?:s)?|asador(?:es)?|mes[óo]n(?:es)?|chiringuito(?:s)?|pizzer[íi]a(?:s)?|hamburgueser[íi]a(?:s)?|helader[íi]a(?:s)?|churrer[íi]a(?:s)?|pub(?:s)?|discoteca(?:s)?|salas?\s+de\s+fiestas?|catering|colectividades|comedor(?:es)?\s+escolares?|restauraci[óo]n|fast\s+food|comida\s+r[áa]pida)\b"
    )
    _AGREEMENT_ALOJAMIENTO_ROLE_RE = re.compile(
        r"(?i)\b(?:camarer[oa]s?\s+de\s+pisos?|gobernant[ae]s?|botones|conserje(?:s)?\s+de\s+noche|recepcionista(?:s)?)\b"
    )
    _AGREEMENT_RESTAURACION_ROLE_RE = re.compile(
        r"(?i)\b(?:riders?|repartidor(?:es|as)?(?:\s+de\s+comida)?)\b"
    )

    @classmethod
    def _extract_query_agreement_subsector(
        cls,
        clean_query: str,
        role_context: Optional[str] = None,
    ) -> Optional[str]:
        """Resuelve el convenio subsectorial aplicable cuando una provincia divide Alojamiento y Restauración."""
        surface = (clean_query or "").strip()
        has_aloj = bool(cls._AGREEMENT_ALOJAMIENTO_EST_RE.search(surface))
        has_rest = bool(cls._AGREEMENT_RESTAURACION_EST_RE.search(surface))
        if has_aloj and not has_rest:
            return "alojamiento"
        if has_rest and not has_aloj:
            return "restauracion"
        if has_aloj and has_rest:
            return None
        role_surface = f"{surface} {role_context or ''}".strip()
        has_aloj_role = bool(cls._AGREEMENT_ALOJAMIENTO_ROLE_RE.search(role_surface))
        has_rest_role = bool(cls._AGREEMENT_RESTAURACION_ROLE_RE.search(role_surface))
        if has_aloj_role and not has_rest_role:
            return "alojamiento"
        if has_rest_role and not has_aloj_role:
            return "restauracion"
        return None

    @classmethod
    def _extract_query_establishment_subsector(cls, clean_query: str) -> Optional[str]:
        """Detecta el subsector de establecimiento en la consulta limpia (sin role_context)."""
        for subsector, pat in cls._QUERY_SUBSECTOR_PATTERNS:
            if pat.search(clean_query):
                return subsector
        return None

    @classmethod
    def _doc_matches_establishment_subsector(
        cls,
        doc: RetrievedDocument,
        subsector: Optional[str],
    ) -> bool:
        """Comprueba si una tabla salarial corresponde al subsector del establecimiento consultado."""
        if not subsector or subsector not in cls._DOC_SUBSECTOR_INCLUDE:
            return False
        header_cols = ""
        for ln in doc.text.splitlines():
            s = ln.strip()
            if s.startswith("|") and s.endswith("|"):
                parts = [p.strip() for p in s.strip("|").split("|")]
                header_cols = " | ".join(parts[1:]) if len(parts) > 1 else s
                break
        surface = f"{doc.titulo_o_capitulo} | {header_cols}".lower()
        if not cls._DOC_SUBSECTOR_INCLUDE[subsector].search(surface):
            return False
        if subsector == "restauracion" and cls._DOC_RESTAURACION_EXCLUDE_RE.search(surface):
            return False
        return True

    @staticmethod
    def _is_monetary_table(doc: RetrievedDocument) -> bool:
        """Determina si un fragmento contiene importes salariales/económicos (con '€'/'euros' o >=4 importes decimales XX,XX)."""
        if "€" in doc.text or "euros" in doc.text.lower():
            return True
        text_no_hours = re.sub(
            r"\b\d{1,2},\d{2}\s*h(?:oras?|rs?)?(?=\b|\.)",
            "",
            doc.text,
            flags=re.IGNORECASE,
        )
        return len(re.findall(r"\b(?:\d{1,3}(?:\.\s?\d{3})+|\d+),\d{2}\b", text_no_hours)) >= 4

    @staticmethod
    def _is_auxiliary_concepts_table(doc: RetrievedDocument) -> bool:
        """Identifica sub-tablas secundarias de pluses/conceptos retributivos o horas extras (sin escala de salario base)."""
        if doc.tipo_contenido != "tabla_salarial":
            return False
        body_head = doc.text.split("\n\n", 1)[-1][:260]
        if re.search(
            r"(?i)\bsalario\s+base\b|\|\s*(?:total\s+mensual|total\s+anual|salarios?\s+mensuales?)\s*\|",
            f"{doc.articulo} {body_head}",
        ):
            return False
        surface = f"{doc.titulo_o_capitulo} {doc.articulo} {body_head}"
        return bool(
            re.search(
                r"(?i)\b(?:otros\s+conceptos|outros\s+conceptos|conceptos\s+econ[óo]micos\s+no\s+incluidos|complementos\s+salariales\s+y\s+extrasalariales|pluses\s+y\s+complementos|horas?\s+extras?\b|valor\s+del?\s+trienio|valor\s+d[íi]a\s+festivo|servi[cz]os?\s+extraordinarios?|d[íi]as?\s+24\s+y\s+31\s+de\s+diciembre|manutenci[óo]n|seguro\s+colectivo)\b",
                surface,
            )
        )

    @classmethod
    def _is_classification_level_table(
        cls,
        doc: RetrievedDocument,
        include_seniority_tables: bool = False,
    ) -> bool:
        """Identifica tablas o anexos de encuadramiento/niveles y categorías sin importes monetarios."""
        if doc.capa_normativa != "convenio_provincial":
            return False
        if cls._is_monetary_table(doc):
            return False
        surface = f"{doc.titulo_o_capitulo} {doc.articulo}"
        if not include_seniority_tables and ("antigüedad" in surface.lower() or "antiguedad" in surface.lower()):
            return False
        if doc.tipo_contenido == "tabla_salarial":
            return bool(
                re.search(
                    r"(?i)\b(?:anexo|nivel|categor[íi]a|clasificaci[óo]n|grupos?\s+profesionales|personal\s+de|establecimientos)\b",
                    surface,
                )
            )
        if doc.tipo_contenido == "articulo":
            return bool(
                re.search(
                    r"(?i)\b(?:categor[íi]as?|clasificaci[óo]n)\s+de\s+(?:los\s+)?establecimientos\b",
                    doc.articulo,
                )
            )
        if doc.tipo_contenido == "anexo_o_disposicion":
            body_head = doc.text.split("\n\n", 1)[-1][:350]
            return bool(
                re.search(
                    r"(?i)\b(?:puestos?\s+de\s+trabajo|ocupaciones\s+y\s+puestos\s+de\s+trabajo|grupos?\s+salarial(?:es)?|niveles?\s+(?:salarial(?:es)?|retributivos?)|adscripci[óo]n\s+de\s+las\s+categor[íi]as|tablas?\s+de\s+correspondencia|clasificaci[óo]n\s+(?:profesional|del\s+personal|de\s+(?:los\s+)?establecimientos|de\s+(?:las\s+)?empresas)|categor[íi]as?\s+o\s+niveles\s+del\s+personal|equivalencias?\s+de\s+clasificaci)\b",
                    f"{surface} {body_head}",
                )
                or (
                    "anexo" in doc.titulo_o_capitulo.lower()
                    and re.search(
                        r"(?i)\b(?:GRUPO\s+(?:[IVX]+|[A-C]\b)|NIVEL\s+(?:[IVX]+|[1-6]\.?[º°]?)|CATEGOR[ÍI]A\s+[A-D]\b|CLASIFICACI[ÓO]N\s+(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA))\b",
                        body_head,
                    )
                )
            )
        return False

    @staticmethod
    def _deduplicate_docs(items: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Deduplica fragmentos preservando orden de llegada y primera aparición."""
        out: list[RetrievedDocument] = []
        seen_ids: set[str] = set()
        for item in items:
            if item.chunk_id not in seen_ids:
                out.append(item)
                seen_ids.add(item.chunk_id)
        return out

    def _run_query_specs(
        self,
        query_specs: list[tuple[str, dict]],
    ) -> dict[str, list[RetrievedDocument]]:
        """Ejecuta múltiples búsquedas híbridas en paralelo cuando se usa Qdrant Cloud."""
        if self.is_cloud and len(query_specs) > 1:
            with ThreadPoolExecutor(max_workers=len(query_specs)) as pool:
                futures = {key: pool.submit(self.hybrid_search, **kw) for key, kw in query_specs}
                return {key: fut.result() for key, fut in futures.items()}
        return {key: self.hybrid_search(**kw) for key, kw in query_specs}

    def retrieve_state_layer_candidates(
        self,
        query: str,
        top_k_per_layer: int = RETRIEVAL_TOP_K_PER_LAYER,
        use_cohere_rerank: bool = True,
        role_context: Optional[str] = None,
        usage_out: Optional[list[dict[str, Any]]] = None,
    ) -> list[RetrievedDocument]:
        """Rama paralela B1: recupera candidatos de Estatuto de los Trabajadores y VI ALEH."""
        search_query = self._build_search_query(query, role_context)
        d_vec = self._get_query_dense_vec(
            search_query,
            usage_out=usage_out,
            step_name="embed_state_layers",
        )
        s_vec = self._get_query_sparse_vec(search_query)
        include_vectors = not self._will_use_cohere(use_cohere_rerank)

        query_specs: list[tuple[str, dict]] = [
            (
                "et",
                {
                    "query": search_query,
                    "capa_normativa": "estatuto_trabajadores",
                    "top_k": top_k_per_layer,
                    "dense_query_vec": d_vec,
                    "sparse_query_vec": s_vec,
                    "include_vectors": include_vectors,
                },
            ),
            (
                "aleh",
                {
                    "query": search_query,
                    "capa_normativa": "aleh_estatal",
                    "top_k": top_k_per_layer,
                    "dense_query_vec": d_vec,
                    "sparse_query_vec": s_vec,
                    "include_vectors": include_vectors,
                },
            ),
        ]
        layer_results = self._run_query_specs(query_specs)
        return self._deduplicate_docs(layer_results["et"] + layer_results["aleh"])

    _TOPIC_BM25_SPECS: tuple[tuple[str, str, str], ...] = (
        (
            "pagas_extra",
            r"(?i)pagas?\s+extra|gratificaci[óo][ns]",
            "pagas extraordinarias gratificaciones extraordinarias gratificacións extraordinarias",
        ),
        (
            "horas_extra",
            r"(?i)horas?\s+extra|divisor\s+anual",
            "horas extraordinarias horas extras valor hora ordinaria recargo divisor anual",
        ),
        (
            "nocturnidad",
            r"(?i)nocturn",
            "nocturnidad nocturnidade jornada nocturna xornada nocturna horas nocturnas recargo",
        ),
        (
            "jornada",
            r"(?i)(?:jornada|xornada)\s+(?:de\s+traba(?:jo|llo)|laboral|anual|m[áa]xima)|horas?\s+anua(?:les|is)",
            "jornada de trabajo jornada laboral xornada laboral xornada de traballo jornada máxima anual horas anuales 40 horas semanales",
        ),
        (
            "descanso_semanal",
            r"(?i)descanso\s+semanal",
            "descanso semanal días consecutivos fin de semana libranza",
        ),
        (
            "vacaciones",
            r"(?i)vacaci[óo]n(?:es|s)",
            "vacaciones vacacións anuales días naturales periodo disfrute desfrute",
        ),
        (
            "ipc",
            r"(?i)\bipc\b|revisi[óo]n\s+salarial|paritaria|paga\s+[úu]nica|plus\s+(?:de\s+)?transporte",
            "revisión salarial ipc comisión paritaria cláusula garantía paga única plus transporte",
        ),
    )

    def retrieve_territory_candidates(
        self,
        query: str,
        provincia: str,
        top_k_per_layer: int = RETRIEVAL_TOP_K_PER_LAYER,
        use_cohere_rerank: bool = True,
        role_context: Optional[str] = None,
        usage_out: Optional[list[dict[str, Any]]] = None,
        topic_source_query: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """Rama paralela B1 por territorio: recupera articulado, tablas estratificadas y anexos de un único convenio."""
        norm_p = provincia.lower().strip()
        search_query = self._build_search_query(query, role_context)
        d_vec = self._get_query_dense_vec(
            search_query,
            usage_out=usage_out,
            step_name=f"embed_territory_{norm_p}",
        )
        s_vec = self._get_query_sparse_vec(search_query)
        include_vectors = not self._will_use_cohere(use_cohere_rerank)

        explicit_years = [2000 + int(m.group(1)) for m in re.finditer(r"\b2\.?0(2[3-9])\b", query)]
        is_remuneration_query = self._is_remuneration_query(query)
        include_seniority_tables = bool(re.search(r"(?i)antig[üu]edad|trienio", query))
        role_stems = self._build_role_stems(role_context)

        clean_q = self._clean_intent_query(
            topic_source_query or query,
            role_context=role_context,
        )
        agreement_subsector: Optional[str] = (
            self._extract_query_agreement_subsector(clean_q, role_context=role_context)
            if norm_p in self.get_split_subsector_territories()
            else None
        )

        topic_boosts: list[str] = []
        if re.search(r"(?i)pagas?\s+extra|gratificaci[óo][ns]", query):
            topic_boosts.append("gratificaciones extraordinarias gratificacións extraordinarias pagas extraordinarias pagas extras")
        if re.search(r"(?i)horas?\s+extra|divisor\s+anual", query):
            topic_boosts.append("horas extraordinarias horas extras valor hora normal divisor anual")
        if re.search(r"(?i)nocturn", query):
            topic_boosts.append("jornada nocturna xornada nocturna horas nocturnas nocturnidad nocturnidade")
        if re.search(r"(?i)\bipc\b|revisi[óo]n\s+salarial|paritaria|paga\s+[úu]nica|plus\s+(?:de\s+)?transporte", query):
            topic_boosts.append("revisión salarial ipc comisión paritaria cláusula garantía paga única plus transporte")
        if re.search(r"(?i)jornada|xornada|vacaci[óo]n|descanso", query):
            topic_boosts.append("jornada anual xornada laboral vacaciones vacacións descanso semanal")
        if is_remuneration_query and not topic_boosts:
            topic_boosts.append("gratificaciones extraordinarias gratificacións extraordinarias pagas extras salario base")

        pay_sparse_vec = (
            self._get_query_sparse_vec(f"{search_query} {' '.join(topic_boosts)}")
            if topic_boosts
            else None
        )

        query_specs: list[tuple[str, dict]] = [
            (
                "prov_art",
                {
                    "query": search_query,
                    "capa_normativa": "convenio_provincial",
                    "provincia": norm_p,
                    "tipo_contenido": "articulo",
                    "subsector": agreement_subsector,
                    "top_k": max(top_k_per_layer + 4, 10),
                    "dense_query_vec": d_vec,
                    "sparse_query_vec": s_vec,
                    "include_vectors": include_vectors,
                },
            ),
        ]
        if pay_sparse_vec is not None:
            query_specs.append(
                (
                    "prov_pay_art",
                    {
                        "query": search_query,
                        "capa_normativa": "convenio_provincial",
                        "provincia": norm_p,
                        "tipo_contenido": "articulo",
                        "subsector": agreement_subsector,
                        "top_k": 6,
                        "dense_query_vec": d_vec,
                        "sparse_query_vec": pay_sparse_vec,
                        "include_vectors": include_vectors,
                    },
                )
            )
        query_specs.append(
            (
                "prov_tables",
                {
                    "query": search_query,
                    "capa_normativa": "convenio_provincial",
                    "provincia": norm_p,
                    "tipo_contenido": "tabla_salarial",
                    "subsector": agreement_subsector,
                    "top_k": max(top_k_per_layer * 6, 36),
                    "dense_query_vec": d_vec,
                    "sparse_query_vec": s_vec,
                    "include_vectors": include_vectors,
                },
            )
        )
        if is_remuneration_query:
            cls_sparse_vec = self._get_query_sparse_vec(
                f"anexo clasificación empresas establecimientos grupos categorías niveles retributivos {query}"
            )
            query_specs.append(
                (
                    "prov_cls_tables",
                    {
                        "query": search_query,
                        "capa_normativa": "convenio_provincial",
                        "provincia": norm_p,
                        "tipo_contenido": "tabla_salarial",
                        "subsector": agreement_subsector,
                        "top_k": 3,
                        "dense_query_vec": d_vec,
                        "sparse_query_vec": cls_sparse_vec,
                        "include_vectors": include_vectors,
                    },
                )
            )
        query_specs.append(
            (
                "prov_anexo",
                {
                    "query": query,
                    "capa_normativa": "convenio_provincial",
                    "provincia": norm_p,
                    "tipo_contenido": "anexo_o_disposicion",
                    "subsector": agreement_subsector,
                    "top_k": 4 if is_remuneration_query else 2,
                    "dense_query_vec": d_vec,
                    "sparse_query_vec": s_vec,
                    "include_vectors": include_vectors,
                },
            )
        )

        layer_results = self._run_query_specs(query_specs)

        topic_surface = f"{query} {topic_source_query or ''}"
        matched_bm25_topics = [
            bm25_terms
            for _, pat, bm25_terms in self._TOPIC_BM25_SPECS
            if re.search(pat, topic_surface)
        ][:4]
        cls_bm25_terms = (
            "categorías clasificación establecimientos hoteles restaurantes estrellas tenedores"
            if is_remuneration_query
            else None
        )
        topic_art_docs: list[RetrievedDocument] = []
        if (matched_bm25_topics or cls_bm25_terms) and getattr(self, "qdrant", None) is not None:
            raw_cls_docs: list[RetrievedDocument] = []
            worker_count = len(matched_bm25_topics) + (1 if cls_bm25_terms else 0)
            if self.is_cloud and worker_count > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    topic_futures = [
                        pool.submit(
                            self.sparse_topic_search,
                            t_terms,
                            capa_normativa="convenio_provincial",
                            provincia=norm_p,
                            tipo_contenido=None,
                            subsector=agreement_subsector,
                            top_k=3,
                        )
                        for t_terms in matched_bm25_topics
                    ]
                    cls_fut = (
                        pool.submit(
                            self.sparse_topic_search,
                            cls_bm25_terms,
                            capa_normativa="convenio_provincial",
                            provincia=norm_p,
                            tipo_contenido="articulo",
                            subsector=agreement_subsector,
                            top_k=2,
                        )
                        if cls_bm25_terms
                        else None
                    )
                    for fut in topic_futures:
                        topic_art_docs.extend(
                            d
                            for d in fut.result()
                            if d.tipo_contenido in ("articulo", "anexo_o_disposicion")
                        )
                    if cls_fut is not None:
                        raw_cls_docs.extend(cls_fut.result())
            else:
                for t_terms in matched_bm25_topics:
                    topic_art_docs.extend(
                        d
                        for d in self.sparse_topic_search(
                            t_terms,
                            capa_normativa="convenio_provincial",
                            provincia=norm_p,
                            tipo_contenido=None,
                            subsector=agreement_subsector,
                            top_k=3,
                        )
                        if d.tipo_contenido in ("articulo", "anexo_o_disposicion")
                    )
                if cls_bm25_terms:
                    raw_cls_docs.extend(
                        self.sparse_topic_search(
                            cls_bm25_terms,
                            capa_normativa="convenio_provincial",
                            provincia=norm_p,
                            tipo_contenido="articulo",
                            subsector=agreement_subsector,
                            top_k=2,
                        )
                    )
            for cls_doc in raw_cls_docs:
                if self._is_classification_level_table(cls_doc, include_seniority_tables):
                    topic_art_docs.append(cls_doc)

        raw_tables = self._deduplicate_docs(
            layer_results["prov_tables"] + layer_results.get("prov_cls_tables", [])
        )
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

        euro_tables = [t for t in filtered_tables if self._is_monetary_table(t)]
        level_tables = [t for t in filtered_tables if t not in euro_tables]
        preferred_euro_ids = self._select_preferred_monetary_tables(euro_tables, explicit_years)
        query_subsector = self._extract_query_establishment_subsector(clean_q)
        asks_hours_extra_q = bool(re.search(r"(?i)horas?\s+extra(?:ordinaria)?s?", clean_q))
        asks_aux_q = bool(
            re.search(
                r"(?i)\b(?:plus|manutenci[óo]n|herramientas|ropa|seguro|quebranto|alojamiento)\b",
                clean_q,
            )
        )
        if euro_tables:
            euro_tables.sort(
                key=lambda t: (
                    bool(preferred_euro_ids) and t.chunk_id not in preferred_euro_ids,
                    bool(query_subsector) and not self._doc_matches_establishment_subsector(t, query_subsector),
                    self._is_auxiliary_concepts_table(t) and not (asks_hours_extra_q or asks_aux_q),
                    bool(role_stems) and not self._matches_primary_role_stem(t, role_stems),
                    bool(role_stems) and not self._matches_role_stem(t, role_stems),
                )
            )

        top_level_candidates: list[RetrievedDocument] = []
        seen_level_sections: set[str] = set()
        classification_annexes = [
            a
            for a in layer_results["prov_anexo"]
            if self._is_classification_level_table(a, include_seniority_tables)
        ]
        cls_levels = sorted(
            [t for t in level_tables if self._is_classification_level_table(t, include_seniority_tables)]
            + classification_annexes,
            key=lambda t: (
                not self._matches_primary_role_stem(t, role_stems),
                not self._matches_role_stem(t, role_stems),
            ),
        )
        for item in cls_levels:
            sec_key = item.titulo_o_capitulo.strip()
            if sec_key not in seen_level_sections:
                top_level_candidates.append(item)
                seen_level_sections.add(sec_key)
                if len(top_level_candidates) >= 4:
                    break
        for item in cls_levels + level_tables:
            if len(top_level_candidates) >= 4:
                break
            if item not in top_level_candidates:
                top_level_candidates.append(item)

        balanced_tables: list[RetrievedDocument] = []
        for seq in (euro_tables[:8], top_level_candidates, euro_tables[8:]):
            for item in seq:
                if item not in balanced_tables:
                    balanced_tables.append(item)

        combined: list[RetrievedDocument] = []
        combined.extend(layer_results["prov_art"])
        if "prov_pay_art" in layer_results:
            combined.extend(layer_results["prov_pay_art"])
        combined.extend(topic_art_docs)
        combined.extend((balanced_tables or raw_tables)[: max(top_k_per_layer + 6, 12)])
        combined.extend(layer_results["prov_anexo"])

        deduped = self._deduplicate_docs(combined)
        for doc in deduped:
            doc.matched_territorio = norm_p
        return deduped

    def rerank_and_select_by_territory(
        self,
        candidates: list[RetrievedDocument],
        query: str,
        territorios: list[str],
        final_top_k: int = RETRIEVAL_FINAL_TOP_K,
        use_cohere_rerank: bool = True,
        rerank_query: Optional[str] = None,
        role_context: Optional[str] = None,
        usage_out: Optional[list[dict[str, Any]]] = None,
    ) -> list[RetrievedDocument]:
        """Ejecuta 1 sola llamada Cohere Rerank sobre candidatos deduplicados y aplica cuotas independientes por territorio."""
        if not candidates:
            return []

        chunk_territories: dict[str, set[str]] = {}
        unique_candidates: list[RetrievedDocument] = []
        seen_ids: set[str] = set()
        for doc in candidates:
            if doc.matched_territorio:
                chunk_territories.setdefault(doc.chunk_id, set()).add(doc.matched_territorio.lower().strip())
            if doc.chunk_id not in seen_ids:
                unique_candidates.append(doc)
                seen_ids.add(doc.chunk_id)

        effective_rerank_query = (rerank_query or query).strip()
        if self._will_use_cohere(use_cohere_rerank):
            t_rerank = time.perf_counter()
            try:
                co_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY", "").strip())
                rerank_resp = co_client.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=effective_rerank_query,
                    documents=[c.text for c in unique_candidates],
                    top_n=len(unique_candidates),
                )
                elapsed_rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)
                meta = getattr(rerank_resp, "meta", None)
                billed = getattr(meta, "billed_units", None)
                search_units = int(getattr(billed, "search_units", None) or 1)
                if usage_out is not None:
                    usage_out.append(
                        {
                            "step": "rerank_cohere",
                            "provider": "cohere_rerank",
                            "model": COHERE_RERANK_MODEL,
                            "status": "ok",
                            "api_calls": 1,
                            "search_units": search_units,
                            "docs_reranked": len(unique_candidates),
                            "latency_ms": elapsed_rerank_ms,
                        }
                    )
                reranked_all: list[RetrievedDocument] = []
                for item in rerank_resp.results:
                    doc = unique_candidates[item.index].model_copy()
                    doc.rerank_score = float(item.relevance_score)
                    reranked_all.append(doc)
                candidates = reranked_all
            except Exception as exc:
                elapsed_rerank_ms = round((time.perf_counter() - t_rerank) * 1000, 1)
                if usage_out is not None:
                    usage_out.append(
                        {
                            "step": "rerank_cohere",
                            "provider": "cohere_rerank",
                            "model": COHERE_RERANK_MODEL,
                            "status": "failed_other",
                            "api_calls": 1,
                            "search_units": 0,
                            "docs_reranked": len(unique_candidates),
                            "latency_ms": elapsed_rerank_ms,
                            "error_detail": str(exc)[:180],
                        }
                    )
                logger.warning(
                    "Aviso: fallo o clave ausente en Cohere Rerank (%s). Usando ordenación RRF híbrida.",
                    exc,
                )
                candidates = sorted(unique_candidates, key=lambda d: d.rerank_score or 0.0, reverse=True)
        else:
            candidates = sorted(unique_candidates, key=lambda d: d.rerank_score or 0.0, reverse=True)

        norm_territorios = list(dict.fromkeys(t.lower().strip() for t in territorios if t and t.strip()))
        if not norm_territorios:
            return candidates[:final_top_k]

        clean_intent_q = self._clean_intent_query(query, effective_rerank_query, role_context=role_context)
        explicit_years = [2000 + int(m.group(1)) for m in re.finditer(r"\b2\.?0(2[3-9])\b", clean_intent_q)]
        is_remuneration_query = self._is_remuneration_query(clean_intent_q)
        include_seniority_tables = bool(re.search(r"(?i)antig[üu]edad|trienio", clean_intent_q))
        needs_tables = (
            self._needs_table_retrieval(query, effective_rerank_query, role_context=role_context)
            or include_seniority_tables
        )

        if len(norm_territorios) == 1:
            total_budget = (
                final_top_k
                if needs_tables
                else min(final_top_k, QUOTA_PROVINCIAL_ARTICLES + 1 + QUOTA_ESTATUTO_CHUNKS)
            )
            if len(candidates) <= total_budget:
                return candidates[:total_budget]
        else:
            per_terr_table_quota = QUOTA_PROVINCIAL_TABLES if needs_tables else 0
            per_terr_quota = QUOTA_PROVINCIAL_ARTICLES + per_terr_table_quota
            total_budget = min(
                20,
                max(
                    final_top_k if needs_tables else (len(norm_territorios) * per_terr_quota + 3),
                    len(norm_territorios) * per_terr_quota + QUOTA_ALEH_CHUNKS + QUOTA_ESTATUTO_CHUNKS,
                ),
            )

        role_stems = self._build_role_stems(role_context)
        area_pattern = self._extract_aleh_area_pattern(role_context)
        target_subsector = self._extract_query_establishment_subsector(clean_intent_q)

        selected: list[RetrievedDocument] = []
        selected_ids: set[str] = set()
        stale_euro_ids: set[str] = set()

        def _pick(predicate, limit: int, pool: Optional[list[RetrievedDocument]] = None) -> None:
            count = 0
            source_docs = pool if pool is not None else candidates
            for doc in source_docs:
                if count >= limit or len(selected) >= total_budget:
                    break
                if doc.chunk_id not in selected_ids and predicate(doc):
                    selected.append(doc)
                    selected_ids.add(doc.chunk_id)
                    count += 1

        def _belongs_to_territory(doc: RetrievedDocument, terr_key: str) -> bool:
            if doc.capa_normativa != "convenio_provincial":
                return False
            tagged = chunk_territories.get(doc.chunk_id, set())
            if tagged:
                return terr_key in tagged
            doc_t = (doc.territorio or "").lower().strip()
            if doc_t == terr_key:
                return True
            return terr_key in {"barcelona", "girona", "tarragona", "lleida"} and doc_t == "catalunya"

        def _select_for_territory(terr_key: str) -> None:
            terr_cands = [d for d in candidates if _belongs_to_territory(d, terr_key)]
            if not terr_cands:
                return

            def _t_art_count() -> int:
                return sum(
                    1
                    for s in selected
                    if _belongs_to_territory(s, terr_key) and s.tipo_contenido == "articulo"
                )

            def _t_normative_count() -> int:
                return sum(
                    1
                    for s in selected
                    if _belongs_to_territory(s, terr_key)
                    and (
                        s.tipo_contenido == "articulo"
                        or (
                            s.tipo_contenido == "anexo_o_disposicion"
                            and not self._is_classification_level_table(s, include_seniority_tables)
                        )
                    )
                )

            def _t_tbl_count() -> int:
                return sum(
                    1
                    for s in selected
                    if _belongs_to_territory(s, terr_key)
                    and (
                        s.tipo_contenido == "tabla_salarial"
                        or self._is_classification_level_table(s, include_seniority_tables)
                    )
                )

            def _t_euro_tbl_count() -> int:
                return sum(
                    1
                    for s in selected
                    if _belongs_to_territory(s, terr_key)
                    and s.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(s)
                )

            prov_euro_tables = [
                d
                for d in terr_cands
                if d.tipo_contenido == "tabla_salarial" and self._is_monetary_table(d)
            ]
            preferred_euro_ids = self._select_preferred_monetary_tables(
                prov_euro_tables,
                explicit_years,
            )
            if preferred_euro_ids:
                stale_euro_ids.update(
                    d.chunk_id for d in prov_euro_tables if d.chunk_id not in preferred_euro_ids
                )
            distinct_cls_sections = {
                d.titulo_o_capitulo.strip()
                for d in terr_cands
                if self._is_classification_level_table(d, include_seniority_tables)
            }
            has_establishment_cls = any(
                self._is_classification_level_table(d, include_seniority_tables)
                and re.search(
                    r"(?i)\b(?:categor[íi]as?|clasificaci[óo]n)\s+de\s+(?:los\s+)?(?:establecimientos|empresas)\b|\b(?:estrellas|tenedores)\b",
                    f"{d.titulo_o_capitulo} {d.articulo} {d.text[:500]}",
                )
                for d in terr_cands
            )
            needs_dual_cls_tables = (
                is_remuneration_query
                and bool(prov_euro_tables)
                and len(distinct_cls_sections) >= 2
                and has_establishment_cls
            )
            asks_extra_pay = bool(
                re.search(
                    r"(?i)pagas?\s+extra(?:ordinaria)?s?|gratificaci[óo]n(?:es)?\s+extraordinaria",
                    clean_intent_q,
                )
            )
            extra_pay_table_row_re = re.compile(
                r"(?i)\|[^\n]*(?:gratificaci[óo]n(?:es)?\s+extraordinaria|paga(?:s)?\s+extraordinaria)[^\n]*\|"
            )
            has_separate_extra_pay_table = bool(
                asks_extra_pay
                and len(prov_euro_tables) >= 2
                and any(
                    extra_pay_table_row_re.search(d.text)
                    and (not preferred_euro_ids or d.chunk_id in preferred_euro_ids)
                    for d in prov_euro_tables
                )
            )
            asks_hours_extra = bool(
                re.search(r"(?i)horas?\s+extra(?:ordinaria)?s?", clean_intent_q)
            )
            hours_extra_tbl_re = re.compile(r"(?i)\bhoras?\s+extras?\b")
            has_separate_hours_extra_table = bool(
                is_remuneration_query
                and asks_hours_extra
                and len(prov_euro_tables) >= 2
                and any(
                    hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}")
                    and self._is_auxiliary_concepts_table(d)
                    and (not preferred_euro_ids or d.chunk_id in preferred_euro_ids)
                    for d in prov_euro_tables
                )
            )
            asks_aux_concepts = bool(
                re.search(
                    r"(?i)\b(?:plus\s+(?:cultural|transporte|altura|asistencia|convenio|vestuario|ropa)|manutenci[óo]n|herramientas|ropa\s+de\s+trabajo|seguro\s+de\s+vida|premio\s+(?:de|por)\s+(?:matrimonio|natalidad|nacimiento|jubilaci[óo]n)|quebranto\s+de\s+moneda|alojamiento)\b",
                    clean_intent_q,
                )
            )
            has_separate_concepts_table = bool(
                is_remuneration_query
                and asks_aux_concepts
                and len(prov_euro_tables) >= 2
                and any(
                    self._is_auxiliary_concepts_table(d)
                    and not hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}")
                    and (not preferred_euro_ids or d.chunk_id in preferred_euro_ids)
                    for d in prov_euro_tables
                )
            )
            effective_article_quota = max(
                1,
                QUOTA_PROVINCIAL_ARTICLES
                - (1 if needs_dual_cls_tables else 0)
                - (1 if (has_separate_extra_pay_table or has_separate_concepts_table) else 0),
            )

            def _is_normative_text(doc: RetrievedDocument) -> bool:
                return doc.tipo_contenido in ("articulo", "anexo_o_disposicion")
            if is_remuneration_query:
                extra_pay_art_pat = r"(?i)gratificaci[óo]n(?:es|s)?\s+extraordinarias?|pagas?\s+extra(?:ordinaria)?s?"
                specific_patterns = (
                    r"(?i)horas?\s+extra(?:ordinaria)?s?",
                    r"(?i)nocturnidad(?:e)?|(?:jornada|xornada)\s+nocturna|horas?\s+nocturnas|de\s+10\s+(?:da|de\s+la)\s+noite|12\s+(?:da|de\s+la)\s+noite",
                    r"(?i)vacaci[óo]n(?:es|s)",
                    r"(?i)(?:jornada|xornada)\s+(?:de\s+traba(?:jo|llo)|laboral|anual|m[áa]xima)|tiemp[oa]\s+parcial|tempo\s+parcial",
                    r"(?i)descanso\s+semanal",
                    r"(?i)comisi[óo]n\s+(?:mixta\s+|mista\s+)?paritaria|paga\s+[úu]nica|plus\s+(?:de\s+)?transporte",
                )
                matched_specific_pats = [
                    pat
                    for pat in specific_patterns
                    if re.search(pat, f"{query} {effective_rerank_query}")
                ]
                if asks_extra_pay or not matched_specific_pats:
                    before_extra = len(selected)
                    _pick(
                        lambda d: _is_normative_text(d)
                        and bool(re.search(extra_pay_art_pat, d.articulo)),
                        1,
                        terr_cands,
                    )
                    if len(selected) == before_extra:
                        _pick(
                            lambda d: _is_normative_text(d)
                            and bool(re.search(extra_pay_art_pat, d.text)),
                            1,
                            terr_cands,
                        )
                for pat in matched_specific_pats:
                    before_spec = len(selected)
                    _pick(
                        lambda d, p=pat: _is_normative_text(d)
                        and bool(re.search(p, d.articulo))
                        and not re.search(r"\(Parte\s+[2-9]\d*/", d.articulo),
                        1,
                        terr_cands,
                    )
                    if len(selected) == before_spec:
                        _pick(
                            lambda d, p=pat: _is_normative_text(d)
                            and bool(re.search(p, d.articulo)),
                            1,
                            terr_cands,
                        )
                    if len(selected) == before_spec:
                        _pick(
                            lambda d, p=pat: _is_normative_text(d)
                            and bool(re.search(p, f"{d.articulo} {d.text[:500]}")),
                            1,
                            terr_cands,
                        )
                    if len(selected) > before_spec:
                        picked_doc = selected[-1]
                        if (
                            picked_doc.tipo_contenido == "articulo"
                            and re.search(r"(?i)\banexo\b", picked_doc.text)
                        ):
                            annex_cands = [
                                (idx, d)
                                for idx, d in enumerate(terr_cands)
                                if d.chunk_id not in selected_ids
                                and d.tipo_contenido == "anexo_o_disposicion"
                                and not self._is_classification_level_table(d, include_seniority_tables)
                                and re.search(pat, f"{d.titulo_o_capitulo} {d.articulo} {d.text[:500]}")
                            ]
                            if annex_cands:
                                annex_cands.sort(
                                    key=lambda item: (-(self._extract_chunk_year(item[1]) or 0), item[0])
                                )
                                _pick(lambda _d: True, 1, [d for _, d in annex_cands])

            _pick(
                lambda d: d.tipo_contenido == "articulo"
                and not (needs_tables and self._is_classification_level_table(d, include_seniority_tables)),
                max(0, effective_article_quota - _t_normative_count()),
                terr_cands,
            )

            if not needs_tables:
                return

            def _pick_with_period_preference(base_pred, limit: int = 1) -> None:
                if limit <= 0:
                    return
                rem = limit
                if preferred_euro_ids:
                    before_pick = len(selected)
                    _pick(
                        lambda d: d.chunk_id in preferred_euro_ids and base_pred(d),
                        limit,
                        terr_cands,
                    )
                    rem -= len(selected) - before_pick
                if rem > 0:
                    _pick(base_pred, rem, terr_cands)

            if area_pattern and target_subsector:
                if role_stems:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_area(d, area_pattern)
                        and self._doc_matches_establishment_subsector(d, target_subsector)
                        and self._matches_primary_role_stem(d, role_stems),
                        1,
                    )
                    if _t_euro_tbl_count() == 0:
                        _pick_with_period_preference(
                            lambda d: d.tipo_contenido == "tabla_salarial"
                            and self._is_monetary_table(d)
                            and not self._is_auxiliary_concepts_table(d)
                            and bool(area_pattern.search(d.text))
                            and self._doc_matches_establishment_subsector(d, target_subsector)
                            and self._matches_primary_role_stem(d, role_stems),
                            1,
                        )
                if _t_euro_tbl_count() == 0:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_area(d, area_pattern)
                        and self._doc_matches_establishment_subsector(d, target_subsector),
                        1,
                    )

            if _t_euro_tbl_count() == 0 and target_subsector:
                if role_stems:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_establishment_subsector(d, target_subsector)
                        and self._matches_primary_role_stem(d, role_stems),
                        1,
                    )
                    if _t_euro_tbl_count() == 0:
                        _pick_with_period_preference(
                            lambda d: d.tipo_contenido == "tabla_salarial"
                            and self._is_monetary_table(d)
                            and not self._is_auxiliary_concepts_table(d)
                            and self._doc_matches_establishment_subsector(d, target_subsector)
                            and self._matches_role_stem(d, role_stems),
                            1,
                        )
                if _t_euro_tbl_count() == 0:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_establishment_subsector(d, target_subsector),
                        1,
                    )
                if _t_euro_tbl_count() == 0:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and self._doc_matches_establishment_subsector(d, target_subsector),
                        1,
                    )

            if _t_euro_tbl_count() == 0 and area_pattern:
                if role_stems:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_area(d, area_pattern)
                        and self._matches_primary_role_stem(d, role_stems),
                        1,
                    )
                    if _t_euro_tbl_count() == 0:
                        _pick_with_period_preference(
                            lambda d: d.tipo_contenido == "tabla_salarial"
                            and self._is_monetary_table(d)
                            and not self._is_auxiliary_concepts_table(d)
                            and self._doc_matches_area(d, area_pattern)
                            and self._matches_role_stem(d, role_stems),
                            1,
                        )
                if _t_euro_tbl_count() == 0:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and not self._is_auxiliary_concepts_table(d)
                        and self._doc_matches_area(d, area_pattern),
                        1,
                    )

            if _t_euro_tbl_count() == 0 and role_stems:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(d)
                    and not self._is_auxiliary_concepts_table(d)
                    and self._matches_role_stem(d, role_stems),
                    1,
                )

            if _t_euro_tbl_count() == 0:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(d)
                    and not self._is_auxiliary_concepts_table(d),
                    1,
                )

            if _t_euro_tbl_count() == 0:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(d),
                    1,
                )

            if _t_tbl_count() < QUOTA_PROVINCIAL_TABLES:
                _pick(
                    lambda d: self._is_classification_level_table(d, include_seniority_tables)
                    and self._matches_primary_role_stem(d, role_stems),
                    1,
                    terr_cands,
                )
                if not any(
                    _belongs_to_territory(s, terr_key)
                    and self._is_classification_level_table(s, include_seniority_tables)
                    for s in selected
                ):
                    _pick(
                        lambda d: self._is_classification_level_table(d, include_seniority_tables)
                        and self._matches_role_stem(d, role_stems),
                        1,
                        terr_cands,
                    )
                if not any(
                    _belongs_to_territory(s, terr_key)
                    and self._is_classification_level_table(s, include_seniority_tables)
                    for s in selected
                ):
                    _pick(
                        lambda d: self._is_classification_level_table(d, include_seniority_tables),
                        1,
                        terr_cands,
                    )
                if needs_dual_cls_tables:
                    selected_cls_sections = {
                        s.titulo_o_capitulo.strip()
                        for s in selected
                        if _belongs_to_territory(s, terr_key)
                        and self._is_classification_level_table(s, include_seniority_tables)
                    }
                    if selected_cls_sections:
                        _pick(
                            lambda d: self._is_classification_level_table(d, include_seniority_tables)
                            and d.titulo_o_capitulo.strip() not in selected_cls_sections
                            and self._matches_role_stem(d, role_stems),
                            1,
                            terr_cands,
                        )
                        cls_count = sum(
                            1
                            for s in selected
                            if _belongs_to_territory(s, terr_key)
                            and self._is_classification_level_table(s, include_seniority_tables)
                        )
                        if cls_count < 2:
                            _pick(
                                lambda d: self._is_classification_level_table(d, include_seniority_tables)
                                and d.titulo_o_capitulo.strip() not in selected_cls_sections,
                                1,
                                terr_cands,
                            )
                if not any(
                    _belongs_to_territory(s, terr_key)
                    and self._is_classification_level_table(s, include_seniority_tables)
                    for s in selected
                ):
                    _pick(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and not self._is_monetary_table(d)
                        and not re.search(r"(?i)pre[áa]mbulo|encabezado|[íi]ndice", d.articulo)
                        and bool(
                            re.search(
                                r"(?i)\b(?:anexo|nivel|categor|grupo|clasificaci|establec|puesto)\b",
                                f"{d.titulo_o_capitulo} {d.articulo} {d.text[:350]}",
                            )
                        ),
                        max(0, QUOTA_PROVINCIAL_TABLES - _t_tbl_count()),
                        terr_cands,
                    )

            if has_separate_extra_pay_table:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(d)
                    and bool(extra_pay_table_row_re.search(d.text)),
                    1,
                )

            if has_separate_concepts_table:
                before_aux = len(selected)
                if target_subsector and role_stems:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and self._is_auxiliary_concepts_table(d)
                        and not hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}")
                        and self._doc_matches_establishment_subsector(d, target_subsector)
                        and self._matches_role_stem(d, role_stems),
                        1,
                    )
                if len(selected) == before_aux:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and self._is_auxiliary_concepts_table(d)
                        and not hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}"),
                        1,
                    )

            if has_separate_hours_extra_table and _t_tbl_count() < QUOTA_PROVINCIAL_TABLES:
                before_he = len(selected)
                if target_subsector and role_stems:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}")
                        and self._doc_matches_establishment_subsector(d, target_subsector)
                        and self._matches_primary_role_stem(d, role_stems),
                        1,
                    )
                    if len(selected) == before_he:
                        _pick_with_period_preference(
                            lambda d: d.tipo_contenido == "tabla_salarial"
                            and self._is_monetary_table(d)
                            and hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}")
                            and self._doc_matches_establishment_subsector(d, target_subsector)
                            and self._matches_role_stem(d, role_stems),
                            1,
                        )
                if len(selected) == before_he:
                    _pick_with_period_preference(
                        lambda d: d.tipo_contenido == "tabla_salarial"
                        and self._is_monetary_table(d)
                        and hours_extra_tbl_re.search(f"{d.titulo_o_capitulo} {d.articulo}"),
                        1,
                    )

            if _t_tbl_count() < QUOTA_PROVINCIAL_TABLES and target_subsector and role_stems:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and self._is_monetary_table(d)
                    and not self._is_auxiliary_concepts_table(d)
                    and self._doc_matches_establishment_subsector(d, target_subsector)
                    and self._matches_primary_role_stem(d, role_stems),
                    max(0, QUOTA_PROVINCIAL_TABLES - _t_tbl_count()),
                )

            if _t_tbl_count() < QUOTA_PROVINCIAL_TABLES and target_subsector:
                _pick_with_period_preference(
                    lambda d: d.tipo_contenido == "tabla_salarial"
                    and not self._is_auxiliary_concepts_table(d)
                    and self._doc_matches_establishment_subsector(d, target_subsector),
                    max(0, QUOTA_PROVINCIAL_TABLES - _t_tbl_count()),
                )

            _pick_with_period_preference(
                lambda d: d.tipo_contenido == "tabla_salarial"
                and not self._is_auxiliary_concepts_table(d),
                max(0, QUOTA_PROVINCIAL_TABLES - _t_tbl_count()),
            )

            _pick_with_period_preference(
                lambda d: d.tipo_contenido == "tabla_salarial",
                max(0, QUOTA_PROVINCIAL_TABLES - _t_tbl_count()),
            )

            if target_subsector and any(
                _belongs_to_territory(s, terr_key)
                and s.tipo_contenido == "tabla_salarial"
                and self._is_monetary_table(s)
                and self._doc_matches_establishment_subsector(s, target_subsector)
                for s in selected
            ):
                other_subs = [s for s in self._DOC_SUBSECTOR_INCLUDE if s != target_subsector]
                stale_euro_ids.update(
                    d.chunk_id
                    for d in terr_cands
                    if d.tipo_contenido == "tabla_salarial"
                    and not self._doc_matches_establishment_subsector(d, target_subsector)
                    and any(self._doc_matches_establishment_subsector(d, os) for os in other_subs)
                )

        for terr_key in norm_territorios:
            _select_for_territory(terr_key)

        aleh_quota = 1 if (not needs_tables and len(norm_territorios) == 1) else QUOTA_ALEH_CHUNKS
        _pick(lambda d: d.capa_normativa == "aleh_estatal", aleh_quota)
        _pick(lambda d: d.capa_normativa == "estatuto_trabajadores", QUOTA_ESTATUTO_CHUNKS)
        _pick(
            lambda d: (needs_tables or d.tipo_contenido != "tabla_salarial")
            and d.chunk_id not in stale_euro_ids
            and (d.rerank_score is None or d.rerank_score >= 0.05),
            max(0, total_budget - len(selected)),
        )

        selected.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        return selected

    def parallel_hierarchical_retrieve(
        self,
        query: str,
        provincia: Optional[str] = None,
        top_k_per_layer: int = RETRIEVAL_TOP_K_PER_LAYER,
        final_top_k: int = RETRIEVAL_FINAL_TOP_K,
        use_cohere_rerank: bool = True,
        rerank_query: Optional[str] = None,
        role_context: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """Recupera en paralelo de las 3 capas normativas (ET + ALEH + Convenios por territorio) y reordena con Cohere."""
        territorios = (
            [t.strip(" _").lower() for t in re.split(r"[,/|]|\s+y\s+", provincia) if t.strip(" _")]
            if provincia
            else []
        )
        candidates = self.retrieve_state_layer_candidates(
            query=query,
            top_k_per_layer=top_k_per_layer,
            use_cohere_rerank=use_cohere_rerank,
            role_context=role_context,
        )
        for terr in territorios:
            candidates.extend(
                self.retrieve_territory_candidates(
                    query=query,
                    provincia=terr,
                    top_k_per_layer=top_k_per_layer,
                    use_cohere_rerank=use_cohere_rerank,
                    role_context=role_context,
                    topic_source_query=rerank_query,
                )
            )
        return self.rerank_and_select_by_territory(
            candidates=candidates,
            query=query,
            territorios=territorios,
            final_top_k=final_top_k,
            use_cohere_rerank=use_cohere_rerank,
            rerank_query=rerank_query,
            role_context=role_context,
        )


def build_and_index_corpus(
    recreate: bool = False,
    source_files: Optional[list[str]] = None,
) -> tuple[AiHorecaVectorStore, list[LegalChunk]]:
    """Fragmenta los documentos procesados y los indexa (completa o incrementalmente) en Qdrant."""
    chunks = chunk_all_processed_documents()
    store = AiHorecaVectorStore()
    store.index_chunks(chunks, recreate=recreate, replace_source_files=source_files)
    return store, chunks


def migrate_local_to_cloud(
    local_dir: Path = QDRANT_STORAGE_DIR,
    batch_size: int = 64,
    recreate: bool = False,
) -> int:
    """Copia vectores y payloads pre-calculados de Qdrant local hacia Qdrant Cloud sin gastar tokens de embedding."""
    qdrant_url, _, _ = get_qdrant_config()
    if not qdrant_url:
        raise ValueError(
            "QDRANT_URL no está configurado en el entorno. Configura QDRANT_URL y QDRANT_API_KEY antes de migrar."
        )

    local_store = AiHorecaVectorStore(storage_dir=local_dir, force_local=True)
    cloud_store = AiHorecaVectorStore(force_local=False)
    try:
        cloud_store.ensure_collection(recreate=recreate)

        total_migrated = 0
        next_offset = None
        while True:
            points, next_offset = local_store.qdrant.scroll(
                collection_name=local_store.collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            batch_structs = [
                models.PointStruct(
                    id=pt.id,
                    vector=pt.vector,
                    payload=pt.payload or {},
                )
                for pt in points
            ]
            cloud_store.qdrant.upsert(
                collection_name=cloud_store.collection_name,
                points=batch_structs,
            )
            total_migrated += len(batch_structs)
            logger.info("Migrados %d puntos a Qdrant Cloud...", total_migrated)
            if next_offset is None:
                break

        final_count = cloud_store.qdrant.count(collection_name=cloud_store.collection_name).count
        logger.info("Migración a Qdrant Cloud completada: %d puntos en '%s'.", final_count, cloud_store.collection_name)
        return final_count
    finally:
        local_store.close()
        cloud_store.close()


def _cli() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Gestor de indexación vectorial híbrida de AiHoreca (Local / Cloud).")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Elimina y recrea la colección completa desde cero.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        dest="source_files",
        help="Reindexa incrementalmente solo el/los archivo(s) Markdown indicado(s) (ej. --source-file convenio_baleares.md).",
    )
    parser.add_argument(
        "--sync-to-cloud",
        action="store_true",
        help="Migra los puntos ya calculados en local hacia Qdrant Cloud sin llamadas a Gemini Embeddings.",
    )
    args = parser.parse_args()

    if args.sync_to_cloud:
        count = migrate_local_to_cloud(recreate=args.recreate)
        print(f"Qdrant Cloud sincronizado: {count} puntos disponibles.")
    else:
        store, _chunks = build_and_index_corpus(recreate=args.recreate, source_files=args.source_files)
        try:
            total = store.qdrant.count(collection_name=store.collection_name).count
            target_label = "Qdrant Cloud" if store.is_cloud else "Qdrant Local"
            print(f"{target_label} actualizado ({store.collection_name}): {total} puntos totales.")
        finally:
            store.close()


if __name__ == "__main__":
    _cli()
