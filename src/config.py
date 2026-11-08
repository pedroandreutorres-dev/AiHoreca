"""Configuración centralizada y fábrica de clientes para AiHoreca."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "data" / "qdrant_storage"
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
SESSIONS_DB_PATH = SESSIONS_DIR / "aihoreca_sessions.db"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
CONVERSATIONS_TXT_PATH = LOGS_DIR / "conversaciones.txt"
TURNS_CSV_PATH = LOGS_DIR / "turnos_metricas.csv"
BOUNDARIES_CACHE_PATH = PROCESSED_DATA_DIR / "boundaries_cache.json"
EVALUATION_RESULTS_PATH = PROCESSED_DATA_DIR / "evaluation_results.json"

# Parámetros de Sincronización Remota de Telemetría (Opcional via variables de entorno)
HF_SYNC_DEBOUNCE_SECONDS = float(os.getenv("HF_SYNC_DEBOUNCE_SECONDS", "15.0"))

GEMINI_LITE_MODEL = "gemini-3.1-flash-lite"
GEMINI_FAST_MODEL = "gemini-3.5-flash"
GEMINI_REASONING_MODEL = "gemini-3.5-flash"
GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)
EMBEDDING_MODEL = "gemini-embedding-001"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"

# Presupuesto de Recuperación Jerárquica y Cuotas por Capa (Parametrizable por entorno Free Tier vs Paid Tier via .env)
RETRIEVAL_TOP_K_PER_LAYER = int(os.getenv("RETRIEVAL_TOP_K_PER_LAYER", "6"))
RETRIEVAL_FINAL_TOP_K = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "10"))
QUOTA_PROVINCIAL_ARTICLES = int(os.getenv("QUOTA_PROVINCIAL_ARTICLES", "4"))
QUOTA_PROVINCIAL_TABLES = int(os.getenv("QUOTA_PROVINCIAL_TABLES", "2"))
QUOTA_ALEH_CHUNKS = int(os.getenv("QUOTA_ALEH_CHUNKS", "2"))
QUOTA_ESTATUTO_CHUNKS = int(os.getenv("QUOTA_ESTATUTO_CHUNKS", "2"))


def load_environment() -> None:
    """Carga variables de entorno desde .env local o desde st.secrets en Streamlit Community Cloud."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path.home() / ".env")
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            for k in (
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "COHERE_API_KEY",
                "HF_LOG_TOKEN",
                "HF_TOKEN",
                "HF_LOG_DATASET_REPO",
            ):
                if k in st.secrets and not os.getenv(k):
                    os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


def get_hf_telemetry_config() -> tuple[Optional[str], Optional[str]]:
    """Devuelve (token, dataset_repo) para sincronización remota de logs si ambos están configurados en el entorno."""
    load_environment()
    token = (os.getenv("HF_LOG_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    repo = (os.getenv("HF_LOG_DATASET_REPO") or "").strip()
    if (
        not token
        or token == "hf_..."
        or not repo
        or "tu-usuario" in repo
    ):
        return None, None
    return token, repo


def get_google_api_key(api_key: Optional[str] = None) -> str:
    """Obtiene y valida la clave GOOGLE_API_KEY desde el entorno."""
    load_environment()
    resolved = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not resolved or "AIza..." in resolved:
        raise ValueError(
            "No se encontró una GOOGLE_API_KEY válida en el archivo .env."
        )
    return resolved


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Instancia el cliente oficial de Google GenAI SDK con timeout de 60s para lotes multimodales."""
    return genai.Client(
        api_key=get_google_api_key(api_key),
        http_options=types.HttpOptions(timeout=60000),
    )
