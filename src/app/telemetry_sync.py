"""Sincronizador Asíncrono de Telemetría hacia Almacenamiento Remoto en Hugging Face Hub (`src/app/telemetry_sync.py`).

Resuelve dos problemas de ingeniería en despliegues sobre contenedores efímeros (ej. Streamlit Community Cloud):
1. Restauración al Arranque (`restore_from_hf_if_needed`):
   - Cuando el contenedor despierta o se redespliega con una base SQLite local vacía, descarga automáticamente
     `aihoreca_sessions.db` desde el repositorio Dataset privado configurado en `HF_LOG_DATASET_REPO`, preservando todo el histórico acumulado.
2. Subida Asíncrona con Ventana de Amortiguación (`trigger_async_sync` con Debounce):
   - Agrupa actualizaciones cercanas (ej. varios turnos seguidos o un voto 👍 segundos después de la respuesta)
     en un único commit Git hacia Hugging Face Hub, evitando bloqueos HTTP `429 Too Many Requests` y sin añadir
     latencia perceptible al usuario en el chat.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import threading
import time
from typing import Optional

from src.config import (
    CONVERSATIONS_TXT_PATH,
    HF_SYNC_DEBOUNCE_SECONDS,
    LOGS_DIR,
    SESSIONS_DB_PATH,
    TURNS_CSV_PATH,
    get_hf_telemetry_config,
)
from src.graph.memory import SessionMemoryStore


class HuggingFaceTelemetrySync:
    """Gestor thread-safe de persistencia remota hacia un repositorio Dataset privado en Hugging Face."""

    def __init__(self, debounce_seconds: Optional[float] = None) -> None:
        self.debounce_seconds = (
            float(debounce_seconds)
            if debounce_seconds is not None
            else HF_SYNC_DEBOUNCE_SECONDS
        )
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._first_pending_ts: Optional[float] = None
        self._upload_in_progress = False
        self._pending_while_uploading = False
        self.last_sync_status: str = "IDLE"
        self.last_sync_at: Optional[str] = None

    def restore_from_hf_if_needed(self, memory_store: SessionMemoryStore) -> bool:
        """Si la base SQLite local no tiene turnos y existe backup en HF Dataset, lo restaura al arrancar."""
        token, repo_id = get_hf_telemetry_config()

        # Verificar cuántos turnos tiene la base SQLite local actual (cerrando explícitamente la conexión)
        local_turn_count = 0
        try:
            with closing(sqlite3.connect(str(memory_store.db_path), timeout=10.0)) as conn:
                row = conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()
                local_turn_count = int(row[0]) if row else 0
        except Exception:
            local_turn_count = 0

        restored = False
        if token and repo_id and local_turn_count == 0:
            try:
                from huggingface_hub import hf_hub_download

                downloaded_db = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename="aihoreca_sessions.db",
                    token=token,
                )
                if downloaded_db and Path(downloaded_db).exists():
                    memory_store.db_path.parent.mkdir(parents=True, exist_ok=True)
                    # Limpiar posibles archivos WAL/SHM previos antes de restaurar el archivo SQLite
                    for suffix in ("-wal", "-shm"):
                        wal_file = Path(f"{memory_store.db_path}{suffix}")
                        if wal_file.exists():
                            try:
                                wal_file.unlink()
                            except Exception:
                                pass
                    # CRÍTICO: Usar copyfile (no copy2) porque los blobs de la caché de Hugging Face Hub
                    # tienen permisos 0444 (solo lectura) en Linux, y copy2 copiaba ese bit readonly a SQLite.
                    shutil.copyfile(downloaded_db, memory_store.db_path)
                    os.chmod(
                        memory_store.db_path,
                        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH,
                    )
                    memory_store._init_schema()
                    restored = True
                    self.last_sync_status = f"RESTORED_FROM_HF ({repo_id})"
            except Exception as exc:
                self.last_sync_status = f"HF_RESTORE_SKIPPED ({type(exc).__name__})"

        # Asegurar que `conversaciones.txt` y `turnos_metricas.csv` existen y reflejan la BD actual
        try:
            memory_store.export_flat_logs()
        except Exception:
            pass

        return restored

    def trigger_async_sync(self) -> None:
        """Encola una subida asíncrona con debounce hacia el Dataset privado de Hugging Face."""
        token, repo_id = get_hf_telemetry_config()
        if not token or not repo_id:
            self.last_sync_status = "LOCAL_ONLY (HF_LOG_TOKEN no configurado)"
            return

        with self._lock:
            now = time.monotonic()
            if self._first_pending_ts is None:
                self._first_pending_ts = now

            if self._upload_in_progress:
                self._pending_while_uploading = True
                return

            if self._timer is not None:
                self._timer.cancel()

            # Si lleva más de 45s acumulando cambios, disparamos en 1s; si no, aplicamos debounce normal
            elapsed_waiting = now - self._first_pending_ts
            delay = 1.0 if elapsed_waiting >= 45.0 else self.debounce_seconds

            self._timer = threading.Timer(delay, self._run_background_upload)
            self._timer.daemon = True
            self._timer.start()
            self.last_sync_status = f"QUEUED (en {delay:.0f}s -> {repo_id})"

    def flush_sync_now(self) -> bool:
        """Fuerza una subida inmediata bloqueante (útil para tests o cierre controlado)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        return self._run_background_upload()

    def _run_background_upload(self) -> bool:
        token, repo_id = get_hf_telemetry_config()
        if not token or not repo_id:
            return False

        with self._lock:
            if self._upload_in_progress:
                self._pending_while_uploading = True
                return False
            self._upload_in_progress = True
            self._first_pending_ts = None
            self._timer = None

        success = False
        try:
            from huggingface_hub import HfApi

            staging_dir = LOGS_DIR / ".hf_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)

            # 1. Copia en caliente transaccionalmente segura de SQLite (`sqlite3.backup`)
            if SESSIONS_DB_PATH.exists():
                staged_db = staging_dir / "aihoreca_sessions.db"
                src_conn = sqlite3.connect(str(SESSIONS_DB_PATH), timeout=15.0)
                dst_conn = sqlite3.connect(str(staged_db), timeout=15.0)
                try:
                    with dst_conn:
                        src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
                    src_conn.close()

            # 2. Copiar `conversaciones.txt` y `turnos_metricas.csv`
            if CONVERSATIONS_TXT_PATH.exists():
                shutil.copy2(CONVERSATIONS_TXT_PATH, staging_dir / "conversaciones.txt")
            if TURNS_CSV_PATH.exists():
                shutil.copy2(TURNS_CSV_PATH, staging_dir / "turnos_metricas.csv")

            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            api = HfApi(token=token)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(staging_dir),
                commit_message=f"Auto-sync telemetria AiHoreca ({now_utc})",
            )
            self.last_sync_at = now_utc
            self.last_sync_status = f"SYNCED ({now_utc} -> {repo_id})"
            success = True
        except Exception as exc:
            self.last_sync_status = f"SYNC_ERROR ({type(exc).__name__}: {exc})"
        finally:
            retrigger = False
            with self._lock:
                self._upload_in_progress = False
                if self._pending_while_uploading:
                    self._pending_while_uploading = False
                    retrigger = True
            if retrigger:
                self.trigger_async_sync()

        return success
