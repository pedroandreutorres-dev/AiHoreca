# Plan de Publicación Web (Streamlit Cloud) y Telemetría Relacional en Hugging Face (`AiHoreca`)

Este documento especifica la arquitectura de despliegue a coste 0 € como **Demo Web en Streamlit Community Cloud (conectada a GitHub)** y el sistema de **persistencia y telemetría relacional de conversaciones** sincronizado automáticamente con el **Repositorio de Dataset Privado en Hugging Face Hub (`tu-usuario/tu-dataset-logs`)**.

---

## 1. Objetivos y Restricciones de Ingeniería

1. **Aislamiento de Sesiones Multi-Usuario en la Web (`Session Privacy`):**
   - Cada visitante en el navegador (`st.session_state["visitor_session_ids"]`) únicamente puede ver y alternar entre las sesiones creadas durante su propia visita. Ningún usuario puede visualizar conversaciones de otros visitantes en la barra lateral.
2. **Sin Panel de Administración Expuesto en el Frontend:**
   - La interfaz web no expone formularios de administración ni botones de descarga de bases de datos. Toda la recolección y persistencia de logs ocurre en el backend de forma transparente.
3. **Persistencia Inmediata sin Dependencia del Voto del Usuario (`Two-Phase Logging`):**
   - **Fase 1 (`T = 0 ms`, obligatoria):** En cuanto el grafo termina de generar la respuesta y **antes** de renderizarla en pantalla, el turno queda persistido en SQLite (`user_feedback = 'SIN_VOTO'`) y volcado a los archivos de log. Si el usuario cierra la pestaña inmediatamente después de leer la respuesta (comportamiento del >90% de los usuarios), la conversación y sus métricas ya están guardadas al 100%.
   - **Fase 2 (`T > 0`, opcional):** Si el usuario pulsa `👍` (`OK`) o `👎` (`KO`) debajo de una respuesta, se ejecuta un `UPDATE` transaccional sobre `(session_id, turn_index)` y se actualizan los artefactos de telemetría.
4. **Separación Relacional entre Lectura Cualitativa (`.txt`) y Explotación Analítica (`.csv` / `.db`):**
   - Un archivo de texto plano limpio (`conversaciones.txt`) agrupado por `session_id` para lectura humana de principio a fin, vinculado mediante la clave compuesta `(session_id, turn_index)` con la tabla estructurada (`turnos_metricas.csv` y tabla `session_turns` en `aihoreca_sessions.db`).
5. **Cero Reinicios en Streamlit Cloud y Control de Rate-Limits en Hugging Face Hub:**
   - Como el código vive en GitHub (desplegado en Streamlit Community Cloud) pero los logs se envían por API al Dataset privado de Hugging Face (`tu-usuario/tu-dataset-logs`), **el repositorio de GitHub no sufre commits en caliente y Streamlit Cloud jamás reinicia la aplicación mientras los usuarios conversan**.
   - Además, la sincronización hacia Hugging Face utiliza una ventana de amortiguación (*debounce* de 15 s) para evitar bloqueos `429 Too Many Requests` y restaura automáticamente `aihoreca_sessions.db` al arrancar el contenedor (`restore_on_startup`).

---

## 2. Arquitectura Híbrida GitHub (Streamlit Cloud) + Hugging Face Dataset Privado (Coste 0€)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. STREAMING WEB APP: STREAMLIT COMMUNITY CLOUD (Conectado a GitHub)         │
│    Huella en RAM: ~180–200 MB (holgado bajo el techo gratuito de 1 GB RAM)   │
│    Incluye `data/qdrant_storage/` (~20,5 MB) pre-calculado en el repo Git.   │
│                                                                              │
│  [1] Al arrancar el contenedor (`restore_on_startup`):                       │
│      • Descarga `aihoreca_sessions.db` desde `tu-usuario/tu-dataset-logs`      │
│        si existe, evitando perder el histórico cuando el contenedor despierta│
│      • Regenera `data/logs/conversaciones.txt` y `turnos_metricas.csv`.      │
│                                                                              │
│  [2] En cada pregunta del usuario (`process_query` — Fase 1):                │
│      • Inserta turno en SQLite con `user_feedback = 'SIN_VOTO'`.             │
│      • Renderiza `conversaciones.txt` (agrupado por `session_id`)            │
│        y `turnos_metricas.csv`.                                              │
│      • Notifica al hilo `TelemetrySyncWorker` (no bloquea la UI).            │
│                                                                              │
│  [3] Si el usuario vota 👍 / 👎 (`record_feedback` — Fase 2):                │
│      • Actualiza `user_feedback = 'OK' | 'KO'` en SQLite para ese            │
│        `(session_id, turn_index)` y regenera `turnos_metricas.csv`.          │
│      • Notifica al hilo `TelemetrySyncWorker`.                               │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       │ Subida asíncrona con Debounce (15s)
                                       │ vía `huggingface_hub.HfApi.upload_folder`
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. ALMACÉN PERSISTENTE DE LOGS: HUGGING FACE DATASET (100% Privado y Gratis) │
│    `https://huggingface.co/datasets/tu-usuario/tu-dataset-logs`                │
│                                                                              │
│  ├── conversaciones.txt      (Diálogos íntegros en texto plano por sesión)   │
│  ├── turnos_metricas.csv     (Tabla relacional con session_id + turn_index)  │
│  └── aihoreca_sessions.db    (Base SQLite completa con chunks y AuditReport) │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Especificación de los Archivos de Log y Esquema Relacional

### 3.1. `data/logs/conversaciones.txt` (Inspección Cualitativa en Texto Plano)

- **Por qué no se escribe mediante `append` lineal simple:** Si dos visitantes conversan simultáneamente en la web, un `append` secuencial mezclaría los turnos de ambos usuarios. Por ello, `conversaciones.txt` se genera mediante consulta ordenada (`ORDER BY s.created_at ASC, t.turn_index ASC`) agrupando todos los turnos de cada `session_id` bajo un único bloque continuo.
- **Formato exacto:**

```text
================================================================================
SESSION_ID: ses-9f4a2c1b | Inicio: 2026-11-08 12:55:02 UTC | Territorio Activo: leon
================================================================================

[Turno 1 | 2026-11-08 12:55:02 UTC] USUARIO:
Mi mujer trabaja de camarera en un bar de Ponferrada. Contrato de 30 horas. ¿Cuál es su sueldo base? ¿Y a cuánto le deben pagar las horas extras?

[Turno 1 | 2026-11-08 12:55:16 UTC] AIHORECA:
Para una camarera en Ponferrada, clasificada en el Grupo Profesional 2.º (Área Funcional 3ª) según el VI ALEH...
[...Respuesta íntegra tal cual se mostró en pantalla...]

--------------------------------------------------------------------------------

[Turno 2 | 2026-11-08 12:56:18 UTC] USUARIO:
¿Y cómo sé a qué clase corresponde?

[Turno 2 | 2026-11-08 12:56:27 UTC] AIHORECA:
Para saber si el establecimiento en Ponferrada está clasificado en 1.ª, 2.ª o 3.ª categoría...
[...Respuesta íntegra del Turno 2...]

================================================================================
```

### 3.2. `data/logs/turnos_metricas.csv` (Tabla Analítica Relacional)

Cada fila representa un turno único identificado por `(session_id, turn_index)` para poder filtrar en Excel o Pandas y cruzar inmediatamente con `conversaciones.txt`:

| Columna | Tipo | Descripción |
| :--- | :--- | :--- |
| `session_id` | `TEXT` | Identificador único de la conversación (clave de cruce con `conversaciones.txt`). |
| `turn_index` | `INTEGER` | Número de turno dentro de la conversación (`1, 2, 3...`). |
| `created_at` | `TEXT` | Timestamp ISO-8601 UTC de generación de la respuesta. |
| `intent` | `TEXT` | `HOSPITALITY_CORE`, `UNSUPPORTED_PROVINCE` o `OUT_OF_SCOPE`. |
| `route_decision` | `TEXT` | `parallel_retrieve`, `ask_clarification` o `block_out_of_scope`. |
| `territorio` | `TEXT` | Territorio normalizado aplicado (`madrid`, `barcelona`, `leon`, etc.). |
| `oficio_detectado` | `TEXT` | Oficio coloquial detectado o heredado en la sesión. |
| `categoria_aleh` | `TEXT` | Encuadramiento oficial en el VI ALEH (Grupo Profesional y Área Funcional). |
| `audit_passed` | `INTEGER` | `1` si superó la auditoría numérica del Nodo 5 (`0` en caso de advertencia). |
| `verified_numbers_count` | `INTEGER` | Cantidad de cifras verificadas en el turno. |
| `unverified_numbers_count` | `INTEGER` | Cantidad de cifras no verificadas (`0` esperado). |
| `total_latency_ms` | `REAL` | Latencia extremo a extremo del turno en milisegundos. |
| `model_router` | `TEXT` | Modelo empleado en el Nodo 1. |
| `model_synthesis` | `TEXT` | Modelo empleado en el Nodo 4. |
| `user_feedback` | `TEXT` | `'SIN_VOTO'` (por defecto al crearse el turno), `'OK'` (`👍`) o `'KO'` (`👎`). |
| `feedback_at` | `TEXT` | Timestamp ISO-8601 UTC del momento en que el usuario votó (o vacío si no votó). |
| `user_query` | `TEXT` | Pregunta original del usuario en ese turno. |
| `effective_query` | `TEXT` | Consulta autocontenida resuelta por el Router. |

### 3.3. `data/sessions/aihoreca_sessions.db` (Base de Datos SQLite Completa)

Contiene las tablas `sessions` y `session_turns` completas, incluyendo las columnas JSON pesadas (`retrieved_docs_json` con los 10 fragmentos y sus `rerank_score`, `audit_report_json` y `node_timings_json`) para diagnóstico profundo en Python.

---

## 4. Plan de Implementación en el Código (Módulos a Modificar/Crear)

1. **[`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py):**
   - Definir `LOGS_DIR = DATA_DIR / "logs"` (`conversaciones.txt` y `turnos_metricas.csv`).
   - Leer variables de entorno opcionales para sincronización con Hugging Face Hub:
     - `HF_LOG_TOKEN`: Token de Hugging Face con permisos `write` sobre el Dataset privado.
     - `HF_LOG_DATASET_REPO`: ID del Dataset privado (ej. `tu-usuario/aihoreca-logs`).
     - `HF_SYNC_DEBOUNCE_SECONDS`: `15` segundos por defecto.

2. **[`src/graph/memory.py`](file:///d:/Proyectos/AiHoreca/src/graph/memory.py) (`SessionMemoryStore`):**
   - Migración automática retrocompatible en `_init_schema()` para asegurar que `session_turns` dispone de las columnas `user_feedback TEXT DEFAULT 'SIN_VOTO'`, `feedback_at TEXT` y `models_used_json TEXT`.
   - En `save_turn_and_update_slots()`: guardar el turno inmediatamente con `user_feedback = 'SIN_VOTO'` y llamar a `export_flat_logs()`.
   - Nuevo método `update_turn_feedback(session_id: str, turn_index: int, feedback: str) -> bool`: actualiza `user_feedback` (`'OK'` o `'KO'`) y `feedback_at` en SQLite y regenera `export_flat_logs()`.
   - Nuevo método `export_flat_logs()`: genera de forma atómica (escribiendo en archivo temporal `.tmp` y reemplazando con `os.replace`) `data/logs/conversaciones.txt` y `data/logs/turnos_metricas.csv`.

3. **`src/app/telemetry_sync.py` *(Nuevo módulo)* (`HuggingFaceTelemetrySync`):**
   - **`restore_from_hf_if_needed()`**: Se ejecuta una sola vez al inicializar `AiHorecaEngineService`. Si `HF_LOG_TOKEN` y `HF_LOG_DATASET_REPO` están configurados y la base local está vacía o recién creada en el contenedor, descarga `aihoreca_sessions.db` desde el Dataset privado de Hugging Face y regenera `conversaciones.txt` y `turnos_metricas.csv`.
   - **`trigger_async_sync()`**: Hilo `daemon` con ventana de amortiguación (*debounce* de 15s) que sube `conversaciones.txt`, `turnos_metricas.csv` y una copia consistente de `aihoreca_sessions.db` (obtenida con `sqlite3.Connection.backup()`) al Dataset privado mediante `HfApi.upload_folder()`.

4. **[`src/app/service.py`](file:///d:/Proyectos/AiHoreca/src/app/service.py) (`AiHorecaEngineService`):**
   - Integrar `restore_from_hf_if_needed()` en `__init__()`.
   - Disparar `trigger_async_sync()` al finalizar `process_query()`.
   - Añadir método *thread-safe* `record_feedback(session_id: str, turn_index: int, feedback: str)` que actualiza SQLite, regenera los archivos planos y dispara `trigger_async_sync()`.

5. **[`src/app/ui.py`](file:///d:/Proyectos/AiHoreca/src/app/ui.py) (Interfaz Streamlit):**
   - **Aislamiento por visitante:** Mantener en `st.session_state["visitor_session_ids"]` la lista de `session_id` creados en esa pestaña del navegador y mostrar en la barra lateral únicamente esas sesiones.
   - **Botones `👍 / 👎` por turno:** Renderizar debajo de cada respuesta del asistente en el historial del chat dos botones compactos (`👍 Útil` / `👎 Mejorar`) que invocan `service.record_feedback(session_id, turn_index, "OK" | "KO")` y muestran confirmación visual sin alterar el resto de la conversación.

6. **Dependencias y Control de Versiones ([`.gitignore`](file:///d:/Proyectos/AiHoreca/.gitignore), [`requirements.txt`](file:///d:/Proyectos/AiHoreca/requirements.txt)):**
   - Añadir `huggingface-hub>=0.25.0` a `requirements.txt`.
   - Configurar `.gitignore` para excluir `.env`, `.venv/`, `.antigravity/`, `data/sessions/`, `data/logs/`, `data/qdrant_storage/.lock` y `data/raw/*.pdf`, manteniendo `data/processed/` y `data/qdrant_storage/collection/` (~20,5 MB) para que Streamlit Cloud arranque con el índice vectorial listo sin re-indexar.

---

## 5. Pasos para Publicar en Streamlit Community Cloud + Hugging Face Dataset

1. **Dataset Privado para Logs en Hugging Face (Completado):**
   - Repositorio privado creado en `https://huggingface.co/datasets/tu-usuario/tu-dataset-logs`.
2. **Crear un Access Token con permiso de escritura en Hugging Face:**
   - En `https://huggingface.co/settings/tokens`, crear un token de tipo `Write` (`hf_...`).
3. **Subir el código del proyecto a un repositorio en GitHub:**
   - Gracias al [`.gitignore`](file:///d:/Proyectos/AiHoreca/.gitignore), ni tu `.env`, ni tu `.venv`, ni tus sesiones locales ni los PDFs crudos se subirán a GitHub, pero sí subirá el índice vectorial pre-calculado (`data/qdrant_storage/`).
4. **Conectar el repositorio en Streamlit Community Cloud (`share.streamlit.io`):**
   - Entrar en `https://share.streamlit.io` con tu cuenta de GitHub y pulsar **Create app**.
   - Seleccionar el repositorio de GitHub, rama `main`, y como **Main file path** indicar: `src/app/ui.py`.
   - En **Advanced settings -> Secrets**, pegar las 4 variables en formato TOML:
     ```toml
     GOOGLE_API_KEY = "AIza..."
     COHERE_API_KEY = "..."
     HF_LOG_TOKEN = "hf_..."
     HF_LOG_DATASET_REPO = "tu-usuario/tu-dataset-logs"
     ```
   - Pulsar **Deploy**.
