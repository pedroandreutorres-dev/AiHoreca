# AiHoreca

Asistente de consultas laborales para el sector de hostelería y turismo en España. Responde dudas sobre salarios, tablas retributivas, clasificación profesional, jornada, permisos, fijos discontinuos y régimen disciplinario cruzando tres niveles normativos:

1. **Estatuto de los Trabajadores** (mínimo legal estatal).
2. **VI Acuerdo Laboral Estatal de Hostelería — ALEH VI** (áreas funcionales 1ª a 6ª, grupos profesionales y faltas/sanciones).
3. **Convenios colectivos provinciales y autonómicos** (tablas salariales y condiciones específicas de cada territorio).

**Territorios indexados actualmente (1.591 fragmentos en Qdrant Cloud):**
- **Comunidad de Madrid**
- **Cataluña** (*Barcelona, Girona, Tarragona y Lleida*)
- **Islas Baleares** (*Mallorca, Menorca, Ibiza y Formentera*)
- **León**
- **Málaga**
- **Sevilla**
- **Alicante**
- **Las Palmas** (*Gran Canaria, Lanzarote y Fuerteventura*)
- **Santa Cruz de Tenerife** (*Tenerife, La Palma, La Gomera y El Hierro*)
- **Cádiz**
- **Granada**

---

## Cómo funciona

```text
Consulta ──► 1. Análisis y validación (LangGraph + Gemini)
                ├─ Fuera del sector o provincia no soportada           ──► Aviso determinista de alcance (0 llamadas BBDD)
                ├─ Sin pregunta normativa o falta provincia/puesto     ──► Bienvenida o solicitud de datos (0 llamadas BBDD)
                └─ Consulta normativa completa (1 o N territorios)
                        │
                        ▼ (fan-out paralelo con Send)
             2. Búsqueda híbrida paralela (Qdrant Cloud + Cohere Rerank)
                • Ramas independientes por territorio + capa estatal (ET y VI ALEH)
                • Vectores densos (gemini-embedding-001, caché single-flight) + BM25 en CPU + top_k/cuotas adaptativos
                        │
                        ▼
             3. Verificación y ampliación condicional (CPU, máx. 1 salto)
                • Comprueba cobertura y remisiones entre artículos/anexos; recupera el fragmento ausente si hace falta
                        │
                        ▼
             4. Síntesis y guardrail numérico (máx. 1 reintento)
                • Resuelve cálculos monetarios/proporcionales en un turno citando artículo y valida cada cifra en CPU
```

- **Ingesta y fragmentación (`src/ingestion/`, `src/retrieval/chunker.py`):** Recorta boletines oficiales (BOE, BOCM, DOGC, BOIB, BOP), aplica correcciones de erratas, extrae texto y tablas con rejilla en CPU (`PyMuPDF`/`pymupdf4llm`) —reservando Gemini Multimodal (`temperature=0.0`) para tablas sin bordes— y trocea por artículo, disposición o tabla conservando su cabecera jerárquica.
- **Recuperación multi-tabla adaptativa (`src/retrieval/vector_store.py`):** Soporta convenios con tabla salarial única o separados en varios anexos (importes, niveles y categorías); ajusta `top_k` según la intención limpia del turno y evita inyectar tablas salariales en consultas no retributivas.
- **Streaming, memoria y RGPD (`src/app/ui.py`, `src/graph/memory.py`, `src/app/telemetry_sync.py`):** Muestra progreso nodo a nodo en Streamlit (`graph.stream`), persiste sesiones en SQLite WAL en $O(1)$ desacoplando en segundo plano la exportación de logs planos y el respaldo opcional en Hugging Face, y purga sesiones inactivas a los 90 días (art. 5.1.e RGPD).
- **Telemetría de tokens y auditoría (`src/app/audit_runner.py`, `src/app/ui.py`):** Registra por paso los tokens reales reportados por la API (`usage_metadata`), failovers, aciertos de caché de embeddings y `search_units` de Cohere; los expone en vivo en el panel *Rayos-X* y los consolida por consulta, modelo y etapa en el informe diario, transcripciones e `historico_metricas.csv`.

---

## Estructura del repositorio

```text
AiHoreca/
├── data/
│   ├── raw/                  # PDFs originales de boletines oficiales (no versionados)
│   ├── processed/            # Convenios en Markdown (.md), caché de páginas y golden set
│   ├── sessions/             # Base de datos SQLite de sesiones (no versionada)
│   └── logs/                 # Telemetría e informes de auditoría (no versionados)
├── notebooks/
│   └── 01_evaluacion_ragas_y_auditoria.ipynb
├── src/
│   ├── config.py             # Modelos, cuotas de búsqueda y conexión a Qdrant
│   ├── ingestion/            # Extracción PDF a Markdown y aplicación de erratas
│   ├── retrieval/            # Chunker e indexación/búsqueda híbrida en Qdrant
│   ├── graph/                # Flujo en LangGraph, esquemas Pydantic y verificador numérico
│   ├── evaluation/           # Evaluador RAGAS y tests de regresión (38 casos)
│   └── app/                  # Streamlit (streaming y Rayos-X), servicio, CLI y auditor diario
├── requirements.txt
└── README.md
```

---

## Instalación y uso

### 1. Entorno
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Variables necesarias en `.env` (ver [`.env.example`](.env.example)):
- `GOOGLE_API_KEY`: clave de Google AI Studio.
- `COHERE_API_KEY`: clave de Cohere.
- `QDRANT_URL` y `QDRANT_API_KEY`: conexión al clúster de Qdrant Cloud (si no se definen, usa almacenamiento local en `data/qdrant_storage/`).
- `HF_LOG_TOKEN` y `HF_LOG_DATASET_REPO`: *(opcional)* respaldo remoto de telemetría en Hugging Face Dataset para despliegues efímeros.

### 2. Añadir o actualizar convenios
```powershell
# 1. Convertir a Markdown un PDF nuevo o todo data/raw/ con Gate de Calidad en CPU
python -m src.ingestion.pdf_ingestor --only convenio_sevilla.pdf --validate
python -m src.ingestion.validator --file convenio_sevilla.md --strict

# 2. Subir o actualizar un convenio concreto en Qdrant Cloud sin reindexar los demás
python -m src.retrieval.vector_store --source-file convenio_sevilla.md

# (Opcional) Migrar colección local a Qdrant Cloud sin recalcular embeddings
python -m src.retrieval.vector_store --sync-to-cloud
```

### 3. Ejecución y tests de regresión
El proyecto cuenta con una suite unitaria/canario en CPU (**52 tests** en `tests/test_ingestion_pipeline.py`) y una batería de **38 casos de prueba (`GD_01`–`GD_38`)** con soporte de reanudación incremental (`--resume`). La regresión completa `--quick --force` alcanzó el `100%` (`32/32`) hasta Las Palmas, y los casos de Santa Cruz de Tenerife (`GD_33`–`GD_34`), Cádiz (`GD_35`–`GD_36`) y Granada (`GD_37`–`GD_38`) quedaron verificados *end-to-end* verbatim (`0` alucinaciones numéricas) junto con comprobación directa (Qdrant + Cohere) sobre provincias salariales previas. La línea base semántica RAGAS está fijada en [`data/processed/evaluation_baseline.json`](data/processed/evaluation_baseline.json) sobre los 24 casos previos (`0.979` en *faithfulness* y `0.997` en *answer relevancy*):

```powershell
# Gate de Calidad del corpus Markdown Silver + suite de ingesta en CPU (0 tokens API)
python -m src.ingestion.validator
python -m unittest discover -s tests -v

# Comprobación rápida determinista RAG en CPU (~5s, sin coste de juez LLM)
python -m src.evaluation.regression --quick

# Evaluación completa con juez semántico RAGAS
python -m src.evaluation.regression --force

# Interfaz web (Streamlit), consulta CLI o auditoría diaria (usa --force-audit-all para regenerar el informe del día)
streamlit run src/app/ui.py
python -m src.app.cli --session demo --query "Soy cocinero en un hotel de 4 estrellas en Palma de Mallorca, ¿cuál es mi salario base?"
python -m src.app.audit_runner
```
