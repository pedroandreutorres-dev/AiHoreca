# AiHoreca — Asistente Legal RAG Agéntico para el Sector de Hostelería

**AiHoreca** es un sistema de **Recuperación Híbrida Orquestada por Grafos de Estado (Agentic Hybrid RAG)** especializado en derecho laboral para el sector de la hostelería y el turismo en España.

El sistema está diseñado para ejecutarse íntegramente en **entornos CPU (sin necesidad de GPU local)** mediante una arquitectura de ingesta híbrida automatizada (`PyMuPDF` + `google-genai`) agnóstica al boletín oficial. Incorpora **control de regresión continuo con coste 0 de tokens**, **auditoría diaria automatizada** y **política de retención de datos alineada con el RGPD (art. 5.1.e)**.

---

## 1. Jerarquía Normativa del Corpus (3 Capas)

El motor aplica el principio de jerarquía normativa y norma más favorable consultando en paralelo tres niveles legales:

1. **Capa 1 — Mínimo de Derecho Necesario (Estatal):** *Estatuto de los Trabajadores* (BOE).
2. **Capa 2 — Estructura Sectorial y Clasificación Profesional (Estatal):** *Acuerdo Laboral de Ámbito Estatal para el Sector de Hostelería — ALEH* (BOE).
3. **Capa 3 — Condiciones Económicas y Laborales (Provincial / Autonómica):** Convenios colectivos provinciales o autonómicos de hostelería y turismo (BOCM, DOGC, BOP, etc.), incluyendo corrección automática de erratas y consultas comparativas multi-territoriales.

---

## 2. Stack Tecnológico (100% CPU)

| Componente | Tecnología | Función Principal |
| :--- | :--- | :--- |
| **Ingesta y Tablas en CPU** | `PyMuPDF` (`pymupdf` + `pymupdf4llm`) | Recorte de páginas en memoria, extracción de texto en CPU y conversión vectorial de tablas a Markdown (`\|...\|`). |
| **Ingesta Multimodal** | `google-genai` SDK + `Pydantic` | Detección de límites de documento `[start, end]` (`boundaries_cache.json`), tablas complejas sin bordes y parches de erratas. |
| **Base Vectorial Híbrida** | `Qdrant` (Local `data/qdrant_storage/`) | Búsqueda paralela en 3 capas, estratificación de contenido y filtrado multi-provincia con `MatchAny`. |
| **Embeddings Densos / Dispersos**| Gemini `gemini-embedding-001` + `FastEmbed` BM25 | Representación semántica (`Cosine`, 3072 dims) combinada con búsqueda léxica en CPU para artículos y cuantías. |
| **Reranker Estratificado** | Cohere `rerank-multilingual-v3.0` | Reordenación *cross-encoder* con cuotas mínimas garantizadas por capa normativa ([`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py)). |
| **Orquestación Agéntica** | `LangGraph` + `Pydantic v2` | Máquina de estados con guardarraíles de entrada/salida, enrutamiento multi-provincia y solicitud de aclaración (HITL). |
| **Auditor Aritmético de Salida** | CPU Determinista ([`src/graph/nodes.py`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py)) | Validación de cifras literales (`€`, `%`) y combinatoria de subconjuntos (*subset-sum*) para salarios base y complementos. |
| **Memoria y RGPD (90 días)** | `SQLite` WAL + Exportación Relacional | Separación de memoria operativa y episódica, con **purga automática de sesiones inactivas a los 90 días** (art. 5.1.e RGPD). |
| **Modelos LLM** | Google AI Studio (`google-genai`) | Enrutador ligero y sintetizador con rotación automática (*failover*) ante latencias o límites de cuota. |
| **Regresión Continua** | Engine Determinista ([`src/evaluation/`](file:///d:/Proyectos/AiHoreca/src/evaluation/)) | Suite de regresión rápida (0 tokens, <5s) contra línea base congelada ([`evaluation_baseline.json`](file:///d:/Proyectos/AiHoreca/data/processed/evaluation_baseline.json)). |
| **Auditoría Diaria** | `audit_runner.py` + Hugging Face Hub | Descarga incremental de telemetría, detección de votos tardíos, seguimiento de KPIs y compactación periódica Git. |
| **Servicio e Interfaz** | `Streamlit` + Consola REPL CLI | Singleton *thread-safe* con aislamiento por visitante, inspector de trazabilidad RAG y consola de terminal. |

---

## 3. Arquitectura del Sistema

### 3.1. Pipeline de Ingesta Híbrido
- **Detección de Límites:** `PyMuPDF` + `Gemini` localizan el rango `[start_page, end_page]` del convenio, ignorando páginas ajenas de boletines oficiales.
- **Extracción en 3 Niveles:** Texto articulado directo en CPU (0 llamadas API), tablas vectoriales mediante `pymupdf4llm` y tablas sin bordes complejas vía Gemini Multimodal (`temperature=0.0`).
- **Auto-Errata Patcher & Chunking:** Aplica correcciones de erratas sobre el convenio matriz y fragmenta jerárquicamente inyectando metadatos `[NORMA | CAPA | TERRITORIO | SECCIÓN]`.

### 3.2. Flujo del Grafo Agéntico ([`src/graph/`](file:///d:/Proyectos/AiHoreca/src/graph/))

```text
               ┌───────────────────────────────┐
               │  Entrada de Usuario / Turno   │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 1. Carga Contexto SQLite WAL  │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 2. analyze_and_route_node     │
               │    Prevalencia Turno > Sesión │
               └───────┬───────────────┬───────┘
                       │               │
     [Fuera de Ámbito] │               │ [Falta Territorio]
                       ▼               ▼
       ┌───────────────────┐       ┌───────────────────────┐
       │ Cortocircuito OOS │       │ ask_clarification     │
       │ (0 tokens gen.)   │       │ (Pide provincia HITL) │
       └───────────────────┘       └───────────────────────┘
                       │               │
                       └───────┬───────┘
                               │ [1 o N Provincias Válidas]
                               ▼
               ┌───────────────────────────────┐
               │ 3. parallel_retrieval_node    │
               │    Qdrant Híbrido + MatchAny  │
               │    Capa 1 + Capa 2 + Capa 3   │
               │    Cohere Rerank Estratificado│
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 4. legal_synthesis_and_audit  │
               │ • Atribución Jurisdiccional   │
               │ • Salarios brutos oficiales   │
               │ • Auditor Aritmético en CPU   │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 5. Persistencia y Telemetría  │
               │    SQLite + CSV + Purga RGPD  │
               └───────────────────────────────┘
```

1. **Prevalencia Temporal del Turno:** La información del mensaje actual manda sobre los *slots* heredados del diálogo anterior, evitando contaminación de estado (*anti-poisoning*).
2. **Atribución Jurisdiccional Estricta:** Prohíbe aplicar convenios fuera de su ámbito territorial. En consultas comparativas (ej. Madrid vs León), el recuperador obtiene los fragmentos con `MatchAny` y la síntesis desglosa las condiciones bajo epígrafes independientes sin cruce analógico.
3. **Auditoría Numérica en CPU:** Verifica cada cifra contra los fragmentos citados y valida sumas complejas mediante combinatoria de subconjuntos (*subset-sum*: $T = \text{Base} + \sum C_i$), sin reglas manuales por concepto salarial.

---

## 4. Estructura del Proyecto

```text
AiHoreca/
├── data/
│   ├── raw/                                  # Boletines Oficiales en PDF (excluidos de Git)
│   ├── processed/                            # Convenios en Markdown, caché de límites y baselines
│   │   ├── *.md                              # Textos normalizados del corpus
│   │   ├── golden_set_produccion.jsonl       # Casos promovidos desde telemetría
│   │   └── evaluation_baseline.json          # Línea base congelada para tests de regresión
│   ├── qdrant_storage/                       # Base vectorial local pre-calculada (21 MB)
│   ├── sessions/                             # Base de datos SQLite multi-sesión (excluida de Git)
│   └── logs/                                 # Telemetría, KPIs históricos y candidatos (excluidos de Git)
├── notebooks/
│   └── 01_evaluacion_ragas_y_auditoria.ipynb # Notebook interactivo de auditoría y análisis
├── src/
│   ├── config.py                             # Parámetros globales, modelos, retención RGPD y cuotas
│   ├── ingestion/                            # Ingesta PDF en 3 niveles y parches de erratas
│   ├── retrieval/                            # Chunking jerárquico y Qdrant híbrido con MatchAny
│   ├── graph/                                # Nodos LangGraph, memoria SQLite, failover y guardarraíles
│   ├── evaluation/                           # Golden Dataset, evaluador y suite de regresión continua
│   └── app/                                  # Servicio central, runner diario, UI Streamlit y CLI
├── docs/                                     # Documentación técnica interna y de despliegue (en .gitignore)
├── Iniciar_AiHoreca.bat                      # Lanzador rápido para entornos Windows
├── requirements.txt                          # Dependencias del proyecto
└── README.md
```

---

## 5. Control de Regresión Continua (`src/evaluation/`)

El sistema incluye una batería de pruebas de regresión ([`src/evaluation/regression.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/regression.py)) que compara en segundos cualquier cambio en los prompts o el grafo contra una línea base congelada ([`data/processed/evaluation_baseline.json`](file:///d:/Proyectos/AiHoreca/data/processed/evaluation_baseline.json)):

- **Cobertura del Golden Dataset (14 Casos):** Audita bloqueos deterministas fuera de ámbito (`GD_01`–`GD_02`), solicitudes proactivas de provincia (`GD_04`), consultas provinciales (`GD_05`–`GD_07`), correferencia y desambiguación (`GD_08`, `GD_13`), consultas estatales puras (`GD_12`) y comparativas multi-territoriales simultáneas (`GD_14`).
- **Modo Rápido Determinista (0 Tokens / ~5 segundos):** Valida enrutamiento, provincias detectadas y tasa de 0 alucinaciones numéricas en CPU sin invocar jueces LLM externos:
  ```powershell
  python -m src.evaluation.regression --quick
  ```
- **Actualización de Línea Base:** Si se introduce una mejora aprobada o se amplía el Golden Set con nuevos casos, se congela el nuevo estándar con:
  ```powershell
  python -m src.evaluation.regression --quick --update-baseline
  ```

---

## 6. Auditoría Diaria y Cumplimiento RGPD (`src/app/`)

- **Auditoría Automática (`src/app/audit_runner.py`):** Sincroniza la telemetría de producción desde Hugging Face Hub, detecta turnos nuevos incrementalmente vía cursor, identifica votos tardíos (`last_feedback_at`), exporta KPIs a `historico_metricas.csv` y deriva anomalías a `candidatos_golden.jsonl`.
  ```powershell
  python -m src.app.audit_runner
  ```
  *(Se ejecuta automáticamente a las 08:00 Europe/Madrid mediante sidecar programado).*
- **Ciclo de Vida RGPD (Art. 5.1.e):**
  1. *Purga en Origen:* [`SessionMemoryStore.purge_inactive_sessions(retention_days=90)`](file:///d:/Proyectos/AiHoreca/src/graph/memory.py) elimina de SQLite sesiones y turnos sin actividad en más de 90 días.
  2. *Purga Local:* Limpieza periódica de archivos de texto y transcripciones antiguas.
  3. *Squash en Git Remoto:* Compactación mensual del historial en Hugging Face (`HfApi.super_squash_history`) para impedir la persistencia residual de datos personales en commits históricos.

---

## 7. Instalación y Uso Rápido

### 7.1. Configuración del Entorno
```powershell
# 1. Crear y activar entorno virtual (Windows / PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno en .env
# GOOGLE_API_KEY=AIza...
# COHERE_API_KEY=...
```

### 7.2. Pipeline de Ejecución
```powershell
# Ingesta e indexación vectorial (solo necesaria si se modifica el corpus original)
python -m src.ingestion.pdf_ingestor
python -m src.retrieval.vector_store --recreate

# Test de regresión antes de arrancar
python -m src.evaluation.regression --quick

# Interfaz Web interactiva (Streamlit)
streamlit run src/app/ui.py

# O bien, consulta directa por terminal (CLI)
python -m src.app.cli --session ses-demo --query "¿Cuál es el salario de un camarero en León?"
```

---

## 8. Metodología de Desarrollo

- **Cero Parches Ad-hoc:** Se prohíbe el uso de diccionarios *hardcoded* de municipios o reglas manuales por consulta. Las desambiguaciones las resuelve el Gatekeeper semántico y los cálculos se auditan mediante formulaciones algebraicas generalistas.
- **Trazabilidad y Reproducibilidad:** Todo código de producción reside estrictamente en `src/`. Ningún cambio pasa a la rama principal si provoca regresiones respecto a la línea base congelada.
