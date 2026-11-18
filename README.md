# AiHoreca

Asistente de consultas laborales para el sector de hostelería y turismo en España. Responde dudas sobre salarios, tablas retributivas, clasificación profesional, jornada, permisos, fijos discontinuos y régimen disciplinario cruzando tres niveles normativos:

1. **Estatuto de los Trabajadores** (mínimo legal estatal).
2. **VI Acuerdo Laboral Estatal de Hostelería — ALEH VI** (áreas funcionales 1ª a 6ª, grupos profesionales y faltas/sanciones).
3. **Convenios colectivos provinciales y autonómicos** (tablas salariales y condiciones específicas de cada territorio).

**Territorios indexados actualmente (782 fragmentos en Qdrant Cloud):**
- **Comunidad de Madrid**
- **Cataluña** (*Barcelona, Girona, Tarragona y Lleida*)
- **Islas Baleares** (*Mallorca, Menorca, Ibiza y Formentera*)
- **León**

---

## Cómo funciona

```text
Consulta ──► 1. Análisis y validación (LangGraph + Gemini)
                ├─ Fuera del sector o provincia no soportada ──► Aviso directo (sin generar respuesta)
                ├─ Falta provincia o puesto de trabajo       ──► Pide el dato necesario al usuario
                └─ Datos completos
                        │
                        ▼
             2. Búsqueda híbrida paralela (Qdrant Cloud + Cohere Rerank)
                • Vectores densos (gemini-embedding-001) + léxicos (BM25 en CPU)
                • Recupera en paralelo Estatuto + ALEH + Convenio (artículos y tablas de €, niveles y categorías)
                        │
                        ▼
             3. Respuesta y verificación numérica
                • Redacta citando artículo y convenio aplicable
                • Valida en CPU que cada cifra (€, %) salga de los fragmentos recuperados o de su cálculo directo
```

- **Ingesta de PDFs (`src/ingestion/`):** Recorta automáticamente las páginas del convenio dentro del boletín oficial (BOE, BOCM, DOGC, BOIB, BOP). El texto y las tablas con rejilla vectorial se extraen en CPU (`PyMuPDF` / `pymupdf4llm`); solo las tablas complejas sin bordes se procesan con Gemini Multimodal (`temperature=0.0`). Si existe corrección de erratas del convenio, se aplica sobre el documento base.
- **Fragmentación (`src/retrieval/chunker.py`):** Divide el Markdown por artículo, disposición o tabla salarial descartando índices iniciales y cabeceras de boletín. Añade a cada bloque su cabecera de contexto (`norma`, `capa`, `territorio`, `sección`, `referencia`).
- **Recuperación multi-tabla (`src/retrieval/vector_store.py`):** Soporta tanto convenios con tabla salarial única como convenios que separan los importes en euros, el nivel salarial del puesto y la categoría del establecimiento en distintos anexos.
- **Memoria y RGPD (`src/graph/memory.py`, `src/app/audit_runner.py`):** Guarda el contexto de cada sesión en SQLite (modo WAL) para mantener el hilo en repreguntas o cambios de provincia, y purga automáticamente las sesiones inactivas a los 90 días (art. 5.1.e RGPD).

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
│   ├── evaluation/           # Evaluador RAGAS y tests de regresión (22 casos)
│   └── app/                  # Interfaz Streamlit, CLI y auditor diario de telemetría
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

### 2. Añadir o actualizar convenios
```powershell
# 1. Convertir a Markdown los PDFs nuevos colocados en data/raw/
python -m src.ingestion.pdf_ingestor

# 2. Subir o actualizar un convenio concreto en Qdrant Cloud sin reindexar los demás
python -m src.retrieval.vector_store --source-file convenio_baleares.md

# (Opcional) Migrar colección local a Qdrant Cloud sin recalcular embeddings
python -m src.retrieval.vector_store --sync-to-cloud
```

### 3. Ejecución y tests de regresión
El proyecto cuenta con una batería de **22 casos de prueba (`GD_01`–`GD_22`)** contrastados contra [`data/processed/evaluation_baseline.json`](data/processed/evaluation_baseline.json) (`100%` en intención, enrutamiento, detección territorial y ausencia de alucinaciones; `0.981` en *faithfulness* y `0.994` en *answer relevancy*):

```powershell
# Comprobación rápida determinista en CPU (~5s, sin coste de juez LLM)
python -m src.evaluation.regression --quick

# Evaluación completa con juez semántico RAGAS
python -m src.evaluation.regression --force

# Interfaz web (Streamlit) o consulta desde terminal (CLI)
streamlit run src/app/ui.py
python -m src.app.cli --session demo --query "Soy cocinero en un hotel de 4 estrellas en Palma de Mallorca, ¿cuál es mi salario base?"
```
