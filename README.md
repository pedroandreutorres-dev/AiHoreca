# AiHoreca — Asistente Legal RAG Agéntico para el Sector de Hostelería

**AiHoreca** es un sistema de **Recuperación Híbrida Orquestada por Grafos de Estado (Agentic Hybrid RAG)** especializado en derecho laboral para el sector de la hostelería y el turismo en España.

El sistema está diseñado para ejecutarse íntegramente en **entornos CPU (sin necesidad de GPU local)** mediante una arquitectura de ingesta híbrida automatizada (`PyMuPDF` + `google-genai`) y agnóstica al boletín oficial, preparada para escalar desde las provincias inicialmente indexadas hacia cualquier convenio colectivo provincial o autonómico sin reglas manuales (*hardcoded*) por documento.

---

## 1. Jerarquía Normativa del Corpus (3 Capas)

El motor aplica el principio de jerarquía normativa y norma más favorable consultando en paralelo tres niveles legales:

1. **Capa 1 — Mínimo de Derecho Necesario (Estatal):**
   - *Estatuto de los Trabajadores* (BOE).
2. **Capa 2 — Estructura Sectorial y Clasificación Profesional (Estatal):**
   - *Acuerdo Laboral de Ámbito Estatal para el Sector de Hostelería — ALEH* (BOE).
3. **Capa 3 — Condiciones Económicas y Laborales (Provincial / Autonómica):**
   - Convenios colectivos provinciales o autonómicos de hostelería y turismo publicados en los respectivos boletines oficiales (BOCM, DOGC, BOP, etc.), incluyendo soporte automático para correcciones de erratas posteriores.

---

## 2. Stack Tecnológico (100% CPU)

| Capa | Tecnología | Rol en el Sistema |
| :--- | :--- | :--- |
| **Motor PDF y Tablas Vectoriales en CPU** | `PyMuPDF` (`pymupdf` + `pymupdf4llm`) | Escaneo de cabeceras, recorte de páginas en memoria, extracción de texto articulado en CPU y conversión directa de tablas con rejilla vectorial completa a Markdown (`|...|`). |
| **Ingesta Multimodal y Tablas Complejas** | `google-genai` SDK + `Pydantic` | Detección automatizada de límites de documento `[start_page, end_page]` (`boundaries_cache.json`), reconstrucción de tablas sin bordes verticales o con celdas combinadas y aplicación automática de correcciones de erratas (`ErrataPatchList`). |
| **Base de Datos Vectorial** | `Qdrant` (Modo local `data/qdrant_storage/`) | Colección vectorial híbrida con búsqueda paralela aislada por capa normativa y estratificación intra-provincial (`articulo`, `tabla_salarial`, `anexo_o_disposicion`). |
| **Embeddings Densos** | Google `gemini-embedding-001` | Representación semántica multilingüe (`Cosine`), indexada por lotes controlados por límite de tokens por minuto. |
| **Embeddings Dispersos** | `FastEmbed` (`Qdrant/bm25` con `IDF`) | Búsqueda léxica en CPU para referencias normativas exactas, grupos profesionales y cuantías. |
| **Reranker y Selección Estratificada** | Cohere `rerank-multilingual-v3.0` + Cuotas Jerárquicas | Reordenación *cross-encoder* con priorización automática de tablas salariales vigentes, cobertura multi-intención retributiva y cuotas mínimas post-rerank por capa normativa parametrizadas en `src/config.py`. |
| **Orquestación Agéntica** | `LangGraph` + `Pydantic v2` | Máquina de estados con descubrimiento territorial dinámico, guardarraíles de entrada/salida (`AuditReport` con validación de cálculos aritméticos derivados `DerivedCalculation`), solicitud de aclaración territorial (HITL) y traducción de oficios coloquiales al ALEH. |
| **Memoria Conversacional y Telemetría** | `SQLite` (`sqlite3` WAL en `data/sessions/`) + Exportación Relacional (`data/logs/`) | Persistencia transaccional aislada por `session_id` con separación entre *Working Memory* (`SessionContextSlots`) y *Episodic Memory* (`semantic_digest`), progresión conversacional sin repetición entre turnos y generación automática de logs en texto plano y CSV. |
| **Modelos LLM** | Google AI Studio (`google-genai`) | Parametrizados centralmente en `src/config.py` con soporte de rotación automática (*failover*) ante límites de cuota o latencia. |
| **Evaluación y Auditoría** | Motor Determinista + Juez Estructurado `RAGAS` (`src/evaluation/`) | Suite de evaluación reproducible sobre un *Golden Dataset* que audita precisión de enrutamiento, cobertura multi-capa, fidelidad numérica y métricas semánticas RAGAS. |
| **Capa de Servicio e Interfaz (`src/app/`)** | `Streamlit` + Consola REPL CLI (`AiHorecaEngineService`) | Singleton *thread-safe* con recarga automática ante cambios en el código (`compute_code_version`), aislamiento de sesiones por visitante web, inspector de trazabilidad por turno y panel de auditoría del corpus. |

---

## 3. Arquitectura del Sistema

### 3.1. Pipeline de Ingesta Híbrido en 3 Niveles e Indexación Vectorial

```text
[ PDFs Crudos en /data/raw/ ]
              │
              ├──► 1. Detector de Límites de Página (PyMuPDF + google-genai + Pydantic)
              │       • Filtra cabeceras de boletín en CPU y devuelve `DocumentBoundary(start_page, end_page)`.
              │       • Persiste en `boundaries_cache.json` y recorta exclusivamente el convenio aplicable.
              │
              ├──► 2. Extracción Híbrida en 3 Niveles a Markdown (CPU + Gemini Multimodal)
              │       • Nivel 1 (Texto articulado): extracción directa en CPU con `pymupdf` (0 llamadas API).
              │       • Nivel 2 (Tablas con rejilla vectorial completa): conversión en CPU con `pymupdf4llm`.
              │       • Nivel 3 (Tablas sin bordes verticales / rowspans): procesamiento multimodal por lotes PDF
              │         (`temperature=0.0`) para reconstruir tablas Markdown estructuradas (`|...|`).
              │
              ├──► 3. Auto-Errata Patcher (PyMuPDF + google-genai + Pydantic)
              │       • Extrae `ErrataPatchList` y aplica/verifica las correcciones sobre el convenio matriz.
              │       ▼
        [ Documentos Markdown Limpios en /data/processed/ ]
              │
              └──► 4. Chunking Jerárquico e Indexación Híbrida (`src/retrieval/`)
                      • Fragmentación por Capítulo -> Artículo / Disposición / Tabla Salarial.
                      • Inyección de cabecera contextual `[NORMA | CAPA | TERRITORIO | SECCIÓN | REFERENCIA]`.
                      • Indexación híbrida en Qdrant (`data/qdrant_storage/`): Embeddings densos + BM25 con IDF.
```

### 3.2. Flujo del Agente y Memoria Multi-Sesión (`src/graph/` en LangGraph + SQLite)

1. **Carga de Contexto por `session_id` (`SessionMemoryStore`):** Recupera de SQLite los `SessionContextSlots` vigentes (`active_territorio`, `active_oficio_coloquial`, `active_categoria_aleh`, `pending_clarification_query`) y una ventana episódica de los últimos turnos con un resumen semántico limpio (`semantic_digest`) de lo explicado previamente por el asistente.
2. **Nodo Unificado de Análisis y Enrutamiento (`analyze_and_route_node`):** Ejecuta un pre-filtro determinista sobre el mensaje actual (`Turno Actual > Slot de Sesión`) y una llamada estructurada (`QueryAnalysisSchema`) que resuelve simultáneamente:
   - **Progresión Conversacional y Correferencia (`Caso A` vs `Caso B`):** Si existía una consulta previa detenida esperando provincia (`pending_clarification_query`), la fusiona con la provincia aportada (**Caso A**). Si es una repregunta en una conversación activa (**Caso B**), mantiene las condiciones del caso presentes en el diálogo pero centra `consulta_autocontenida_resuelta` exclusivamente en la nueva duda del turno actual sin volver a formular preguntas ya respondidas.
   - **Guardrail de Dominio:** Clasifica en `HOSPITALITY_CORE`, `UNSUPPORTED_PROVINCE` (provincias aún no indexadas) o `OUT_OF_SCOPE` (materias ajenas al ámbito laboral de hostelería).
   - **Gatekeeper Contextual:** Evalúa si `requiere_convenio_provincial == True` y comprueba si existe un territorio activo válido.
   - **Traductor Semántico al ALEH:** Mapea denominaciones coloquiales de puestos a su Área Funcional y Grupo Profesional oficial del Acuerdo Estatal.
3. **Nodos de Cortocircuito Determinista (`block_out_of_scope_node` / `ask_clarification_node`):** Rechazan consultas fuera de alcance (protegiendo los *slots* previos de la sesión contra *State Poisoning*) o solicitan proactivamente la provincia al usuario sin consumir tokens de generación.
4. **Nodo de Recuperación Paralela (`parallel_retrieval_node`):** Consulta simultáneamente en `Qdrant` el Estatuto de los Trabajadores, el ALEH y el Convenio Provincial estratificado (`articulo` + `tabla_salarial` + `anexo_o_disposicion`), priorizando las tablas salariales del año vigente, garantizando cobertura en consultas retributivas compuestas y aplicando cuotas jerárquicas post-rerank parametrizadas en `src/config.py`.
5. **Nodo de Síntesis y Auditoría Numérica (`legal_synthesis_and_audit_node`):**
   - **Síntesis con Continuidad Conversacional:** Recibe el historial reciente de la sesión para recordar los detalles aportados por el usuario sin repetir tablas ni explicaciones de turnos anteriores, adaptando el formato de salida a la intención de la pregunta actual (tablas salariales comparativas con/sin prorrateo y cálculos de hora extra en consultas retributivas; respuesta directa en consultas conceptuales o procedimentales).
   - **Auditor Determinista de Salida (`AuditReport` + `DerivedCalculation`):** Verifica en CPU cada cifra literal (`€`, `%`) contra los fragmentos recuperados y valida matemáticamente las operaciones aritméticas derivadas (proporcionalidad de jornada, prorrateo de pagas extraordinarias y fórmulas de horas extras).

---

## 4. Estructura del Proyecto

```text
AiHoreca/
├── data/
│   ├── raw/                          # Boletines Oficiales y textos consolidados en PDF
│   ├── processed/                    # Convenios procesados (.md), caché de límites y resultados de evaluación
│   ├── qdrant_storage/               # Colección vectorial híbrida local (Qdrant)
│   ├── sessions/                     # Base de datos SQLite multi-sesión (generada en ejecución)
│   └── logs/                         # Logs relacionales en texto plano y CSV (generados en ejecución)
├── notebooks/
│   └── 01_evaluacion_ragas_y_auditoria.ipynb # Notebook de auditoría del corpus y evaluación RAGAS
├── src/
│   ├── config.py                     # Configuración centralizada, rutas, modelos y cuotas de recuperación
│   ├── ingestion/
│   │   ├── schemas.py                # Contratos Pydantic (DocumentBoundary, ErrataPatchList)
│   │   └── pdf_ingestor.py           # Motor de ingesta híbrida en 3 niveles y Auto-Errata Patcher
│   ├── retrieval/
│   │   ├── schemas.py                # Contratos Pydantic (LegalChunk, RetrievedDocument)
│   │   ├── chunker.py                # Chunker jurídico jerárquico agnóstico a la provincia
│   │   └── vector_store.py           # Qdrant híbrido (Dense + BM25 IDF) + Cohere Rerank Estratificado
│   ├── graph/
│   │   ├── state.py                  # Estado LegalGraphState y esquemas Pydantic del grafo y auditoría
│   │   ├── memory.py                 # Gestor SQLite de sesiones, slots y exportación de logs relacionales
│   │   ├── nodes.py                  # Nodos del grafo, precedencia temporal, failover y Guardrail Numérico
│   │   └── workflow.py               # Compilación del StateGraph y suite de validación conversacional
│   ├── evaluation/
│   │   ├── golden_dataset.py         # Batería de casos de prueba (guardrails, RAG provincial y multi-turno)
│   │   └── evaluator.py              # Auditoría de corpus y evaluador determinista + Juez Semántico RAGAS
│   └── app/
│       ├── service.py                # Servicio singleton thread-safe AiHorecaEngineService
│       ├── telemetry_sync.py         # Sincronizador asíncrono opcional de telemetría hacia almacenamiento remoto
│       ├── ui.py                     # Interfaz web en Streamlit con aislamiento por visitante e inspector RAG
│       └── cli.py                    # Consola interactiva REPL y modo de consulta por línea de comandos
├── .env.example                      # Plantilla de variables de entorno
├── .gitignore                        # Reglas de exclusión para control de versiones
├── Iniciar_AiHoreca.bat              # Lanzador rápido opcional para entornos Windows
├── arquitectura_rag_hosteleria.md    # Especificación detallada de diseño de arquitectura
├── despliegue_web_y_telemetria_hf.md # Guía de publicación web y configuración de telemetría remota
├── requirements.txt                  # Dependencias del proyecto
└── README.md
```

---

## 5. Sistema de Evaluación y Auditoría Continua (`src/evaluation/`)

El proyecto incluye un framework de evaluación reproducible ([`src/evaluation/evaluator.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/evaluator.py)) sobre un conjunto de casos de control ([`src/evaluation/golden_dataset.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/golden_dataset.py)) que combina métricas deterministas en CPU con evaluación semántica mediante juez estructurado (`RAGAS`):

- **Métricas Deterministas de Enrutamiento y Recuperación:**
  - `Intent / Route / Territorio Accuracy`: Precisión en la detección de dominio, activación de guardarraíles y resolución territorial.
  - `Layer Coverage Score`: Presencia equilibrada de las tres capas normativas (Estatuto, ALEH y Convenio aplicable).
  - `Citation Recall@K`: Recuperación efectiva de los artículos y tablas salariales de referencia dentro de la ventana de contexto.
  - `Zero Numerical Hallucination Rate`: Verificación en el Nodo 5 de que el 100% de las cifras económicas y porcentajes emitidos provienen literalmente de los fragmentos recuperados o de cálculos derivados auditados.
- **Métricas Semánticas (`RAGAS`):**
  - `Faithfulness`: Fidelidad estricta de la respuesta respecto al contexto normativo recuperado.
  - `Answer Relevancy`: Adecuación directa a la consulta resuelta del turno.
  - `Context Recall Semántico`: Cobertura de la verdad de referencia (*ground truth*) en los fragmentos seleccionados.
  - `Hierarchy Application Score`: Correcta aplicación del principio de jerarquía normativa y norma más favorable (`Art. 3.3 ET`).

Los resultados de cada ejecución se exportan automáticamente a `data/processed/evaluation_results.json` y pueden consultarse tanto desde [`notebooks/01_evaluacion_ragas_y_auditoria.ipynb`](file:///d:/Proyectos/AiHoreca/notebooks/01_evaluacion_ragas_y_auditoria.ipynb) como desde la pestaña de Auditoría de la interfaz web.

---

## 6. Instalación, Configuración y Ejecución

1. **Crear/activar el entorno virtual e instalar dependencias:**
   ```bash
   # En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1

   # En Linux / macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Configurar variables de entorno:**
   Copia `.env.example` a `.env` y define las claves de API necesarias:
   ```ini
   GOOGLE_API_KEY=AIza...
   COHERE_API_KEY=...
   ```

3. **Comandos de ejecución por módulo:**
   ```bash
   # Fase 1: Auditar límites de páginas o ejecutar ingesta PDF -> Markdown
   python -m src.ingestion.pdf_ingestor --inspect-only
   python -m src.ingestion.pdf_ingestor

   # Fase 2: Fragmentar Markdowns e indexar en Qdrant local
   python -m src.retrieval.vector_store --recreate

   # Fase 3: Ejecutar la suite de validación multi-sesión del grafo agéntico
   python -m src.graph.workflow

   # Fase 4: Ejecutar la auditoría del corpus y el benchmark de evaluación
   python -m src.evaluation.evaluator

   # Fase 5 (Opción A - Interfaz Web en Streamlit):
   streamlit run src/app/ui.py
   # (En Windows también puedes ejecutar directamente Iniciar_AiHoreca.bat)

   # Fase 5 (Opción B - Consola interactiva en terminal CLI REPL o consulta puntual):
   python -m src.app.cli --session ses-demo
   python -m src.app.cli --session ses-demo --query "Consulta laboral de ejemplo"
   ```

### 6.1. Parametrización y Escalado de Modelos / Contexto (`src/config.py`)

Tanto los modelos de lenguaje como el presupuesto de fragmentos recuperados por capa normativa están centralizados en [`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py) y pueden ajustarse mediante variables de entorno sin modificar la lógica del grafo:
- **Modelos (`GEMINI_LITE_MODEL`, `GEMINI_REASONING_MODEL`):** Permite alternar entre modelos ligeros orientados a cuotas gratuitas (*Free Tier*) y modelos de mayor capacidad bajo facturación de pago (*Paid Tier*).
- **Presupuesto de Contexto RAG (`RETRIEVAL_TOP_K_PER_LAYER`, `RETRIEVAL_FINAL_TOP_K`, `QUOTA_*`):** Permite ampliar dinámicamente el número total de fragmentos y las cuotas por capa normativa para consultas jurídicas de alta complejidad.

### 6.2. Registro Relacional de Conversaciones y Despliegue Web

- **Registro Local Automático (Por defecto):** Cada turno completado se registra de forma inmediata (`T = 0 ms`, sin depender de que el usuario emita una valoración `👍 / 👎`) en tres formatos sincronizados localmente:
  1. `data/logs/conversaciones.txt`: Diálogos en texto plano limpio agrupados por `session_id` para lectura directa.
  2. `data/logs/turnos_metricas.csv`: Tabla relacional vinculada mediante `(session_id, turn_index)` con las métricas de enrutamiento, auditoría, latencia y valoración (`SIN_VOTO`, `OK`, `KO`).
  3. `data/sessions/aihoreca_sessions.db`: Base de datos SQLite completa.
- **Sincronización Remota Opcional (Despliegues Cloud Efímeros):** Si se despliega la aplicación en un contenedor en la nube con disco efímero (como *Streamlit Community Cloud*), cada usuario puede configurar opcionalmente en su entorno (`HF_LOG_TOKEN` y `HF_LOG_DATASET_REPO`) un repositorio de Dataset privado propio para persistir y restaurar automáticamente los logs en segundo plano. Consulta la guía detallada en [`despliegue_web_y_telemetria_hf.md`](file:///d:/Proyectos/AiHoreca/despliegue_web_y_telemetria_hf.md).

---

## 7. Metodología de Desarrollo (Modular-First)

- **Todo el código de producción reside en `src/`:** La ingesta documental, la indexación vectorial, la memoria multi-sesión, la telemetría y el grafo de `LangGraph` se implementan directamente en módulos `.py` tipados con `Pydantic`.
- **Uso de `notebooks/`:** Reservado exclusivamente para importar los módulos de `src/` y visualizar auditorías del corpus o resultados de evaluación.
