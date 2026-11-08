# AiHoreca — Asistente Legal RAG Agéntico para el Sector de Hostelería

**AiHoreca** es un sistema de **Recuperación Híbrida Orquestada por Grafos de Estado (Agentic Hybrid RAG)** especializado en derecho laboral para el sector de la hostelería en España.

El proyecto está diseñado para ejecutarse íntegramente en **entornos CPU (sin GPU local)** y cuenta con una arquitectura de ingesta híbrida automatizada (`PyMuPDF` + `google-genai`) y agnóstica a la provincia, pensada para escalar desde el MVP inicial (**Madrid, Barcelona/Cataluña y León**) hacia las 50 provincias españolas sin reglas manuales (*hardcoded*) por boletín oficial.

---

## 1. Jerarquía Normativa del Corpus (3 Capas)

El motor respeta el principio de jerarquía normativa y norma más favorable consultando en paralelo tres niveles legales:

1. **Capa 1 — Mínimo de Derecho Necesario (Estatal):**
   - *Estatuto de los Trabajadores* (`estatuto_trabajadores_consolidado.pdf` - BOE).
2. **Capa 2 — Estructura Sectorial y Clasificación Profesional (Estatal):**
   - *VI Acuerdo Laboral de Ámbito Estatal para el Sector de Hostelería — ALEH* (`aleh_vi_estatal.pdf` - BOE).
3. **Capa 3 — Condiciones Económicas y Laborales (Provincial / Autonómica):**
   - **Madrid:** Convenio Colectivo de Hostelería y Actividades Turísticas (`convenio_madrid.pdf` - BOCM).
   - **Barcelona / Cataluña:** Convenio Colectivo Interprovincial de Hostelería y Turismo de Cataluña (`convenio_catalunya_bcn.pdf` + `convenio_catalunya_bcn_correccion_erratas.pdf` - DOGC).
   - **León:** Convenio Colectivo Provincial de Hostelería y Turismo de León (`convenio_leon_texto.pdf` - BOP de León).

---

## 2. Stack Tecnológico (100% CPU)

| Capa | Tecnología | Rol en el Sistema |
| :--- | :--- | :--- |
| **Motor PDF y Tablas Vectoriales en CPU** | `PyMuPDF` (`pymupdf` + `pymupdf4llm`) | Escaneo de cabeceras sin ruido, recorte en memoria, extracción de texto articulado en CPU y conversión directa de tablas con rejilla vectorial completa a Markdown (`|...|`) a coste 0€. |
| **Ingesta Multimodal y Tablas sin Bordes** | `google-genai` SDK (`gemini-3.5-flash` + cadena de fallback) + `Pydantic` | Detección de límites `[start_page, end_page]` (con caché en `boundaries_cache.json`), reconstrucción de tablas sin bordes verticales / *rowspans* complejos y parcheo automático de erratas (`ErrataPatchList`). |
| **Base de Datos Vectorial** | `Qdrant` (Modo local `data/qdrant_storage/`) | Colección híbrida `aihoreca_legal_corpus` (`648` chunks) con búsqueda paralela aislada por las 3 capas normativas y estratificación intra-provincial (`articulo`, `tabla_salarial`, `anexo_o_disposicion`). |
| **Embeddings Densos** | Google `gemini-embedding-001` | Representación semántica (`3072` dimensiones, `Cosine`) vía API Free Tier, indexada en lotes acotados (`<= 55.000` chars) bajo el límite de `30.000 TPM`. |
| **Embeddings Dispersos** | `FastEmbed` (`Qdrant/bm25` con `IDF`) | Búsqueda léxica en CPU para referencias exactas (*"Artículo 14"*, *"Grupo Profesional 2.º"*, *"1.731,57 €"*). |
| **Reranker y Selección Estratificada** | Cohere `rerank-multilingual-v3.0` + Cuotas Jerárquicas (`Top-10`) | Reordenación cross-encoder con priorización del año salarial vigente (`_extract_chunk_year`), recuperación multi-intención retributiva (pagas extras + horas extras) y cuotas post-rerank parametrizadas en `src/config.py` (`4` artículos provinciales + `2` tablas + `2` ALEH + `2` ET). |
| **Orquestación Agéntica** | `LangGraph` + `Pydantic v2` | Máquina de estados con descubrimiento territorial dinámico, guardarraíles de entrada/salida (`AuditReport` con verificación de cálculos derivados `DerivedCalculation`), repregunta HITL si falta provincia y traducción de oficios al ALEH VI. |
| **Memoria Conversacional Multi-Sesión** | `SQLite` (`sqlite3` WAL local en `data/sessions/aihoreca_sessions.db`) | Persistencia transaccional aislada por `session_id` con separación de *Working Memory* (`SessionContextSlots`) y *Episodic Memory* (ventana de 4 turnos con `semantic_digest` limpio de la respuesta previa), progresión conversacional sin repetición entre turnos y blindaje contra *State Poisoning*. |
| **Modelos LLM (Free Tier -> Paid Tier Ready)** | Google AI Studio (`google-genai`) | Parametrizado en `src/config.py`: `gemini-3.1-flash-lite` para clasificación, correferencia y enrutamiento (`QueryAnalysisSchema`); `gemini-3.5-flash` (con failover automático de 90s) para síntesis jurídica final (`LegalSynthesisSchema`). |
| **Evaluación y Auditoría Empírica** | Motor Determinista + Juez Estructurado `RAGAS` (`src/evaluation/`) | Benchmark sobre 12 casos del *Golden Dataset* (`100%` enrutamiento, `100%` cobertura de 3 capas, `100%` cero alucinación numérica, `98,57%` *Faithfulness* y `100%` *Answer Relevancy*). |
| **Capa de Servicio e Interfaz (`src/app/`)** | `Streamlit` + Consola REPL CLI (`AiHorecaEngineService`) | Singleton *thread-safe* para Qdrant Local con recarga automática ante cambios de código (`compute_code_version`), gestor visual de sesiones SQLite, inspector de Rayos-X por turno (Ruta, `AuditReport`, `Top-10` chunks con `rerank_score`) y dashboard de auditoría Fase 4. |

---

## 3. Arquitectura del Sistema

### 3.1. Pipeline de Ingesta Híbrido en 3 Niveles e Indexación Vectorial

```text
[ PDFs Crudos en /data/raw/ ]
              │
              ├──► 1. Detector de Límites de Página (PyMuPDF + google-genai + Pydantic)
              │       • Filtra cabeceras de boletín en CPU y devuelve `DocumentBoundary(start_page, end_page)`.
              │       • Persiste en `boundaries_cache.json` y recorta exclusivamente el Convenio de Hostelería.
              │
              ├──► 2. Extracción Híbrida en 3 Niveles a Markdown (CPU + Gemini Multimodal)
              │       • Nivel 1 (Texto articulado): extracción directa en CPU con `pymupdf` (0 coste API).
              │       • Nivel 2 (Tablas con rejilla vectorial completa): conversión en CPU con `pymupdf4llm`.
              │       • Nivel 3 (Tablas sin bordes verticales / rowspans): envío por lotes PDF a `google-genai`
              │         (`temperature=0.0`) para generar tablas Markdown exactas (`|...|`).
              │
              ├──► 3. Auto-Errata Patcher (PyMuPDF + google-genai + Pydantic)
              │       • Extrae `ErrataPatchList` y aplica/verifica las correcciones sobre el convenio matriz.
              │       ▼
        [ 5 Markdowns Limpios en /data/processed/ (~944 KB, 1.659 filas de tabla |...|) ]
              │
              └──► 4. Chunking Jerárquico e Indexación Híbrida (`src/retrieval/`)
                      • Fragmentación por Capítulo -> Artículo / Disposición / Tabla Salarial (`648` chunks).
                      • Inyección de cabecera contextual `[NORMA | CAPA | TERRITORIO | SECCIÓN | REFERENCIA]`.
                      • Indexación en Qdrant (`data/qdrant_storage/`): `gemini-embedding-001` (`3072` dims)
                        + `FastEmbed` (`Qdrant/bm25` con `IDF` en CPU).
```

### 3.2. Flujo del Agente y Memoria Multi-Sesión (`src/graph/` en LangGraph + SQLite)

1. **Carga de Contexto por `session_id` (`SessionMemoryStore`):** Recupera de `data/sessions/aihoreca_sessions.db` los `SessionContextSlots` vigentes (`active_territorio`, `active_oficio_coloquial`, `active_categoria_aleh`, `pending_clarification_query`) y una ventana episódica de los últimos 4 turnos que incluye un resumen semántico limpio (`semantic_digest` sin tablas Markdown pesadas) de lo explicado por el asistente en turnos previos.
2. **Nodo Unificado 0 + 1 + 2 (`analyze_and_route_node`):** Ejecuta un pre-filtro determinista sobre el turno actual (`Turno Actual > Slot de Sesión`) y una única llamada estructurada (`QueryAnalysisSchema` en `gemini-3.1-flash-lite`) que resuelve simultáneamente:
   - **Progresión Conversacional y Correferencia (`Caso A` vs `Caso B`):** Si existía `pending_clarification_query` (consulta detenida esperando provincia), la fusiona con la provincia aportada (**Caso A**). Si es una repregunta en conversación activa (`pending_clarification_query = null`, **Caso B**), hereda los datos del caso del historial (territorio, puesto o varios puestos, horas de contrato) pero centra `consulta_autocontenida_resuelta` exclusivamente en la nueva duda del turno actual sin re-preguntar lo ya resuelto en turnos anteriores.
   - **Guardrail de Dominio:** Clasifica en `HOSPITALITY_CORE`, `UNSUPPORTED_PROVINCE` (ej. Sevilla, Valencia) o `OUT_OF_SCOPE` (tráfico, otros sectores).
   - **Gatekeeper Contextual:** Evalúa si `requiere_convenio_provincial == True` y comprueba si existe territorio válido.
   - **Traductor Semántico ALEH VI:** Mapea el oficio coloquial (*"friegaplatos"*, *"pinche"*, *"rider"*, *"camarera"*) a su Área Funcional (1ª–6ª) y Grupo Profesional del VI ALEH.
3. **Nodos de Cortocircuito Determinista (`block_out_of_scope_node` / `ask_clarification_node`):** Rechazan consultas fuera de alcance (protegiendo los *slots* previos de la sesión contra *State Poisoning*) o solicitan proactivamente la provincia al usuario con **0 tokens adicionales de generación**.
4. **Nodo 3 (`parallel_retrieval_node`):** Consulta simultáneamente en `Qdrant` el Estatuto de los Trabajadores, el VI ALEH y el Convenio Provincial estratificado (`articulo` + `tabla_salarial` + `anexo_o_disposicion`), priorizando las tablas salariales del año más reciente (`_extract_chunk_year`), garantizando cobertura multi-intención retributiva (gratificaciones extraordinarias/pagas extras + horas extraordinarias) y aplicando cuotas jerárquicas post-rerank parametrizadas en `src/config.py` (`RETRIEVAL_FINAL_TOP_K = 10`: `4` artículos provinciales + `2` tablas salariales + `2` ALEH + `2` Estatuto).
5. **Nodos 4 y 5 (`legal_synthesis_and_audit_node`):**
   - **Síntesis con Continuidad Conversacional y Adecuación a la Intención:** Recibe el historial reciente de la sesión para recordar todas las condiciones del usuario (ej. contrato de 30h) sin repetir tablas ni explicaciones de turnos anteriores. Cuando el turno consulta retribuciones, presenta una tabla comparativa clara (**Sueldo 40h**, **Sueldo jornada contratada sin prorratear** en $N$ pagas y **Sueldo jornada contratada con pagas prorrateadas en 12 meses**) y calcula el valor en `€/hora` de las horas extraordinarias; cuando la repregunta es conceptual o procedimental, responde directamente a la duda sin forzar tablas salariales.
   - **Auditor Determinista de Salida (`AuditReport` + `DerivedCalculation`):** Verifica en CPU cada cifra literal (`€`, `%`) contra los fragmentos recuperados y valida matemáticamente los cálculos aritméticos derivados (proporcionalidad de jornada parcial, prorrateo en 12/14/15 pagas y fórmula de hora extra).

---

## 4. Estructura del Proyecto

```text
AiHoreca/
├── data/
│   ├── raw/                          # Boletines Oficiales y textos consolidados en PDF (6 archivos)
│   ├── processed/                    # Convenios limpios (.md), boundaries_cache.json y evaluation_results.json
│   ├── qdrant_storage/               # Colección vectorial híbrida local (648 fragmentos indexados)
│   └── sessions/                     # Base de datos SQLite multi-sesión (aihoreca_sessions.db)
├── notebooks/
│   └── 01_evaluacion_ragas_y_auditoria.ipynb # Notebook ejecutado de auditoría del corpus y evaluación RAGAS
├── src/
│   ├── config.py                     # Configuración centralizada, rutas, modelos Free Tier y cliente GenAI
│   ├── ingestion/
│   │   ├── schemas.py                # Contratos Pydantic (DocumentBoundary, ErrataPatchList)
│   │   └── pdf_ingestor.py           # Motor de ingesta híbrida en 3 niveles y Auto-Errata Patcher
│   ├── retrieval/
│   │   ├── schemas.py                # Contratos Pydantic (LegalChunk, RetrievedDocument)
│   │   ├── chunker.py                # Chunker jurídico jerárquico agnóstico a la provincia
│   │   └── vector_store.py           # Qdrant híbrido (Dense 3072 + BM25 IDF) + Cohere Rerank Estratificado
│   ├── graph/
│   │   ├── state.py                  # Estado LegalGraphState y esquemas Pydantic (SessionContextSlots, QueryAnalysis, LegalSynthesis, AuditReport)
│   │   ├── memory.py                 # Gestor transaccional SQLite de sesiones, slots y memoria episódica compacta
│   │   ├── nodes.py                  # Nodos del grafo, precedencia temporal, failover y Guardrail Numérico de Salida
│   │   └── workflow.py               # Compilación del StateGraph y suite de validación multi-sesión (6 turnos en 2 sesiones)
│   ├── evaluation/
│   │   ├── golden_dataset.py         # 12 casos auditados del Golden Dataset (guardrails, RAG provincial y memoria multi-turno)
│   │   └── evaluator.py              # Auditoría de corpus y evaluador determinista + Juez Semántico RAGAS
│   └── app/
│       ├── service.py                # Singleton thread-safe AiHorecaEngineService (Qdrant + LangGraph + SQLite)
│       ├── ui.py                     # Interfaz web multi-sesión en Streamlit con inspector de Rayos-X RAG y Dashboard
│       └── cli.py                    # Consola interactiva REPL y modo one-shot por línea de comandos
├── .env.example                      # Plantilla de variables de entorno (API Keys y Dataset HF de logs)
├── .gitignore                        # Exclusión de secretos (.env), entorno (.venv), sesiones locales y PDFs crudos
├── Iniciar_AiHoreca.bat              # Lanzador directo en 1 clic (Windows) con liberador de bloqueo Qdrant/puerto 8501
├── arquitectura_rag_hosteleria.md    # Especificación detallada de arquitectura y resultados empíricos
├── despliegue_web_y_telemetria_hf.md # Especificación de publicación en Streamlit Cloud + telemetría en Hugging Face Dataset
├── requirements.txt                  # Dependencias del entorno
└── README.md
```

---

## 5. Resultados Empíricos de Evaluación (Benchmark Fase 4 — 12 Casos)

| Métrica de Auditoría | Resultado |
| :--- | :---: |
| **Exactitud de Intención, Ruta y Territorio (`Intent` / `Route` / `Territorio Accuracy`)** | **100,00%** (`12/12`) |
| **Cobertura de las 3 Capas Normativas (`Mean Layer Coverage`)** | **100,00%** |
| **Recall de Artículos y Tablas Clave (`Citation Recall@8`)** | **100,00%** |
| **Tasa de Cero Alucinación Numérica (`Zero Numerical Hallucination Rate` — Nodo 5)** | **100,00%** (`12/12`) |
| **RAGAS `Faithfulness` (Fidelidad estricta al contexto recuperado)** | **98,57%** |
| **RAGAS `Answer Relevancy` (Relevancia a la consulta resuelta)** | **100,00%** |
| **RAGAS `Context Recall Semántico`** | **97,14%** |
| **RAGAS `Hierarchy Application Score` (Jerarquía Normativa Art. 3.3 ET)** | **100,00%** |

---

## 6. Instalación, Configuración y Ejecución

1. **Activar el entorno virtual (`.venv`) e instalar dependencias:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configurar variables de entorno:**
   Copia `.env.example` a `.env` y define tus claves de API (y opcionalmente el token y repositorio privado de Hugging Face para sincronización automática de logs):
   ```ini
   GOOGLE_API_KEY=AIza...
   COHERE_API_KEY=...
   HF_LOG_TOKEN=hf_...
   HF_LOG_DATASET_REPO=Andregon79/aihoreca-logs
   ```

3. **Comandos de ejecución por módulo:**
   ```powershell
   # Fase 1: Auditar límites de páginas o ejecutar ingesta PDF -> Markdown
   python -m src.ingestion.pdf_ingestor --inspect-only
   python -m src.ingestion.pdf_ingestor

   # Fase 2: Fragmentar Markdowns e indexar los 648 chunks en Qdrant local
   python -m src.retrieval.vector_store --recreate

   # Fase 3 y 3.5: Ejecutar la suite de validación multi-sesión del grafo agéntico (6 turnos en 2 sesiones)
   python -m src.graph.workflow

   # Fase 4: Ejecutar la auditoría del corpus y el benchmark completo del Golden Dataset (12 casos)
   python -m src.evaluation.evaluator

   # Fase 5 (Opción A - 1 Clic en Windows): Doble clic en el lanzador de la raíz o desde terminal:
   .\Iniciar_AiHoreca.bat

   # Fase 5 (Opción B - Comando directo Streamlit):
   streamlit run src/app/ui.py

   # Fase 5 (Opción C - Consola interactiva multi-sesión en terminal CLI REPL o One-Shot):
   python -m src.app.cli --session ses-demo
   python -m src.app.cli --session ses-demo --query "Soy pinche en León, ¿cuál es mi salario base?"
   ```

### 6.1. Escalado de Free Tier a Paid Tier (`src/config.py`)

Tanto los modelos de inferencia como el presupuesto de fragmentos recuperados están centralizados en `src/config.py`. Al pasar de una API Key gratuita a facturación de pago (Pay-as-you-go), basta con ajustar las constantes de `src/config.py` sin tocar la lógica del grafo ni de la base vectorial:

- **Modelos (`GEMINI_LITE_MODEL`, `GEMINI_REASONING_MODEL`):** Sustituir `gemini-3.1-flash-lite-preview` por `gemini-3.5-flash` en el Router (Nodo 1) y usar `gemini-3.5-pro` / `gemini-3.5-flash` en el Sintetizador (Nodo 4).
- **Presupuesto de Contexto RAG (`RETRIEVAL_TOP_K_PER_LAYER`, `RETRIEVAL_FINAL_TOP_K`, `QUOTA_*`):** Ampliar, por ejemplo, de `10` a `16` fragmentos finales (`6` artículos provinciales + `3` tablas salariales + `4` ALEH + `3` Estatuto de los Trabajadores) para consultas jurídicas multi-artículo de alta complejidad.

### 6.2. Publicación Web (Streamlit Community Cloud) y Telemetría Relacional en Hugging Face (`despliegue_web_y_telemetria_hf.md`)

La arquitectura híbrida a coste 0 € combina:
- **Alojamiento Web en Streamlit Community Cloud (`share.streamlit.io` conectado a GitHub):** El archivo [`.gitignore`](file:///d:/Proyectos/AiHoreca/.gitignore) excluye secretos (`.env`), entorno virtual (`.venv/`), sesiones locales (`data/sessions/`), logs locales (`data/logs/`), el bloqueo `.lock` de Qdrant y los PDFs crudos (`data/raw/*.pdf`), pero **conserva en el repositorio `data/processed/` y el índice pre-calculado `data/qdrant_storage/` (~20,5 MB)** para que la aplicación arranque instantáneamente en la nube sin re-indexar.
- **Persistencia Remota Automática en Hugging Face Dataset Privado (`Andregon79/aihoreca-logs`):** En lugar de hacer `git push` al repositorio de GitHub (lo que reiniciaría la aplicación en Streamlit Cloud en mitad de una conversación), el backend sincroniza en segundo plano `conversaciones.txt` (texto plano agrupado por `session_id`), `turnos_metricas.csv` y `aihoreca_sessions.db` hacia el Dataset privado de Hugging Face en `T = 0 ms` (sin depender de que el usuario pulse `👍 / 👎`), restaurando el histórico automáticamente si el contenedor se reinicia. Consulta todos los detalles en [`despliegue_web_y_telemetria_hf.md`](file:///d:/Proyectos/AiHoreca/despliegue_web_y_telemetria_hf.md).

---

## 7. Metodología de Desarrollo (Modular-First)

- **Todo el código core reside en `src/`:** La ingesta, indexación vectorial, memoria multi-sesión y el grafo de `LangGraph` se implementan directamente en módulos `.py` tipados con `Pydantic`.
- **Uso de `notebooks/`:** Reservado exclusivamente para importar los módulos de `src/` y ejecutar evaluaciones cuantitativas (`RAGAS`) o inspecciones de calidad de datos.
