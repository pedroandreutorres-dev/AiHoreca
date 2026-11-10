# AiHoreca — Asistente Legal RAG Agéntico para el Sector de Hostelería

**AiHoreca** es un sistema de **Recuperación Híbrida Orquestada por Grafos de Estado (Agentic Hybrid RAG)** especializado en derecho laboral para el sector de la hostelería y el turismo en España.

El sistema está diseñado para ejecutarse íntegramente en **entornos CPU (sin necesidad de GPU local)** mediante una arquitectura de ingesta híbrida automatizada (`PyMuPDF` + `google-genai`) y agnóstica al boletín oficial, preparada para escalar desde las provincias inicialmente indexadas hacia cualquier convenio colectivo provincial o autonómico sin reglas manuales (*hardcoded*) por documento. Incorpora además **control de regresión continuo con coste 0 de tokens**, **auditoría diaria automatizada** y **política de retención de datos alineada con el RGPD (art. 5.1.e)**.

---

## 1. Jerarquía Normativa del Corpus (3 Capas)

El motor aplica el principio de jerarquía normativa y norma más favorable consultando en paralelo tres niveles legales:

1. **Capa 1 — Mínimo de Derecho Necesario (Estatal):**
   - *Estatuto de los Trabajadores* (BOE).
2. **Capa 2 — Estructura Sectorial y Clasificación Profesional (Estatal):**
   - *Acuerdo Laboral de Ámbito Estatal para el Sector de Hostelería — ALEH* (BOE).
3. **Capa 3 — Condiciones Económicas y Laborales (Provincial / Autonómica):**
   - Convenios colectivos provinciales o autonómicos de hostelería y turismo publicados en los respectivos boletines oficiales (BOCM, DOGC, BOP, etc.), incluyendo soporte automático para correcciones de erratas posteriores y consultas comparativas multi-territoriales.

---

## 2. Stack Tecnológico (100% CPU)

| Capa | Tecnología | Rol en el Sistema |
| :--- | :--- | :--- |
| **Motor PDF y Tablas Vectoriales en CPU** | `PyMuPDF` (`pymupdf` + `pymupdf4llm`) | Escaneo de cabeceras, recorte de páginas en memoria, extracción de texto articulado en CPU y conversión directa de tablas con rejilla vectorial completa a Markdown (`\|...\|`). |
| **Ingesta Multimodal y Tablas Complejas** | `google-genai` SDK + `Pydantic` | Detección automatizada de límites de documento `[start_page, end_page]` (`boundaries_cache.json`), reconstrucción de tablas sin bordes verticales o con celdas combinadas y aplicación automática de correcciones de erratas (`ErrataPatchList`). |
| **Base de Datos Vectorial** | `Qdrant` (Modo local `data/qdrant_storage/`) | Colección vectorial híbrida con búsqueda paralela aislada por capa normativa, estratificación intra-provincial (`articulo`, `tabla_salarial`, `anexo_o_disposicion`) y soporte multi-provincia simultáneo (`MatchAny`). |
| **Embeddings Densos** | Google `gemini-embedding-001` | Representación semántica multilingüe (`Cosine`), indexada por lotes controlados por límite de tokens por minuto. |
| **Embeddings Dispersos** | `FastEmbed` (`Qdrant/bm25` con `IDF`) | Búsqueda léxica en CPU para referencias normativas exactas, grupos profesionales y cuantías económicas. |
| **Reranker y Selección Estratificada** | Cohere `rerank-multilingual-v3.0` + Cuotas Jerárquicas | Reordenación *cross-encoder* con priorización automática de tablas salariales vigentes, cobertura multi-intención retributiva y cuotas mínimas post-rerank por capa normativa parametrizadas en [`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py). |
| **Orquestación Agéntica** | `LangGraph` + `Pydantic v2` | Máquina de estados con descubrimiento territorial dinámico, enrutamiento multi-provincia, guardarraíles de entrada/salida ([`AuditReport`](file:///d:/Proyectos/AiHoreca/src/graph/state.py#L38-L50)), validación combinatoria de subconjuntos aritméticos (*subset-sum*), solicitud de aclaración territorial (HITL) y traducción de oficios coloquiales al ALEH. |
| **Memoria Conversacional y RGPD** | `SQLite` (`sqlite3` WAL en `data/sessions/`) + Exportación Relacional (`data/logs/`) | Persistencia transaccional aislada por `session_id` con separación entre *Working Memory* ([`SessionContextSlots`](file:///d:/Proyectos/AiHoreca/src/graph/state.py#L18-L35)) y *Episodic Memory* (`semantic_digest`), prevalencia temporal del turno actual y **purga automática de sesiones inactivas a los 90 días** (art. 5.1.e RGPD). |
| **Modelos LLM** | Google AI Studio (`google-genai`) | Parametrizados centralmente en [`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py) con soporte de rotación automática (*failover*) ante límites de cuota o latencia. |
| **Evaluación Continua y Regresión** | Suite de Regresión Determinista + Juez `RAGAS` ([`src/evaluation/`](file:///d:/Proyectos/AiHoreca/src/evaluation/)) | Suite de control de calidad continua ([`regression.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/regression.py)) con modo rápido (0 tokens del juez, <10s) contra una línea base congelada ([`evaluation_baseline.json`](file:///d:/Proyectos/AiHoreca/data/processed/evaluation_baseline.json)), y Golden Dataset extensible desde producción ([`golden_set_produccion.jsonl`](file:///d:/Proyectos/AiHoreca/data/processed/golden_set_produccion.jsonl)). |
| **Auditoría Diaria y Sincronización Remota** | Runner de Auditoría ([`src/app/audit_runner.py`](file:///d:/Proyectos/AiHoreca/src/app/audit_runner.py)) + Hugging Face Hub | Sincronización bidireccional de telemetría (`telemetry_sync.py`), pipeline de auditoría automática con sidecar programado diario (08:00 Europe/Madrid), histórico permanente de KPIs y compactación periódica Git (`super_squash_history`). |
| **Capa de Servicio e Interfaz (`src/app/`)** | `Streamlit` + Consola REPL CLI ([`AiHorecaEngineService`](file:///d:/Proyectos/AiHoreca/src/app/service.py#L25-L100)) | Singleton *thread-safe* con recarga automática ante cambios en el código (`compute_code_version`), aislamiento de sesiones por visitante web, inspector de trazabilidad por turno y panel de auditoría del corpus. |

---

## 3. Arquitectura del Sistema

### 3.1. Pipeline de Ingesta Híbrido en 3 Niveles e Indexación Vectorial

```text
[ PDFs Crudos en /data/raw/ ]
              │
              ├──► 1. Detector de Límites de Página (PyMuPDF + google-genai + Pydantic)
              │       • Filtra cabeceras de boletín en CPU y devuelve DocumentBoundary(start_page, end_page).
              │       • Persiste en boundaries_cache.json y recorta exclusivamente el convenio aplicable.
              │
              ├──► 2. Extracción Híbrida en 3 Niveles a Markdown (CPU + Gemini Multimodal)
              │       • Nivel 1 (Texto articulado): extracción directa en CPU con pymupdf (0 llamadas API).
              │       • Nivel 2 (Tablas con rejilla vectorial completa): conversión en CPU con pymupdf4llm.
              │       • Nivel 3 (Tablas sin bordes verticales / rowspans): procesamiento multimodal por lotes PDF
              │         (temperature=0.0) para reconstruir tablas Markdown estructuradas (|...|).
              │
              ├──► 3. Auto-Errata Patcher (PyMuPDF + google-genai + Pydantic)
              │       • Extrae ErrataPatchList y aplica/verifica las correcciones sobre el convenio matriz.
              │       ▼
        [ Documentos Markdown Limpios en /data/processed/ ]
              │
              └──► 4. Chunking Jerárquico e Indexación Híbrida (src/retrieval/)
                      • Fragmentación por Capítulo -> Artículo / Disposición / Tabla Salarial.
                      • Inyección de cabecera contextual [NORMA | CAPA | TERRITORIO | SECCIÓN | REFERENCIA].
                      • Indexación híbrida en Qdrant (data/qdrant_storage/): Embeddings densos + BM25 con IDF.
                      • Soporte de filtrado multi-provincia con models.MatchAny para consultas comparativas.
```

### 3.2. Flujo del Agente y Memoria Multi-Sesión ([`src/graph/`](file:///d:/Proyectos/AiHoreca/src/graph/))

```text
                               ┌────────────────────────────────┐
                               │   Entrada de Usuario / Turno   │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   1. Carga Contexto SQLite     │
                               │  (Working Memory + Digest)     │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   2. analyze_and_route_node    │
                               │  Prevalencia Turno > Sesión    │
                               │  Multi-Provincia / Guardrail   │
                               └───────┬───────────────┬────────┘
                                       │               │
                     [Fuera de Ámbito] │               │ [Requiere Aclaración HITL]
                                       ▼               ▼
                       ┌───────────────────┐       ┌───────────────────────┐
                       │ Cortocircuito OOS │       │ ask_clarification     │
                       │ (0 tokens gen.)   │       │ (Pide provincia HITL) │
                       └───────────────────┘       └───────────────────────┘
                                       │               │
                                       └───────┬───────┘
                                               │ [Consulta Válida: 1 o N provincias]
                                               ▼
                               ┌────────────────────────────────┐
                               │   3. parallel_retrieval_node   │
                               │  Qdrant Híbrido (MatchAny)     │
                               │  Capa 1 (ET) + Capa 2 (ALEH)   │
                               │  Capa 3 (Convenio Provincial)  │
                               │  Cohere Rerank + Cuotas        │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │ 4. legal_synthesis_and_audit   │
                               │ • Atribución Jurisdiccional    │
                               │   Estricta (Sin analogía)      │
                               │ • Tablas comparativas claras   │
                               │ • Auditor Aritmético CPU       │
                               │   (Subset-Sum combinatorio)    │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   5. Persistencia y Telemetría │
                               │  SQLite WAL + CSV + TXT        │
                               │  Purga 90d RGPD si corresponde │
                               └────────────────────────────────┘
```

1. **Carga de Contexto por `session_id` ([`SessionMemoryStore`](file:///d:/Proyectos/AiHoreca/src/graph/memory.py#L30-L150)):** Recupera de SQLite los [`SessionContextSlots`](file:///d:/Proyectos/AiHoreca/src/graph/state.py#L18-L35) vigentes (`active_territorio`, `active_oficio_coloquial`, `active_categoria_aleh`, `pending_clarification_query`) y la ventana episódica con el resumen semántico (`semantic_digest`).
2. **Nodo Unificado de Análisis y Enrutamiento ([`analyze_and_route_node`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py#L68-L150)):**
   - **Prevalencia Temporal del Turno Actual:** Si el mensaje del usuario introduce explícitamente una nueva provincia, oficio o contexto, tiene precedencia absoluta sobre los *slots* heredados del turno previo.
   - **Enrutamiento Multi-Provincia:** Detecta consultas comparativas entre territorios soportados (ej. `madrid, leon`) sin recurrir a diccionarios *hardcoded*, delegando la normalización en la LLM estructurada y admitiendo múltiples provincias simultáneas.
   - **Correferencia Dinámica:** Resuelve acumulaciones de turnos anteriores sin duplicar preguntas ya contestadas.
   - **Guardrail de Dominio y Traducción al ALEH:** Separa consultas del ámbito (`HOSPITALITY_CORE`), provincias no indexadas (`UNSUPPORTED_PROVINCE`) o materias ajenas (`OUT_OF_SCOPE`), mapeando denominaciones coloquiales al Área y Grupo oficial del ALEH.
3. **Nodos de Cortocircuito Determinista ([`block_out_of_scope_node`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py#L152-L170) / [`ask_clarification_node`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py#L172-L195)):** Rechazan materias no laborales o solicitan la provincia faltante con coste 0 de tokens de generación.
4. **Nodo de Recuperación Paralela ([`parallel_retrieval_node`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py#L197-L270)):**
   - Ejecuta búsquedas simultáneas en Qdrant para el Estatuto de los Trabajadores, el ALEH y los Convenios Provinciales implicados mediante filtros `models.MatchAny(any=[...])`.
   - Aplica Cohere Rerank estratificado garantizando las cuotas mínimas por capa parametrizadas en [`src/config.py`](file:///d:/Proyectos/AiHoreca/src/config.py).
5. **Nodo de Síntesis y Auditoría Numérica ([`legal_synthesis_and_audit_node`](file:///d:/Proyectos/AiHoreca/src/graph/nodes.py#L272-L420)):**
   - **Principio de Atribución Jurisdiccional Estricta:** Regla generalista que prohíbe aplicar convenios provinciales fuera de su ámbito territorial. Permite responder comparativas multi-territoriales estructurándolas obligatoriamente bajo epígrafes separados y prohíbe expresamente mezclar cuantías o aplicar normas de una provincia a otra por analogía.
   - **Salarios Brutos sin Suposiciones Paramétricas:** Presenta las cifras en importes brutos oficiales según convenio, advirtiendo sobre las deducciones legales aplicables sin inventar porcentajes fijos no fundamentados en los textos normativos.
   - **Auditor Aritmético Generalista en CPU:** Valida todas las cifras literales (`€`, `%`) contra los fragmentos recuperados y verifica las sumas retributivas complejas mediante un algoritmo generalista de combinatoria de subconjuntos (*subset-sum validation*: $T = \text{Base} + \sum C_i$), soportando tablas salariales con $k$ complementos arbitrarios sin necesidad de parches manuales por concepto.

---

## 4. Estructura del Proyecto

```text
AiHoreca/
├── data/
│   ├── raw/                                  # Boletines Oficiales y textos consolidados en PDF
│   ├── processed/
│   │   ├── *.md                              # Convenios normalizados a Markdown limpio
│   │   ├── boundaries_cache.json             # Caché de límites de página por documento
│   │   ├── golden_set_produccion.jsonl       # Casos reales promovidos desde producción
│   │   └── evaluation_baseline.json          # Línea base congelada para tests de regresión
│   ├── qdrant_storage/                       # Colección vectorial híbrida local (Qdrant)
│   ├── sessions/                             # Base de datos SQLite multi-sesión (con purga a 90 días)
│   └── logs/
│       ├── conversaciones.txt                # Transcripciones en texto plano por sesión
│       ├── turnos_metricas.csv               # Registro relacional turno a turno con votos OK/KO
│       ├── historico_metricas.csv            # Histórico diario permanente de KPIs de producción
│       ├── candidatos_golden.jsonl           # Cola de turnos marcados para revisión técnica
│       └── reportes_auditoria/               # Informes diarios generados por el audit runner
├── notebooks/
│   └── 01_evaluacion_ragas_y_auditoria.ipynb # Notebook interactivo de auditoría y benchmark
├── src/
│   ├── config.py                             # Configuración centralizada, rutas, retención RGPD y cuotas
│   ├── ingestion/
│   │   ├── schemas.py                        # Contratos Pydantic (DocumentBoundary, ErrataPatchList)
│   │   └── pdf_ingestor.py                   # Motor de ingesta híbrida en 3 niveles y Auto-Errata Patcher
│   ├── retrieval/
│   │   ├── schemas.py                        # Contratos Pydantic (LegalChunk, RetrievedDocument)
│   │   ├── chunker.py                        # Chunker jurídico jerárquico agnóstico a la provincia
│   │   └── vector_store.py                   # Qdrant híbrido (Dense + BM25) con MatchAny + Cohere Rerank
│   ├── graph/
│   │   ├── state.py                          # Estado LegalGraphState y esquemas Pydantic del grafo
│   │   ├── memory.py                         # Gestor SQLite de sesiones, slots, purga 90d y telemetría
│   │   ├── nodes.py                          # Nodos del grafo, precedencia temporal, failover y Guardrail Numérico
│   │   └── workflow.py                       # Compilación del StateGraph y suite de validación multi-turno
│   ├── evaluation/
│   │   ├── golden_dataset.py                 # Golden Dataset unificado (14 casos: core + producción)
│   │   ├── evaluator.py                      # Evaluador determinista + Juez Semántico RAGAS single-pass
│   │   └── regression.py                     # Suite de regresión continua contra baseline oficial
│   └── app/
│       ├── service.py                        # Servicio singleton thread-safe AiHorecaEngineService
│       ├── telemetry_sync.py                 # Sincronizador de telemetría bidireccional (pull/push HF Hub)
│       ├── audit_runner.py                   # Runner de auditoría diaria, KPIs, squash Git y cola Golden Set
│       ├── ui.py                             # Interfaz web en Streamlit con inspector RAG y votos OK/KO
│       └── cli.py                            # Consola interactiva REPL y modo de consulta por terminal
├── .env.example                              # Plantilla de variables de entorno
├── .gitignore                                # Reglas de exclusión para control de versiones
├── Iniciar_AiHoreca.bat                      # Lanzador rápido opcional para entornos Windows
├── requirements.txt                          # Dependencias del proyecto
└── README.md
```

---

## 5. Control de Calidad Continua y Regresión (`src/evaluation/`)

El sistema cuenta con un marco formal de garantía de calidad y regresión ([`src/evaluation/regression.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/regression.py)) que valida cualquier modificación en los prompts, chunkers o lógica del grafo antes de su despliegue a producción.

### 5.1. Golden Dataset Unificado (14 Casos Auditados)

El conjunto de prueba ([`src/evaluation/golden_dataset.py`](file:///d:/Proyectos/AiHoreca/src/evaluation/golden_dataset.py)) combina los 12 casos base de diseño con casos reales promovidos desde producción ([`data/processed/golden_set_produccion.jsonl`](file:///d:/Proyectos/AiHoreca/data/processed/golden_set_produccion.jsonl)):

| ID del Caso | Categoría | Consulta Clave | Comportamiento Verificado |
| :--- | :--- | :--- | :--- |
| `GD_01_OOS_TRAFICO` | `guardrail_routing` | Indemnización por accidente de tráfico | Bloqueo determinista `block_out_of_scope` (0 tokens). |
| `GD_02_OOS_OTRO_SECTOR` | `guardrail_routing` | Convenio de consultoría informática | Bloqueo determinista `block_out_of_scope` (0 tokens). |
| `GD_03_UNSUPPORTED_SEVILLA` | `guardrail_routing` | Tablas de hostelería de Sevilla | Aviso pedagógico `UNSUPPORTED_PROVINCE`. |
| `GD_04_HITL_FRIEGAPLATOS_SIN_PROVINCIA` | `guardrail_routing` | Friegaplatos sin indicar provincia | Gatekeeper solicita provincia (`ask_clarification`). |
| `GD_05_MADRID_HITL_RESOLUCION` | `provincial_rag` | Resolución indicando "Madrid" | Recuperación 3 capas y tablas salariales de Madrid. |
| `GD_06_MADRID_SEGUIMIENTO_NOCTURNIDAD` | `provincial_rag` | Turno nocturno en Madrid | Art. 27 Convenio Madrid y porcentaje legal de recargo. |
| `GD_07_LEON_PINCHE_SALARIO` | `provincial_rag` | Pinche de cocina en León | Nivel Salarial V, ALEH Área 2ª y tablas de León. |
| `GD_08_LEON_SWITCH_VALENCIA_ANTI_POISONING` | `multi_turn_memory` | Cambio de local a Valencia en turno 3 | Prevalencia del turno actual y protección anti-poisoning. |
| `GD_09_LEON_RETORNO_VACACIONES_Y_PERMISOS` | `multi_turn_memory` | Retorno a León: vacaciones y festivos | Retención de contexto episódico y Art. 12/25 de León. |
| `GD_10_BARCELONA_RIDER_Y_FALTA_MUY_GRAVE` | `provincial_rag` | Repartidor en Barcelona / falta muy grave | ALEH Anexo I, Art. 40-41 y normativa catalana. |
| `GD_11_GIRONA_CAMARERA_PISOS` | `provincial_rag` | Camarera de pisos en Girona | Mapeo interprovincial de Cataluña a Girona. |
| `GD_12_ESTATAL_ALEH_AREAS_FUNCIONALES` | `provincial_rag` | 6 Áreas Funcionales del VI ALEH | Consulta estatal pura resuelta sin forzar provincia. |
| `GD_13_DESAMBIGUACION_LOCAL_PONFERRADA` | `guardrail_routing` | Bar "Café Madrid" en Ponferrada | Desambiguación semántica correcta a León (no Madrid). |
| `GD_14_COMPARATIVA_MADRID_LEON` | `provincial_rag` | Colmenar Viejo vs Camponaraya | Comparativa multi-provincia con `MatchAny` y epígrafes separados. |

### 5.2. Ejecución de la Suite de Regresión

```powershell
# 1. Modo Rápido Determinista (0 tokens de juez, ~5-10s):
# Valida los 14 casos contra la línea base oficial (evaluation_baseline.json)
python -m src.evaluation.regression --quick

# 2. Probar un caso individual durante la fase de desarrollo o depuración:
python -m src.evaluation.regression --quick --case-id GD_14_COMPARATIVA_MADRID_LEON

# 3. Forzar recomputación del grafo omitiendo la caché local:
python -m src.evaluation.regression --quick --force

# 4. Modo Completo con Juez Semántico RAGAS (evalúa faithfulness, relevancy y hierarchy):
python -m src.evaluation.regression --force

# 5. Actualizar la línea base oficial tras validar mejoras aprobadas:
python -m src.evaluation.regression --quick --update-baseline
```

#### Opciones de la CLI de Regresión (`src.evaluation.regression`)

| Parámetro | Tipo | Descripción |
| :--- | :--- | :--- |
| `--quick` | Flag | Modo rápido determinista en CPU (0 tokens de juez, ~5s). Evalúa enrutamiento, provincias y 0 alucinaciones. |
| `--force` | Flag | Fuerza la reejecución completa del grafo ignorando la caché local de turnos. |
| `--case-id <ID>` | String | Filtra la ejecución a un único caso de prueba específico (ej. `GD_14_COMPARATIVA_MADRID_LEON`). |
| `--update-baseline` | Flag | Sobrescribe la línea base oficial (`evaluation_baseline.json`) con las métricas de la corrida actual. |
| `--json` | Flag | Emite el reporte de regresión en formato JSON para integración con pipelines CI/CD. |

**Códigos de Salida (Exit Codes para CI/CD):**
- `0`: Todos los casos de prueba pasan satisfactoriamente sin regresión respecto a la línea base oficial.
- `1`: Se detecta al menos una regresión (un caso que previamente pasaba ahora falla).

---

## 6. Auditoría Diaria Automatizada y RGPD (`src/app/`)

### 6.1. Pipeline de Auditoría Diaria (`src/app/audit_runner.py`)

El sistema incluye un ejecutor automatizado que procesa diariamente la telemetría acumulada:

```text
[ HF Dataset Remoto ] ──(pull)──► [ Descarga Telemetría ]
                                          │
                                          ▼
                               [ Detección Turnos Nuevos ] ──(Cursor incremental)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         [ Auditoría de Calidad ]                   [ Purga RGPD (90 días) ]
         • Tasa de satisfacción OK/KO               • Purga sesiones locales SQLite
         • Detección de votos tardíos               • Purga CSVs / TXTs / reportes
         • Detección alucinaciones numéricas        • Compactación Git HF Dataset
         • Detección incidencias territoriales        (super_squash_history)
                    │
                    ▼
         [ Salida de Resultados ]
         • Reporte Markdown en data/logs/reportes_auditoria/
         • Actualización de data/logs/historico_metricas.csv
         • Encolado en data/logs/candidatos_golden.jsonl
```

- **Ejecución Incremental Inteligente:** Si no existen turnos nuevos respecto a la última ejecución, finaliza en menos de 1 segundo con coste 0.
- **Detección de Votos Tardíos:** Si un usuario vota `OK` o `KO` minutos después de que finalizara la consulta, el auditor detecta la discrepancia vía `last_feedback_at` y actualiza los indicadores.
- **Cola de Promoción al Golden Set (`candidatos_golden.jsonl`):** Todo turno con *feedback* negativo (`KO`), discrepancia territorial o alerta numérica se añade automáticamente a una cola técnica para su examen e incorporación como test de regresión.

```powershell
# Ejecución manual de la auditoría diaria:
python -m src.app.audit_runner

# Salida en JSON para pipelines automáticos o webhooks:
python -m src.app.audit_runner --json

# Forzar descarga desde Hugging Face y reauditoría completa:
python -m src.app.audit_runner --force-pull --force-audit-all
```

> **Automatización Programada:** En entornos locales o servidores de trabajo, este proceso se ejecuta automáticamente cada mañana a las **08:00 (Europe/Madrid)** mediante el sidecar programado `daily-telemetry-audit` (`CRON_TZ=Europe/Madrid 0 8 * * *`).

### 6.2. Cumplimiento del RGPD (Art. 5.1.e — Limitación de Plazo de Conservación)

De conformidad con el principio de limitación del plazo de conservación del Reglamento General de Protección de Datos:

1. **Purga en Origen (SQLite):** El método [`SessionMemoryStore.purge_inactive_sessions(retention_days=90)`](file:///d:/Proyectos/AiHoreca/src/graph/memory.py) elimina automáticamente sesiones, mensajes y *slots* inactivos con más de 90 días de antigüedad. Se ejecuta en cada ciclo de auditoría y periódicamente en el inicio del servicio [`AiHorecaEngineService`](file:///d:/Proyectos/AiHoreca/src/app/service.py).
2. **Purga en Archivos Locales:** Los archivos de transcripciones (`conversaciones.txt`), informes diarios y registros históricos antiguos son limpiados automáticamente para evitar almacenamiento indefinido de datos personales.
3. **Compactación Git en Hugging Face (`HfApi.super_squash_history`):** Para evitar que los registros purgados persistan en el historial de *commits* de Git del repositorio remoto de datasets, el sistema ejecuta periódicamente (cada 30 días) una compactación profunda del árbol Git.

---

## 7. Instalación, Configuración y Uso

### 7.1. Instalación del Entorno

1. **Crear y activar el entorno virtual:**
   ```powershell
   # En Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # En Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno:**
   Copia `.env.example` a `.env` y configura las credenciales necesarias:
   ```ini
   GOOGLE_API_KEY=AIza...
   COHERE_API_KEY=...
   # Opcionales para telemetría y sincronización remota:
   HF_LOG_TOKEN=hf_...
   HF_LOG_DATASET_REPO=tu-usuario/aihoreca-telemetria
   ```

### 7.2. Pipeline de Puesta en Marcha (Paso a Paso)

```powershell
# Paso 1: Ingesta de convenios (PDF -> Markdown limpio con extracción de tablas)
python -m src.ingestion.pdf_ingestor

# Paso 2: Chunking jerárquico e indexación híbrida en Qdrant local
python -m src.retrieval.vector_store --recreate

# Paso 3: Verificación de calidad mediante la suite de regresión continua
python -m src.evaluation.regression --quick

# Paso 4 (Opción A): Iniciar la Interfaz Web interactiva (Streamlit)
streamlit run src/app/ui.py
# (En Windows también puedes hacer doble clic en Iniciar_AiHoreca.bat)

# Paso 4 (Opción B): Ejecutar en modo terminal interactivo (CLI REPL)
python -m src.app.cli --session ses-demo

# Paso 4 (Opción C): Realizar una consulta directa puntual por terminal
python -m src.app.cli --session ses-demo --query "¿Cuál es el salario de un camarero en León?"
```

---

## 8. Metodología de Desarrollo (Modular-First y Soluciones Generalistas)

- **Cero Parches Ad-hoc (*No Quick Fixes*):** El sistema prohíbe explícitamente introducir diccionarios *hardcoded* de municipios o excepciones específicas por caso de prueba. Las desambiguaciones geográficas y las correferencias son resueltas por el modelo de lenguaje en el Gatekeeper, y los cálculos retributivos se auditan mediante formulaciones algebraicas combinatorias generalistas.
- **Todo el código reside en `src/`:** La ingesta documental, la indexación vectorial, la orquestación en grafos, la memoria conversacional y los runners de calidad están implementados en módulos Python tipados con contratos `Pydantic v2`.
- **Trazabilidad y Reproducibilidad:** Cualquier mejora en la base de conocimiento o en el comportamiento del agente debe quedar validada por la suite de regresión contra la línea base antes de ser integrada en la rama principal.
