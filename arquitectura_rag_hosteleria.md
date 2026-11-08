# Diseño de Arquitectura: Asistente Legal RAG Agéntico para el Sector de Hostelería (Proyecto: AiHoreca)

## 1. Definición del Alcance y Caso de Negocio (MVP Escalable a 50 Provincias)

El sistema es un asistente jurídico-laboral especializado en el sector de la **Hostelería en España**, diseñado bajo una arquitectura **Agentic Hybrid RAG** (Recuperación Híbrida Orquestada por Grafos de Estado) con un **pipeline de ingesta híbrido 100% automatizado (`PyMuPDF` + `google-genai`) y agnóstico a la provincia**.

### 1.1. Corpus Documental (Jerarquía Normativa de 3 Capas)

Para demostrar escalabilidad territorial hacia las 50 provincias españolas sin reglas manuales (*hardcoded*) por documento, el MVP valida tres provincias con estructuras de publicación heterogéneas:

1. **Capa 1 - Mínimo de Derecho Necesario (Estatal):**
   * *Real Decreto Legislativo 2/2015, por el que se aprueba el texto refundido de la Ley del Estatuto de los Trabajadores (BOE).*

2. **Capa 2 - Estructura Sectorial y Clasificación Profesional (Estatal):**
   * *VI Acuerdo Laboral de Ámbito Estatal para el Sector de Hostelería - ALEH (BOE).* Regula grupos profesionales, áreas funcionales, régimen disciplinario y subrogación.

3. **Capa 3 - Condiciones Económicas y de Trabajo (Provincial / Autonómica):**
   * **Madrid:** *Convenio Colectivo del Sector de Hostelería y Actividades Turísticas de la Comunidad de Madrid (BOCM).*
   * **Barcelona / Cataluña:** *Convenio Colectivo Interprovincial del Sector de la Industria de Hostelería y Turismo de Cataluña (DOGC) + Corrección de Erratas oficial.* *(Reto técnico: resolución automática de parches de erratas publicadas en boletines posteriores sin intervención manual).*
   * **León:** *Convenio Colectivo Provincial de Hostelería y Turismo de León (BOP de León).* *(Reto técnico: boletín provincial completo con decenas de páginas de anuncios municipales de Ayuntamientos, Juntas Vecinales y otros ministerios mezclados con el convenio).*

---

## 2. Stack Tecnológico (Restricción: Ejecución sin GPU / Coste 0€)

| Componente | Tecnología Seleccionada | Justificación de Ingeniería |
| :--- | :--- | :--- |
| **Motor PDF en CPU y Tablas Vectoriales** | **`PyMuPDF` (`pymupdf` + `pymupdf4llm`)** | Escrito en C; escanea cabeceras en milisegundos, recorta páginas en memoria, extrae texto articulado en CPU y convierte directamente a Markdown las tablas con rejilla vectorial completa (`col_count >= 3` sin celdas colapsadas) a coste 0€. |
| **Ingesta Multimodal de Tablas sin Bordes y Límites** | **`google-genai` SDK** (`gemini-3.5-flash` + cadena de fallback Free Tier) | Procesamiento nativo de bytes PDF (`types.Part.from_bytes(..., mime_type="application/pdf")`) con `temperature=0.0` exclusivamente para tablas sin bordes verticales o con *rowspans* complejos, detección de `[start_page, end_page]` (cacheada en `boundaries_cache.json`) y aplicación de erratas (`Pydantic`). |
| **Base de Datos Vectorial** | **`Qdrant`** (Local en disco `data/qdrant_storage/`) | Colección híbrida `aihoreca_legal_corpus` (`648` chunks) con búsqueda paralela aislada por las 3 capas normativas y filtrado estricto por *payloads* (`capa_normativa`, `provincias_aplicables`, `tipo_contenido`). |
| **Embeddings Densos** | **Google `gemini-embedding-001`** (API Free Tier) | Vectores semánticos multilingües de `3072` dimensiones (`Cosine`), indexados de forma incremental en lotes acotados (`<= 55.000` caracteres) para respetar la ventana de `30.000 TPM` del Free Tier. |
| **Embeddings Dispersos (Léxico)** | **FastEmbed (`Qdrant/bm25` con `IDF`)** (Local CPU) | Ejecución en milisegundos por CPU para capturar coincidencias exactas de códigos legales (*"Artículo 14"*, *"Grupo Profesional 2.º"*, *"1.731,57 €"*). |
| **Reranker (Cross-Encoder + Cuota Jerárquica)** | **Cohere Rerank API** (`rerank-multilingual-v3.0`) + Fallback Calibrado (`numpy`) | Reordena en una escala semántica común `[0, 1]` los candidatos de las 3 capas normativas; incorpora selección estratificada post-rerank para impedir la monopolización geográfica y respaldo en CPU (`0.75 * Cosine + 0.25 * RRF`). |
| **Orquestación y Estado** | **LangGraph** (`langgraph`) | Máquina de estados finitos (`StateGraph`) con enrutamiento condicional, descubrimiento territorial dinámico y guardarraíles de entrada/salida. |
| **Memoria Conversacional Multi-Sesión** | **SQLite (`sqlite3` WAL en CPU)** (`data/sessions/aihoreca_sessions.db`) | Almacén transaccional (`SessionMemoryStore`) con aislamiento por `session_id`, separación entre *Working Memory* (`SessionContextSlots`) y *Episodic Memory* (ventana compacta de 4 turnos), resolución de correferencias HITL y blindaje contra *State Poisoning*. |
| **Modelos de Lenguaje (LLM - Free Tier)** | **Google AI Studio (`google-genai`)** | `gemini-3.1-flash-lite` para el nodo unificado de Guardrail + Gatekeeper + Correferencia + Traductor ALEH (ahorro del 66% en cuota RPD/RPM); `gemini-3.5-flash` (con rotación automática y *cooldown* de 90s hacia `gemini-3.6-flash` / `gemini-3.5-flash-lite` / `gemini-3.1-flash-lite`) para síntesis jurídica. |
| **Validación y Contratos** | **Pydantic v2** | Garantiza respuestas tipadas (JSON Schema) en pre-procesado de documentos, enrutamiento, slots de memoria, síntesis jurídica, auditoría numérica (`AuditReport`) y juez de evaluación (`RAGASJudgeSchema`). |
| **Evaluación Empírica y Auditoría** | **Motor Híbrido Determinista + Juez Estructurado RAGAS (`src/evaluation/`)** | Evalúa los 12 casos del *Golden Dataset* (`guardrail_routing`, `provincial_rag`, `multi_turn_memory`) en 1 única llamada estructurada por caso generativo (evitando el *rate-limit* `429` de las 15 llamadas asíncronas de `ragas` estándar en Free Tier), persistiendo en `data/processed/evaluation_results.json` y `notebooks/01_evaluacion_ragas_y_auditoria.ipynb`. |
| **Capa de Servicio e Interfaz (`src/app/`)** | **`AiHorecaEngineService` + `Streamlit` (`ui.py`) + `CLI` (`cli.py`)** | Singleton *thread-safe* con `threading.Lock()` que previene colisiones de *File Lock* en Qdrant Local, expone gestión multi-sesión en SQLite, inspector de Rayos-X por turno y panel de auditoría del benchmark. |

---

### 2.1. Arquitectura del Pipeline de Ingesta Híbrido Automatizado (`PyMuPDF` + `google-genai`)

Queda **prohibido** escribir expresiones regulares o condicionales específicos por provincia (`if "leon"`, `if "catalunya"`). El pipeline de ingesta en `src/ingestion/` opera mediante tres etapas genéricas aplicables a cualquier Boletín Oficial de España:

```text
[ PDFs Crudos en /data/raw/ ]
              │
              ├──► 1. Detector de Límites y Recorte en CPU (PyMuPDF + google-genai + Pydantic)
              │       • PyMuPDF (`pymupdf`) extrae un índice ligero de cabeceras/pies por página en CPU (<0,1 s)
              │         filtrando previamente las cabeceras repetitivas del boletín (`_BULLETIN_HEADER_NOISE`).
              │       • `google-genai` devuelve un esquema `DocumentBoundary(doc_type, territorio, start_page, end_page)`
              │         que se persiste en `data/processed/boundaries_cache.json`.
              │       • PyMuPDF recorta el PDF en memoria conservando únicamente el rango `[start_page, end_page]`
              │         (eliminando sumarios, índices de puntos, anuncios de Ayuntamientos y edictos ajenos).
              │
              ├──► 2. Extracción Híbrida en 3 Niveles a Markdown (PyMuPDF / pymupdf4llm CPU + google-genai Multimodal)
              │       • Nivel 1 — Página de texto articulado (`not _page_has_complex_table(page)`): PyMuPDF extrae
              │         el texto estructurado directamente en CPU (0 llamadas a API, fidelidad literal 100%).
              │       • Nivel 2 — Página con tabla de rejilla vectorial completa (`_has_clean_vector_table_grid(page)`):
              │         `pymupdf4llm` convierte la tabla a Markdown pipe (`|...|`) en CPU (<0,1 s, 0 llamadas a API).
              │       • Nivel 3 — Página con tabla sin bordes verticales o celdas colapsadas: se envía en lotes de 3 págs.
              │         vía `types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")` a la cadena `Flash`
              │         (`temperature=0.0`, con respaldo por layout de texto ante timeouts `504`).
              │
              └──► 3. Auto-Errata Patcher (PyMuPDF + google-genai + Pydantic)
                      • Si un documento fue clasificado como `correccion_erratas`:
                      • Extrae su texto con PyMuPDF y solicita a `google-genai` un esquema `ErrataPatchList`
                        con objetos `ErrataPatch(contexto_articulo, donde_dice, debe_decir)`.
                      • Aplica dinámicamente las sustituciones (o verifica que el texto corregido ya consta en la
                        edición en castellano) sobre el Markdown del convenio matriz.
                      ▼
        [ 5 Markdowns Limpios en /data/processed/ (~944 KB, 1.659 filas de tabla |...|) ]
```

---

### 2.2. Arquitectura de Chunking Jurídico e Indexación Híbrida (`src/retrieval/`)

El módulo `src/retrieval/` transforma los 5 documentos Markdown en una base vectorial híbrida de **648 fragmentos (`LegalChunk`)** almacenada en `data/qdrant_storage/`:

1. **Chunker Jerárquico Agnóstico a la Provincia (`src/retrieval/chunker.py`):**
   - Fragmenta por unidad lógica (`TÍTULO`/`CAPÍTULO`/`ANEXO` → `Artículo`/`Disposición`/`Tabla Salarial`), descartando falsos positivos de menciones a leyes externas en mitad de una frase.
   - Si una tabla salarial extensa supera los `2.600` caracteres, repite automáticamente la cabecera de columnas (`|...|` y `|---|---|`) en cada sub-fragmento.
   - Inyecta al inicio de cada fragmento su cabecera contextual completa:
     `[NORMA: ... | CAPA: ... | TERRITORIO: ... | SECCIÓN: ... | REFERENCIA: ...]`.
2. **Indexación Incremental Acotada por Tokens (`src/retrieval/vector_store.py`):**
   - Combina vectores densos de `gemini-embedding-001` (`3072` dims, agrupados en lotes `<= 55.000` caracteres para no superar el techo de `30.000 TPM` del Free Tier) con vectores dispersos léxicos `FastEmbed` (`Qdrant/bm25` con modificador `IDF`) en CPU.
3. **Recuperación Paralela Multi-Capa Estratificada + Reranking (`parallel_hierarchical_retrieve`):**
   - Ejecuta búsquedas híbridas (`Prefetch Dense + Prefetch BM25 -> RRF`) reutilizando un único vector de consulta (`d_vec` + `s_vec` en memoria, 1 sola llamada API de embedding) sobre las 3 capas normativas (`estatuto_trabajadores`, `aleh_estatal` y `convenio_provincial`).
   - **Estratificación Intra-Provincial (`articulo` + `tabla_salarial` + `anexo_o_disposicion`) y Cobertura Multi-Intención Retributiva:** Dentro de `convenio_provincial`, recupera por separado el mejor articulado regulador (`top_k=6`), las mejores tablas salariales y las disposiciones adicionales (coste 0 API adicional, `<3 ms` en Qdrant local). Cuando detecta consultas salariales o de horas extras, garantiza que entren simultáneamente en los candidatos tanto el artículo de *Gratificaciones Extraordinarias / Pagas Extras* como el de *Horas Extraordinarias / Nocturnidad*.
   - **Priorización Automática de Tablas Salariales Vigentes (`_extract_chunk_year`):** Si el usuario no solicita un año histórico explícito (ej. *"en 2023"*), prioriza las tablas salariales del año más reciente del convenio (limpiando previamente rangos de cabecera como `2023-2025` para identificar el año exacto de cada sub-tabla).
   - **Mitigación de Monopolización Geográfica (*Cross-Encoder Geographical Dominance*):** Tras reordenar todos los candidatos con **Cohere `rerank-multilingual-v3.0`** (o con el fallback `0.75 * Cosine + 0.25 * RRF`), aplica una **Selección por Cuotas Jerárquicas** parametrizada en `src/config.py` (`RETRIEVAL_FINAL_TOP_K = 10`: `>= 4` artículos provinciales, `>= 2` tablas salariales provinciales, `>= 2` fragmentos del VI ALEH y `>= 2` fragmentos del Estatuto de los Trabajadores).

---

## 3. Arquitectura del Grafo y Memoria Multi-Sesión (`src/graph/` en LangGraph + SQLite)

El sistema opera mediante una máquina de estados finitos (`StateGraph(LegalGraphState)`) implementada en `src/graph/state.py`, `src/graph/memory.py`, `src/graph/nodes.py` y `src/graph/workflow.py`, optimizada para minimizar el consumo de peticiones RPM/RPD del Free Tier, mantener contexto multi-turno aislado por `session_id` y garantizar cero alucinaciones numéricas.

### 3.1. Diagrama de Flujo y Memoria Persistente

```text
           [ Entrada del Usuario (`session_id` + `query`) ]
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ `SessionMemoryStore` (`data/sessions/aihoreca_sessions.db`)  │
   │ • Carga `SessionContextSlots` (`active_territorio`,          │
   │   `active_categoria_aleh`, `pending_clarification_query`)    │
   │ • Carga Ventana Episódica Compacta (últimos 4 turnos, <250t) │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ NODO UNIFICADO (0 + 1 + 2): `analyze_and_route_node`                 │
   │ • Precedencia Temporal: `Turno Actual > Slots de Sesión Activa`      │
   │ • 1 llamada estructurada (`QueryAnalysisSchema` en `flash-lite`):    │
   │   1) Resolución de Correferencia y Fusión de HITL (`effective_query`)│
   │   2) Guardrail de Dominio (`HOSPITALITY_CORE` | `UNSUPPORTED` | `OOS`)│
   │   3) Gatekeeper Contextual (`requiere_convenio_provincial`)          │
   │   4) Traductor VI ALEH (Oficio coloquial -> Área Funcional y Grupo)  │
   └──────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼ (`OUT_OF_SCOPE` o       ▼ (`requiere_convenio`    ▼ (`HOSPITALITY_CORE` +
           `UNSUPPORTED_PROVINCE`)   y `territorio is None`)   Contexto Válido)
┌─────────────────────────┐ ┌─────────────────────────┐ ┌───────────────────────────────┐
│ NODO BLOQUEO            │ │ NODO CLARIFICACIÓN HITL │ │ NODO 3: `parallel_retrieve`   │
│ (`block_out_of_scope`)  │ │ (`ask_clarification`)   │ │ • Qdrant 3 Capas Estratificado│
│ • 0 tokens generación   │ │ • 0 tokens generación   │ │ • Cohere `rerank-v3.0`        │
│ • NO sobreescribe slots │ │ • Guarda `pending_query`│ │ • Selección por Cuota (`Top-8`)│
│   válidos de la sesión  │ │   en SQLite esperando   │ └───────────────┬───────────────┘
│   (Anti-State-Poisoning)│ │   provincia del usuario │                 │
└───────────┬─────────────┘ └───────────┬─────────────┘                 ▼
            │                           │               ┌───────────────────────────────┐
            │                           │               │ NODO 4 y 5:                   │
            │                           │               │ `legal_synthesis_and_audit`   │
            │                           │               │ • Síntesis con Jerarquía      │
            │                           │               │   Normativa (`Art. 3.3 ET`)   │
            │                           │               │ • Guardrail Numérico en CPU   │
            │                           │               │   (`AuditReport`: verifica €  │
            │                           │               │   y % contra los chunks)      │
            │                           │               └───────────────┬───────────────┘
            └───────────────────────────┴───────────────────────────────┤
                                                                        ▼
                           [ Persistencia de Turno + Actualización de Slots en SQLite ]
                                                                        │
                                                                        ▼
                                                          [ Respuesta Final Auditada ]
```

### 3.2. Especificación Técnica de los Módulos en Producción (`src/graph/`)

#### A. Gestor de Memoria Conversacional Multi-Sesión (`src/graph/memory.py` — `SessionMemoryStore`)
Resuelve los problemas críticos de continuidad conversacional multi-turno mediante SQLite local en modo `WAL` (`data/sessions/aihoreca_sessions.db`), sin añadir columnas rígidas por atributo ni parches *ad-hoc*:
1. **Separación entre *Working Memory* y *Episodic Memory* Enriquecida (`semantic_digest`):**
   - Tabla `sessions`: mantiene los `SessionContextSlots` universales de cada sesión (`active_territorio`, `active_oficio_coloquial`, `active_categoria_aleh`, `active_area_funcional_aleh`, `pending_clarification_query`).
   - Tabla `session_turns`: almacena la respuesta completa en Markdown para interfaz/auditoría y un resumen semántico compacto (`assistant_summary` con citas, cifras y `semantic_digest` de hasta `420` caracteres sin tablas Markdown pesadas) para que `get_compact_history(max_turns=4)` aporte tanto al Router (Nodo 1) como al Sintetizador (Nodo 4) visibilidad real de lo que el usuario planteó (horas de contrato, uno o varios puestos, tipo de local) y de lo que el asistente ya explicó en turnos anteriores.
2. **Precedencia Temporal Estricta (`Current Turn > Session Slots`):**
   - `detect_province_deterministic(query)` evalúa **únicamente el mensaje del turno actual**, nunca el texto concatenado del historial. Si el usuario cambia de provincia en el turno actual, la nueva provincia prevalece; si no menciona ninguna, hereda `active_territorio` de la sesión.
3. **Blindaje contra Envenenamiento de Estado (*Anti-State-Poisoning*):**
   - Si en mitad de una sesión válida (ej. en León) el usuario realiza una consulta hipotética sobre una provincia no soportada (ej. *"¿Y si me voy a Valencia?"*), el turno se bloquea con `UNSUPPORTED_PROVINCE` y se registra en `session_turns`, pero `save_turn_and_update_slots()` **preserva intactos** `active_territorio = "leon"` y `active_categoria_aleh` para los turnos siguientes.

#### B. Nodo Unificado 0 + 1 + 2 (`analyze_and_route_node`)
Combina reglas deterministas en CPU con una única llamada estructurada (`QueryAnalysisSchema`) sobre `gemini-3.1-flash-lite`:
1. **Descubrimiento Territorial Dinámico (`get_indexed_territories`):** Lee automáticamente `boundaries_cache.json` y expande comunidades multiprovinciales (`catalunya` -> `barcelona`, `girona`, `tarragona`, `lleida`) sin un solo `if "leon"` hardcodeado.
2. **Progresión Conversacional sin Repetición (`consulta_autocontenida_resuelta`):**
   - **Caso A (Desbloqueo HITL):** Si existía `pending_clarification_query` del turno anterior y el usuario aporta la provincia (*"Trabajo en Madrid"*), fusiona la pregunta pendiente con la provincia.
   - **Caso B (Repregunta en conversación activa):** Si `pending_clarification_query` es `null` y el usuario hace una repregunta o seguimiento (*"¿Y cómo sé a qué clase corresponde?"*, *"¿Y cuántos días tengo por mudanza?"*), hereda del historial los datos del caso (provincia, puesto(s), horas de contrato) pero centra `consulta_autocontenida_resuelta` y `query_busqueda_hibrida` **exclusivamente en la nueva duda del turno actual**, prohibiendo arrastrar preguntas ya respondidas en turnos anteriores.
3. **Traductor VI ALEH:** Mapea oficios coloquiales (*"friegaplatos"*, *"pinche"*, *"rider"*, *"camarera de pisos"*) a su denominación técnica, Grupo Profesional y Área Funcional (1ª a 6ª) del VI ALEH.

#### C. Nodos de Cortocircuito (`block_out_of_scope_node` y `ask_clarification_node`)
- **`block_out_of_scope_node`:** Rechazo determinista en `0,0 ms` (0 tokens LLM, 0 llamadas a Qdrant/Cohere), diferenciando entre consultas ajenas a la hostelería y provincias fuera del MVP.
- **`ask_clarification_node`:** Adelanta al usuario su encuadramiento estatal según el **VI ALEH**, guarda `pending_clarification_query` en SQLite y solicita que especifique su provincia (**Madrid**, **Barcelona/Cataluña** o **León**).

#### D. Nodo 3 (`parallel_retrieval_node`) y Nodos 4–5 (`legal_synthesis_and_audit_node`)
- **Nodo 3:** Ejecuta `parallel_hierarchical_retrieve()` sobre `effective_query` / `query_busqueda_hibrida`, devolviendo `10` fragmentos equilibrados entre el Estatuto de los Trabajadores (`2`), el VI ALEH (`2`) y el Convenio Provincial (`4` artículos + `2` tablas salariales vigentes).
- **Nodo 4 (Síntesis con Continuidad Conversacional y Adecuación a la Intención):** Recibe `conversation_history` junto con los fragmentos recuperados:
  - **Cero repetición entre turnos:** Recuerda todos los detalles aportados por el usuario en turnos previos (como un contrato parcial de 30 horas o varios puestos consultados) y tiene prohibido repetir tablas salariales o explicaciones ya entregadas en turnos anteriores salvo que se pida recalcularlas.
  - **Estructura adaptada a la pregunta:** Cuando el turno actual consulta sueldos o retribuciones, presenta la tabla comparativa en `€/mes` (**Jornada Completa 40h**, **Jornada Contratada SIN prorratear** en $N$ pagas y **Jornada Contratada CON pagas extras prorrateadas en 12 meses**) y calcula el importe en `€/hora` de las horas extraordinarias. Cuando la repregunta es conceptual, clasificatoria o procedimental, responde directa y naturalmente a esa duda sin forzar tablas salariales.
- **Nodo 5 (Guardrail Numérico de Salida con Verificación de Cálculos Derivados — `AuditReport`):** Ejecuta `_audit_numerical_and_citation_fidelity()`, verificando en CPU tanto las cifras literales extraídas de los fragmentos (`€` y `%`) como las operaciones aritméticas registradas en `calculos_derivados` (`DerivedCalculation`: proporcionalidad de jornada parcial `Horas / 40`, prorrateo en `12 / 14 / 15` pagas y fórmulas de hora extraordinaria).

### 3.3. Resultados Empíricos de Validación Multi-Sesión (6 Turnos Entrelazados en 2 Sesiones)

| Turno | `session_id` | Mensaje del Usuario (`query`) | `effective_query` Resuelta | Ruta | `active_territorio` (Post) | Cifras Verificadas / Resultado (`AuditReport`) |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | `ses-madrid-friegaplatos` | *"Soy friegaplatos a jornada completa, ¿cuánto tengo que cobrar de sueldo base y cuántos días de vacaciones me tocan?"* | *(Guarda `pending_clarification_query`)* | `ask_clarification` | `null` | Adelanta ALEH VI (`Grupo III`) y pide provincia |
| **2** | `ses-leon-pinche` | *"Soy ayudante de cocina (pinche) en un restaurante en León. ¿Cuál es mi salario base según las tablas salariales?"* | *"Soy ayudante de cocina en un restaurante en León, ¿cuál es mi salario base...?"* | `parallel_retrieve` -> `synthesis` | **`"leon"`** | `1.223,60 €`, `1.185,31 €`, `1.147,07 €`, `24,65 €` (**100%**) |
| **3** | `ses-madrid-friegaplatos` | ***"Trabajo en Madrid."*** | ***"Soy friegaplatos a jornada completa en Madrid, ¿cuánto tengo que cobrar de sueldo base y cuántos días de vacaciones me tocan...?"*** | `parallel_retrieve` -> `synthesis` | **`"madrid"`** | `1.101,69 €`, `1.032,48 €`, `183,87 €`, `30 días` (**100%**) |
| **4** | `ses-madrid-friegaplatos` | ***"¿Y cómo se regula o cuánto se paga por nocturnidad en mi caso?"*** | ***"Soy friegaplatos a jornada completa en Madrid, ¿cómo se regula y cuánto se paga el plus de nocturnidad...?"*** | `parallel_retrieve` -> `synthesis` | **`"madrid"`** | `1%` (`22h-00h`), `25%` (`00h-08h`), `1.049,28 €`, `1.057,56 €` (**100%**) |
| **5** | `ses-leon-pinche` | ***"¿Y si me ofrecen irme de cocinero a un hotel en Valencia, cuánto cobraría allí?"*** | *"Si acepto una oferta de trabajo como cocinero en un hotel en Valencia..."* | `block_out_of_scope` (`UNSUPPORTED`) | **`"leon"`** *(Intacto)* | Bloquea `valencia` en `0,01 ms` sin corromper el slot `"leon"` |
| **6** | `ses-leon-pinche` | ***"Vale, quedándome aquí en mi restaurante actual, ¿cuántos días de vacaciones anuales y permisos tengo?"*** | ***"Soy ayudante de cocina en un restaurante en León, ¿cuántos días de vacaciones anuales y permisos retribuidos me corresponden...?"*** | `parallel_retrieve` -> `synthesis` | **`"leon"`** | `Art. 24.º` (`31 días` vacaciones) y `Art. 23.º` (Permisos León) |

---

## 4. Evaluación Cuantitativa (RAGAS + Métricas Deterministas) y Auditoría del Índice (`src/evaluation/`)

El módulo `src/evaluation/` (`golden_dataset.py` y `evaluator.py`) y el notebook ejecutado `notebooks/01_evaluacion_ragas_y_auditoria.ipynb` implementan el benchmark empírico reproducible del sistema con persistencia en `data/processed/evaluation_results.json`.

### 4.1. Decisión de Ingeniería: Por qué un Juez RAGAS Estructurado de Pasada Única (`RAGASJudgeSchema`)
La función `ragas.evaluate()` estándar dispara entre **10 y 15 llamadas LLM asíncronas concurrentes por muestra** para descomponer afirmaciones y puntuar métricas por separado. En el **Free Tier de Google AI Studio (5–15 RPM)**, ese patrón agota la cuota en el segundo caso (`429 RESOURCE_EXHAUSTED`).
Nuestra arquitectura resuelve este cuello de botella combinando:
1. **Métricas Deterministas Exactas en CPU (0 llamadas API):** `intent_accuracy`, `route_accuracy`, `territorio_accuracy`, `layer_coverage_score` (presencia de las 3 capas normativas), `citation_recall_at_k` (presencia de artículos/anexos clave en el `Top-8`), `ground_truth_number_hit` y `zero_numerical_hallucination` (verificación literal de cada `€` y `%` en el Nodo 5).
2. **Juez Semántico RAGAS en 1 Sola Llamada Pydantic (`_evaluate_ragas_single_pass`):** Evalúa simultáneamente `faithfulness`, `answer_relevancy`, `context_recall_semantic` y `hierarchy_application_score` (aplicación del principio de norma más favorable `Art. 3.3 ET`) con justificación experta (`judge_critique`).

### 4.2. Auditoría Estructural del Índice Híbrido Qdrant (`648` Chunks)

| Documento Procesado (`data/processed/`) | Capa Normativa | Territorio(s) | `articulo` | `tabla_salarial` | `anexo_o_disposicion` | **Total Chunks** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `estatuto_trabajadores_consolidado.md` | `estatuto_trabajadores` | `estatal` | 204 | 0 | 47 | **251** (`38.7%`) |
| `aleh_vi_estatal.md` | `aleh_estatal` | `estatal` | 30 | 4 | 0 | **34** (`5.2%`) |
| `convenio_catalunya_bcn.md` | `convenio_provincial` | `barcelona, girona, tarragona, lleida, catalunya` | 83 | 25 | 40 | **148** (`22.8%`) |
| `convenio_madrid.md` | `convenio_provincial` | `madrid` | 74 | 47 | 15 | **136** (`21.0%`) |
| `convenio_leon_texto.md` | `convenio_provincial` | `leon` | 60 | 19 | 0 | **79** (`12.2%`) |
| **TOTAL CORPUS INDEXADO** | **3 Capas Jerárquicas** | **Estatal + 6 Provincias/CCAA** | **451** | **95** | **102** | **648** (`100%`) |

*Estadísticas de longitud de fragmento (caracteres):* `min = 175`, `p25 = 806`, `mediana = 1.632`, `media = 1.651,1`, `p75 = 2.716`, `max = 2.897` (acotado bajo el techo de `2.900` caracteres).

### 4.3. Resultados Globales y Caso por Caso sobre el Golden Dataset (`12` Casos)

* **Exactitud de Enrutamiento y Guardrails (`Intent` / `Route` / `Territorio`):** **100,00%** (`12/12`).
* **Cobertura de las 3 Capas Normativas (`Mean Layer Coverage`) y `Citation Recall@8`:** **100,00%**.
* **Tasa de Cero Alucinación Numérica (`Zero Numerical Hallucination Rate` - Nodo 5):** **100,00%** (`12/12`).
* **Métricas Semánticas RAGAS:** `Faithfulness = 98,57%`, `Answer Relevancy = 100,00%`, `Context Recall Semántico = 97,14%`, `Hierarchy Application Score = 100,00%`.
* **Latencias Medias por Nodo:** Nodo 1 (`analyze_and_route`): `3.899,3 ms` | Nodo 3 (`parallel_retrieval` Qdrant + Cohere): **`1.472,2 ms`** | Nodo 4+5 (`legal_synthesis_and_audit`): `12.697,6 ms`.

| Caso ID (`GoldenTestCase`) | Categoría | Ruta Ejecutada | Territorio | Capas | Cit@8 | 0-Halluc | Faith. | Relev. | CtxRec | Hier. |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GD_01_OOS_TRAFICO` | `guardrail_routing` | `block_out_of_scope` | `None` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_02_OOS_OTRO_SECTOR` | `guardrail_routing` | `block_out_of_scope` | `None` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_03_UNSUPPORTED_SEVILLA` | `guardrail_routing` | `block_out_of_scope` | `sevilla` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_04_HITL_FRIEGAPLATOS_SIN_PROVINCIA` | `guardrail_routing` | `ask_clarification` | `None` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_05_MADRID_HITL_RESOLUCION` | `multi_turn_memory` | `parallel_retrieve` | `madrid` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_06_MADRID_SEGUIMIENTO_NOCTURNIDAD` | `multi_turn_memory` | `parallel_retrieve` | `madrid` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_07_LEON_PINCHE_SALARIO` | `provincial_rag` | `parallel_retrieve` | `leon` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_08_LEON_SWITCH_VALENCIA_ANTI_POISONING` | `multi_turn_memory` | `block_out_of_scope` | `valencia` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_09_LEON_RETORNO_VACACIONES_Y_PERMISOS` | `multi_turn_memory` | `parallel_retrieve` | `leon` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |
| `GD_10_BARCELONA_RIDER_Y_FALTA_MUY_GRAVE` | `provincial_rag` | `parallel_retrieve` | `barcelona` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `0.95` | `1.00` |
| `GD_11_GIRONA_CAMARERA_PISOS` | `provincial_rag` | `parallel_retrieve` | `girona` | `1.00` | `1.00` | `True` | `0.90` | `1.00` | `0.85` | `1.00` |
| `GD_12_ESTATAL_ALEH_AREAS_FUNCIONALES` | `provincial_rag` | `parallel_retrieve` | `None` | `1.00` | `1.00` | `True` | `1.00` | `1.00` | `1.00` | `1.00` |

---

## 5. Despliegue Web y Telemetría Relacional (`despliegue_web_y_telemetria_hf.md`)

El diseño de publicación como Demo Web en **Streamlit Community Cloud** y el subsistema de observabilidad relacional (local en `data/logs/` por defecto y sincronizable opcionalmente con un Dataset privado propio en Hugging Face Hub) se detallan en [`despliegue_web_y_telemetria_hf.md`](file:///d:/Proyectos/AiHoreca/despliegue_web_y_telemetria_hf.md):

1. **Aislamiento por Visitante en UI (`st.session_state["visitor_session_ids"]`):** Cada navegador visualiza únicamente sus propias sesiones en la barra lateral, sin exponer sesiones ajenas ni paneles de administración en el frontend.
2. **Modelo Relacional de Logs (`session_id` + `turn_index`):**
   - `data/logs/conversaciones.txt`: Diálogos en texto plano limpio agrupados por `session_id` (`ORDER BY created_at, turn_index`) para inspección cualitativa sin entrelazado de usuarios concurrentes.
   - `data/logs/turnos_metricas.csv` y `data/sessions/aihoreca_sessions.db`: Tabla estructurada por `(session_id, turn_index)` con intención, ruta, territorio, auditoría numérica, latencia y feedback (`SIN_VOTO`, `OK`, `KO`).
3. **Persistencia en Dos Fases y Sincronización con Debounce:**
   - **Fase 1 (`T = 0 ms`):** El turno se guarda síncronamente al generarse la respuesta con `user_feedback = 'SIN_VOTO'`, garantizando el registro del 100% de las interacciones aunque el usuario no pulse `👍 / 👎`.
   - **Fase 2 (`T > 0`):** Si el usuario vota `👍 / 👎`, se ejecuta un `UPDATE` sobre `(session_id, turn_index)` y se encola la resincronización.
   - **Control de Rate-Limits y Restauración al Arranque:** Un hilo en segundo plano con *debounce* (`15 s`) sube los artefactos al Dataset privado de Hugging Face (`HfApi.upload_folder`), y al arrancar el contenedor restaura `aihoreca_sessions.db` desde dicho Dataset para no perder histórico tras reinicios del Space.

---

## 6. Metodología de Desarrollo y Control del Agente (Enfoque Modular-First)

Para garantizar la mantenibilidad y evitar el acoplamiento del estado oculto (*hidden state*) típico de los entornos interactivos, el desarrollo se segmenta rígidamente:

*   **Ingeniería de Producción (`/src`):** Todo el pipeline de ingesta híbrida (`PyMuPDF` + `google-genai`), el motor RAG, la orquestación (`LangGraph`), la memoria multi-sesión (`SQLite`) y la base vectorial (`Qdrant`) se desarrollan **directamente en módulos `.py` tipados con `Pydantic`**.
*   **Evaluación y Auditoría (`/notebooks`):** Los Jupyter Notebooks (`.ipynb`) solo importan funciones ya construidas en `/src` para auditar artefactos puntuales o ejecutar métricas de evaluación empírica con `RAGAS`. **Prohibido** utilizar los notebooks para construir o prototipar pipelines antes de pasarlos a `/src`.
