# Roadmap de Mejoras del Grafo (LangGraph) — AiHoreca

Documento de seguimiento para evolucionar la arquitectura agéntica de AiHoreca desde un pipeline lineal con ramificación inicial hacia un grafo con autocorrección determinista, paralelismo nativo multi-territorio (`Send`), recuperación en dos saltos para remisiones cruzadas y observabilidad en tiempo real.

**Línea base de partida (`22/22` casos — `evaluation_baseline.json`):**
- `intent_accuracy`: `1.0000` | `route_accuracy`: `1.0000` | `territorio_accuracy`: `1.0000` | `zero_hallucination_rate`: `1.0000`
- `mean_faithfulness`: `0.9812` | `mean_answer_relevancy`: `0.9938` | `mean_retrieval_latency_ms`: `1871.0 ms` | `mean_synthesis_latency_ms`: `11190.5 ms`

---

## Arquitectura Objetivo del Grafo

```text
START ──► analyze_and_route
            ├── [OUT_OF_SCOPE / UNSUPPORTED_PROVINCE] ──► block_out_of_scope ──► END
            ├── [Falta provincia o puesto]            ──► ask_clarification  ──► END
            └── [Datos completos: 1 o N territorios]
                     │ (Send paralelo por territorio + rama estatal)
                     ▼
              retrieve_territory (xN) + retrieve_state_layers
                     │ (fan-in reducer)
                     ▼
              verify_retrieval (CPU)
                ├── [Falta remisión cruzada / score bajo, hop < 1] ──► expand_retrieval ──┐
                └── [Contexto suficiente] ◄───────────────────────────────────────────────┘
                     │
                     ▼
              synthesize_response ◄──────────┐ (retry < 1 con warnings del auditor)
                     │                       │
                     ▼                       │
              audit_output (CPU) ──[FAIL]────┘
                     │ [PASS o retry == 1]
                     ▼
                    END
```

---

## Bloque A — Estabilidad, Autocorrección, Latencia Cloud y Deuda Técnica (Completado)

- [x] **A1. Separar `synthesize_response` + `audit_output` y cerrar ciclo de autocorrección**
  - **Archivos:** `src/graph/state.py`, `src/graph/nodes.py`, `src/graph/workflow.py`
  - **Problema:** `legal_synthesis_and_audit_node` ejecuta `_audit_numerical_and_citation_fidelity()` justo antes de `END`; si `passed == False` (`unverified_numbers`, falta de citas o fuga territorial), la respuesta fallida se entrega igual al usuario.
  - **Cambio:** Separar `synthesize_response` y `audit_output` como nodos propios en `build_aihoreca_graph()`. Añadir campos de control en `LegalGraphState` (`audit_retry_count`, `audit_feedback` o reutilizar `audit_report`) y una arista condicional tras `audit_output`: si `passed == False` y `retry < 1`, reintenta `synthesize_response` inyectando `unverified_numbers` y `warnings` para que corrija la cifra o cita; si vuelve a fallar, aplica advertencia o abstención determinista.
- [x] **A2. Paralelismo real en consultas a Qdrant Cloud**
  - **Archivos:** `src/retrieval/vector_store.py` (`parallel_hierarchical_retrieve`, `hybrid_search`)
  - **Problema:** Las 5–6 consultas a Qdrant Cloud (`estatuto_trabajadores`, `aleh_estatal`, `articulo`, `articulo pay_sparse`, `tabla_salarial`, `anexo_o_disposicion`) se ejecutan secuencialmente (`mean_retrieval_latency_ms = 1871.0 ms`).
  - **Cambio:** Lanzar en paralelo con `ThreadPoolExecutor` (compartiendo `d_vec` y `s_vec` ya calculados) las búsquedas por capa/tipo antes del Rerank de Cohere.
- [x] **A3. Territorios dinámicos en nodos terminales/esquemas y resumen preciso en `memory.py`**
  - **Archivos:** `src/graph/nodes.py` (`block_out_of_scope_node`, `ask_clarification_node`), `src/graph/state.py` (`QueryAnalysisSchema.intent`), `src/graph/memory.py` (`save_turn_and_update_slots`)
  - **Problema:** Los mensajes de `block_out_of_scope_node`, `ask_clarification_node` y `QueryAnalysisSchema` siguen listando a mano `"Madrid, Cataluña y León"` sin incluir `"Islas Baleares"` ni futuros territorios de `get_indexed_territories()`. Además, en `memory.py`, `ask_clarification` registra `"Solicitó provincia al usuario"` incluso cuando lo que faltaba era el puesto de trabajo.
  - **Cambio:** Construir la lista de cobertura dinámicamente desde `self.indexed_territories` y distinguir en `assistant_summary` si se solicitó provincia, puesto o ambos.
- [x] **Verificación Bloque A:**
  - `python -m src.evaluation.regression --quick` (`22/22` pass).
  - Test unitario/E2E verificando el reintento del ciclo `synthesize_response` $\rightarrow$ `audit_output` y medición comparativa de `mean_retrieval_latency_ms` contra `1871.0 ms`.

---

## Bloque B — Grafo Multi-Territorio (`Send`) y Recuperación de 2º Salto (Siguiente)

- [ ] **B1. Tipado `list[str]` + Fan-Out / Fan-In con `Send()` de LangGraph para comparativas**
  - **Archivos:** `src/graph/state.py`, `src/graph/nodes.py`, `src/graph/workflow.py`, `src/retrieval/vector_store.py` (`_build_filter`, `parallel_hierarchical_retrieve`)
  - **Problema:** `territorio_normalizado` es `Optional[str]` (ej. `"madrid,_leon"` en `GD_14`) y `MatchAny` comparte un único `QUOTA_PROVINCIAL_TABLES = 2` entre todas las provincias comparadas, permitiendo que una provincia acapare las tablas de la otra (más crítico cuando un convenio necesita 3 tablas como Baleares).
  - **Cambio:** Añadir `territorios_normalizados: list[str]` (manteniendo compatibilidad con evaluación/memoria) y usar `Send()` para ejecutar en paralelo una rama `retrieve_territory` independiente por cada territorio (con sus propias cuotas completas) + `retrieve_state_layers` compartida (`ET` + `ALEH`), reduciendo con `Annotated[list[RetrievedDocument], operator.add]` y deduplicación por `chunk_id`.
- [ ] **B2. Nodo `verify_retrieval` + salto condicional `expand_retrieval` (máx. 1 salto)**
  - **Archivos:** `src/graph/state.py`, `src/graph/nodes.py`, `src/graph/workflow.py`, `src/retrieval/vector_store.py`
  - **Problema:** El grafo pasa directo de retrieval a síntesis aunque un artículo cite expresamente otro artículo/disposición/anexo ausente en `retrieved_docs` (ej. remisión de nocturnidad a `Disposición adicional décima` o de `Art. 24º` a anexos) o cuando la relevancia máxima de Cohere sea baja.
  - **Cambio:** Nodo determinista en CPU que inspecciona remisiones internas explícitas en los primeros fragmentos provinciales; solo si falta una referencia citada y `retrieval_hop < 1`, dispara `expand_retrieval` para traerla sin añadir latencia ni tokens cuando el contexto ya es suficiente.
- [ ] **Verificación Bloque B:**
  - `python -m src.evaluation.regression --force` (`22/22` pass, verificando especialmente `GD_14_COMPARATIVA_MADRID_LEON` y casos multi-anexo).

---

## Bloque C — Streaming en Interfaz y Optimización I/O de Memoria

- [ ] **C1. Streaming de progreso nodo a nodo en la interfaz (`graph.stream`)**
  - **Archivos:** `src/graph/workflow.py`, `src/app/` (UI Streamlit)
  - **Problema:** `run_legal_query()` usa `graph.invoke()`, bloqueando la UI ~12s hasta terminar todo el grafo sin indicar en qué fase está.
  - **Cambio:** Exponer generador/callback con `graph.stream(..., stream_mode="updates")` manteniendo `run_legal_query()` compatible para evaluación/CLI.
- [ ] **C2. Desacoplar `export_flat_logs()` del camino crítico de cada turno**
  - **Archivos:** `src/graph/memory.py` (`save_turn_and_update_slots`, `export_flat_logs`)
  - **Problema:** `save_turn_and_update_slots()` reescribe de cero en disco el CSV y TXT histórico en cada consulta del usuario ($O(N)$ síncrono).
  - **Cambio:** Pasar a escritura incremental por turno o diferir la regeneración completa al proceso batch diario (`audit_runner`).
- [ ] **Verificación Bloque C:**
  - Confirmar renderizado progresivo por nodos en Streamlit, integridad de `sessions/*.db` y `logs/`, y ejecución completa de `python -m src.evaluation.regression --quick` (`22/22`).
