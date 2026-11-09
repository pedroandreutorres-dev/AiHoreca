# AiHoreca — Guía Rápida de Comandos de Control de Calidad y Regresión

Esta guía reúne los comandos de terminal para ejecutar la suite de regresión y control de calidad continua (`src/evaluation/regression.py`), el evaluador RAGAS (`src/evaluation/evaluator.py`) y el motor de auditoría diaria (`src/app/audit_runner.py`).

Todos los comandos se ejecutan desde el directorio raíz del proyecto (`d:\Proyectos\AiHoreca`) en PowerShell o CMD.

---

## 1. Suite de Regresión Continua (`src.evaluation.regression`)

Compara la ejecución actual contra la línea base oficial (`data/processed/evaluation_baseline.json`).

### ⚡ Modo Rápido Determinista (Recomendado antes de cada push)
> **Coste de tokens del Juez**: 0 tokens.  
> **Tiempo de ejecución**: ~5 a 10 segundos.  
> Comprueba enrutamiento, provincias, rutas y ausencia de alucinaciones numéricas sobre los 14 casos.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick
```

---

### 🎯 Ejecutar un Caso Específico (Depuración rápida)
Ejecuta la prueba únicamente sobre el caso que estés optimizando:

```powershell
# Caso de desambiguación Ponferrada (Café Madrid -> León)
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --case-id GD_13_DESAMBIGUACION_LOCAL_PONFERRADA

# Caso comparativa multi-provincia (Colmenar Viejo vs Camponaraya)
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --case-id GD_14_COMPARATIVA_MADRID_LEON

# Caso de cambio de provincia y blindaje anti-state-poisoning
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --case-id GD_08_LEON_SWITCH_VALENCIA_ANTI_POISONING
```

---

### 🔄 Forzar Recomputación Completa (Sin usar caché de turnos)
Si has modificado prompts del generador o la lógica del grafo y quieres que vuelva a consultar a los modelos para todos los casos:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --force
```

---

### 🏆 Modo Completo con Juez Semántico RAGAS Single-Pass
> **Uso recomendado**: Antes de un lanzamiento mayor o cambio estructural de prompts.  
> Evalúa métricas semánticas de 0.0 a 1.0 (*faithfulness*, *answer_relevancy*, *context_recall*, *hierarchy_score*).

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.regression --force
```

---

### 💾 Actualizar la Línea Base Oficial
Cuando añadas nuevos casos válidos al Golden Set o apruebes una mejora en el sistema, fija los nuevos resultados como la línea base oficial contra la que se comparará el futuro:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --update-baseline
```

---

### 📊 Salida en JSON Estructurado (Integración con scripts o CI)

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.regression --quick --json
```

* **Exit code 0**: Todos los casos de la línea base pasan sin regresión.
* **Exit code 1**: Se ha detectado una regresión (un caso que antes pasaba ahora falla).

---

## 2. Motor de Auditoría Diaria Local (`src.app.audit_runner`)

Descarga la telemetría de Hugging Face, archiva transcripciones completas en local, detecta turnos nuevos e incidencias, y compacta el historial remoto.

### 📥 Ejecución de Auditoría Diaria Incremental
> Si no hay turnos nuevos desde la última ejecución, finaliza en 1 segundo con 0 coste de tokens.

```powershell
.\.venv\Scripts\python.exe -m src.app.audit_runner
```

### 🔁 Forzar Descarga desde Hugging Face y Reauditoría Completa

```powershell
.\.venv\Scripts\python.exe -m src.app.audit_runner --force-pull --force-audit-all
```

### 📄 Salida en JSON para automatizaciones

```powershell
.\.venv\Scripts\python.exe -m src.app.audit_runner --json
```

---

## 3. Catálogo Actual del Golden Dataset (14 Casos Auditados)

| Identificador (`case_id`) | Categoría | Consulta Resumida | Comportamiento Clave Verificado |
|---|---|---|---|
| `GD_01_OOS_TRAFICO` | `guardrail_routing` | Indemnización por accidente de tráfico | Bloqueo determinista `block_out_of_scope` |
| `GD_02_OOS_OTRO_SECTOR` | `guardrail_routing` | Convenio de consultoría informática | Bloqueo determinista `block_out_of_scope` |
| `GD_03_UNSUPPORTED_SEVILLA` | `guardrail_routing` | Tablas de hostelería de Sevilla | Aviso pedagógico `UNSUPPORTED_PROVINCE` |
| `GD_04_HITL_FRIEGAPLATOS_SIN_PROVINCIA` | `guardrail_routing` | Friegaplatos sin indicar provincia | Gatekeeper solicita provincia (`ask_clarification`) |
| `GD_05_MADRID_HITL_RESOLUCION` | `provincial_rag` | Resolución indicando "Madrid" | Recuperación 3 capas y tablas de Madrid |
| `GD_06_MADRID_SEGUIMIENTO_NOCTURNIDAD` | `provincial_rag` | Turno nocturno en Madrid | Art. 27 Convenio Madrid y porcentaje recargo |
| `GD_07_LEON_PINCHE_SALARIO` | `provincial_rag` | Pinche de cocina en León | Nivel Salarial V, ALEH Área 2ª y tablas León |
| `GD_08_LEON_SWITCH_VALENCIA_ANTI_POISONING` | `multi_turn_memory` | Cambio de local a Valencia en el turno 3 | Limpieza de slots antiguos y cambio a Valencia |
| `GD_09_LEON_RETORNO_VACACIONES_Y_PERMISOS` | `multi_turn_memory` | Retorno a León: vacaciones y festivos | Retención de contexto y Art. 12/25 de León |
| `GD_10_BARCELONA_RIDER_Y_FALTA_MUY_GRAVE` | `provincial_rag` | Repartidor en Barcelona / falta muy grave | ALEH Anexo I, Art. 40-41 y Cataluña |
| `GD_11_GIRONA_CAMARERA_PISOS` | `provincial_rag` | Camarera de pisos en Girona | Mapeo interprovincial de Cataluña a Girona |
| `GD_12_ESTATAL_ALEH_AREAS_FUNCIONALES` | `provincial_rag` | 6 Áreas Funcionales del VI ALEH | Consulta estatal pura sin requerir provincia |
| `GD_13_DESAMBIGUACION_LOCAL_PONFERRADA` | `guardrail_routing` | Bar "Café Madrid" en Ponferrada | Desambiguación geográfica (León, no Madrid) |
| `GD_14_COMPARATIVA_MADRID_LEON` | `provincial_rag` | Colmenar Viejo vs Camponaraya | Comparativa multi-provincia (`MatchAny`) |

---

## 4. Archivos Clave del Sistema de Calidad

* **Línea base oficial**: [`data/processed/evaluation_baseline.json`](file:///d:/Proyectos/AiHoreca/data/processed/evaluation_baseline.json)
* **Casos añadidos desde producción**: [`data/processed/golden_set_produccion.jsonl`](file:///d:/Proyectos/AiHoreca/data/processed/golden_set_produccion.jsonl)
* **Cola de turnos marcados para revisión**: [`data/logs/candidatos_golden.jsonl`](file:///d:/Proyectos/AiHoreca/data/logs/candidatos_golden.jsonl)
* **Histórico permanente de métricas**: [`data/logs/historico_metricas.csv`](file:///d:/Proyectos/AiHoreca/data/logs/historico_metricas.csv)
