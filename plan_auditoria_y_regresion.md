# Plan de trabajo: Retención RGPD, Auditoría Diaria y Regresión de Calidad

> Documento vivo. Última actualización: 09/11/2026.
> Decisión tomada: **plazo de conservación = 90 días** (sin actividad), en origen y en local.

## Estado de partida

- Automatización `AiHoreca Daily Telemetry Audit` activa (08:00 Europe/Madrid, Gemini Pro). Primera ejecución programada correcta el 09/11/2026.
- Código de retención y `audit_runner.py` implementados en local, **pendientes de commit/push**.
- Golden set existente: 12 casos en `src/evaluation/golden_dataset.py`; evaluador híbrido (determinista + juez RAGAS single-pass) en `src/evaluation/evaluator.py`.

---

## Paso 1 — Purga de 90 días en producción

**Objetivo:** que Streamlit Cloud aplique realmente la política de conservación (RGPD art. 5.1.e).

- [x] Revisar diff: `src/config.py`, `src/graph/memory.py`, `src/app/service.py`, `src/app/telemetry_sync.py`, `src/app/audit_runner.py` (nuevo).
- [x] Verificar que `data/logs/`, `data/sessions/` y `.env` siguen excluidos por `.gitignore`.
- [x] Comprobación sintáctica/import de los módulos modificados.
  - Corregido antes del commit: faltaban `import time` (habría tumbado la app al arrancar) y la importación de `SESSION_RETENTION_DAYS` (la purga habría fallado en silencio).
  - Test de purga en BD temporal: sesión de 2020 eliminada, sesión reciente conservada.
- [x] Commit `9d14c81`: `feat: 90-day session retention at source + local daily audit pipeline`.
- [x] `git push` → redepliegue automático en Streamlit Cloud.
- [x] Verificación post-deploy: lanzar 1 consulta en la web y confirmar commit nuevo en `Andregon79/aihoreca-logs`.

**Funcionamiento:**
- `SessionMemoryStore.purge_inactive_sessions(90)` borra sesiones con `updated_at` > 90 días y sus turnos.
- `AiHorecaEngineService._apply_retention_policy_if_due()` la ejecuta al arrancar y como máximo cada 12 h; si borra, re-exporta logs y sincroniza con HF.
- La compactación mensual del historial de HF (`super_squash_history`) elimina los datos borrados de commits antiguos → borrado efectivo total en ≤ ~120 días.

---

## Paso 2 — Correcciones de `audit_runner.py`

- [x] **Bug etiquetas de feedback:** usar `OK` / `KO` (valores reales de la app) en lugar de `ME_GUSTA` / `NO_ME_GUSTA`. Afecta a recuentos 👍/👎 y a la detección de turnos con voto negativo.
- [x] **Votos tardíos:** guardar `last_feedback_at` en `.sync_cursor.json` e incluir turnos ya auditados que reciben voto después, en sección propia del informe.
- [x] **Zona horaria:** nombre y fechas del informe en Europe/Madrid.
- [x] **No sobrescribir:** si hay varias ejecuciones con turnos nuevos el mismo día, añadir al informe existente con cabecera de hora.
- [x] **Retención local 90 días:**
  - Borrar `data/logs/sesiones/*.txt` cuya sesión lleve > 90 días sin actividad (o ya no exista en la BD purgada).
  - Las métricas de cada ejecución se escriben en `data/logs/historico_metricas.csv` en el momento (sin texto libre, conservación indefinida), así borrar informes de > 90 días no pierde información.
- [x] **Candidatos al golden set:** activado por defecto (`--no-export-candidates` para desactivar); vuelca turnos marcados (KO, cifras no verificadas, auditoría fallida, votos tardíos KO) a `data/logs/candidatos_golden.jsonl`, sin duplicados, con caducidad de 90 días.
- [x] Línea base regenerada con `--force-audit-all` (09/11/2026): 51 turnos, 👍 4 / 👎 3, 7 turnos marcados, 7 candidatos.
- [x] Salida de consola en UTF-8 (tildes y emojis correctos en Windows).
- [x] Prompt de la automatización diaria actualizado a las nuevas secciones del informe.

---

## Paso 3 — Golden set ampliable y regresión antes de cada cambio

**Flujo:** informe diario → revisión humana (¿fallo real?) → caso reformulado sin datos personales + respuesta correcta → golden set → regresión → push solo si no empeora.

- [ ] Nuevo fichero `data/processed/golden_set_produccion.jsonl` validado con `GoldenTestCase` (añadir campos de trazabilidad: `origen`, `fecha`, `tipo_fallo`).
- [ ] `get_golden_dataset()` carga los 12 casos actuales + los del fichero.
- [ ] Guardar línea base: `data/processed/evaluation_baseline.json`.
- [ ] Comando `python -m src.evaluation.regression`:
  - **Bloquea** si un caso que pasaba ahora falla en checks deterministas (intención, ruta, territorio, cifras, citas).
  - **Avisa** si la media del juez RAGAS cae > 0,05 (faithfulness, answer correctness, context recall).
  - `--update-baseline` para fijar nueva línea base tras un cambio aceptado.
- [ ] Cuándo ejecutar: antes de push de cambios en prompts, retrieval, modelos o auditor numérico (no en cada commit, por coste de tokens).
- [ ] Opcional: hook `pre-push` que lo lance solo si cambian `src/graph/` o `src/retrieval/`.

---

## Política de conservación de datos (resumen)

| Dato | Ubicación | Plazo |
|---|---|---|
| Sesiones y turnos | SQLite en producción / HF Dataset | 90 días sin actividad (+ ≤30 días hasta compactación del historial HF) |
| Transcripciones por sesión | `data/logs/sesiones/` (local) | 90 días |
| Informes de auditoría con extractos | `data/logs/auditorias/` (local) | 90 días |
| Métricas agregadas sin texto libre | `data/logs/historico_metricas.csv` | Indefinido |
| Casos del golden set (reformulados, anonimizados) | `data/processed/golden_set_produccion.jsonl` | Indefinido |
