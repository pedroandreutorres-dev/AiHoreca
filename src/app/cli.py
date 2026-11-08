"""Consola Interactiva Multi-Sesión (Terminal REPL) para AiHoreca (`src/app/cli.py`).

Ejecutar con:
    python -m src.app.cli
    python -m src.app.cli --session ses-mi-consulta
    python -m src.app.cli --session ses-test --query "Soy pinche en León, ¿cuál es mi salario base?"
"""

from __future__ import annotations

import argparse
import sys

from src.app.service import AiHorecaEngineService, TurnExecutionPayload


def _print_turn_summary(payload: TurnExecutionPayload) -> None:
    """Imprime en consola la respuesta jurídica junto con el resumen de telemetría y auditoría."""
    print("\n" + "=" * 95)
    print(
        f"[TELEMETRÍA] Sesión: {payload.session_id} | Intent: {payload.intent} | "
        f"Ruta: {payload.route_decision} | Territorio: {payload.territorio_normalizado or payload.provincia_detectada or 'N/A'}"
    )
    print(f"[CORREFERENCIA] Query Resuelta: {payload.effective_query}")
    if payload.categoria_profesional_aleh:
        print(
            f"[VI ALEH] Categoría: {payload.categoria_profesional_aleh} | "
            f"Área: {payload.area_funcional_aleh or 'N/A'}"
        )
    print(
        f"[GUARDRAIL NODO 5] Passed: {payload.audit_report.passed} | "
        f"Cifras Verificadas: {payload.audit_report.verified_numbers} | "
        f"No Verificadas: {payload.audit_report.unverified_numbers} | "
        f"Latencia: {payload.total_latency_ms:.0f} ms"
    )
    print("-" * 95)
    print(payload.final_response)
    print("=" * 95 + "\n")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Consola Interactiva Multi-Sesión de AiHoreca")
    parser.add_argument("--session", type=str, default="ses-cli-default", help="ID de sesión en SQLite")
    parser.add_argument("--query", type=str, default=None, help="Ejecuta un único turno y termina (modo one-shot)")
    parser.add_argument("--reset", action="store_true", help="Reinicia la sesión indicada antes de empezar")
    args = parser.parse_args()

    service = AiHorecaEngineService.get_instance()
    try:
        active_sid = args.session.strip()
        if args.reset:
            service.reset_or_delete_session(active_sid)
        service.create_session(active_sid)

        if args.query:
            payload = service.process_chat_turn(session_id=active_sid, query=args.query)
            _print_turn_summary(payload)
            return

        print("=" * 95)
        print(" AiHoreca — Consola Jurídica Multi-Sesión (Comandos: /sessions, /switch <id>, /slots, /reset, /exit)")
        print(f" Sesión activa inicial: {active_sid}")
        print("=" * 95)

        while True:
            try:
                raw_in = input(f"\n[{active_sid}] Consulta > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo de AiHoreca CLI...")
                break

            if not raw_in:
                continue

            if raw_in.lower() in ("/exit", "/quit", "salir"):
                break
            if raw_in.lower() == "/sessions":
                for s in service.list_active_sessions():
                    print(
                        f"  * {s.session_id:<24} | Turnos: {s.num_turns:<2} | "
                        f"Territorio: {str(s.active_territorio):<10} | Categoría: {s.active_categoria_aleh}"
                    )
                continue
            if raw_in.lower() == "/slots":
                slots = service.get_session_slots(active_sid)
                print(f"  [Slots de {active_sid}]: {slots.model_dump()}")
                continue
            if raw_in.lower() == "/reset":
                service.reset_or_delete_session(active_sid)
                service.create_session(active_sid)
                print(f"  [OK] Sesión '{active_sid}' reiniciada en SQLite.")
                continue
            if raw_in.lower().startswith("/switch "):
                parts = raw_in.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    active_sid = service.create_session(parts[1].strip())
                    print(f"  [OK] Cambiado a sesión '{active_sid}'.")
                continue

            payload = service.process_chat_turn(session_id=active_sid, query=raw_in)
            _print_turn_summary(payload)
    finally:
        service.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_cli()
