"""
ASis — CLI entry point.

Provides a Rich-powered command-line interface for interacting
with the AI assistant. Handles the conversation loop, user
confirmation prompts, and graceful error handling.
"""

import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from app.agent.graph import build_graph
from app.agent.state import create_initial_state
from app.config.logging_config import get_logger, setup_logging
from app.config.settings import get_settings
from app.models.llm import LLMProviderError, check_ollama_connection
from app.services.email_monitor import EmailMonitor
from app.services.folder_monitor import get_folder_monitor

logger = get_logger(__name__)
console = Console()


# ── UI Helpers ──────────────────────────────────────────────────────────


def print_banner() -> None:
    """Display the application welcome banner."""
    banner = Text.from_markup(
        "[bold cyan]╔══════════════════════════════════════════════════╗\n"
        "║          🤖  ASis — AI Assistant  🤖             ║\n"
        "║    Tu asistente inteligente de servicios locales  ║\n"
        "╚══════════════════════════════════════════════════╝[/]"
    )
    console.print(banner)
    console.print()


def print_help() -> None:
    """Display available CLI commands."""
    help_text = """
[bold]Comandos disponibles:[/]
  [cyan]/help[/]     — Muestra esta ayuda
  [cyan]/clear[/]    — Limpia el historial de conversación
  [cyan]/tools[/]    — Lista las herramientas disponibles
  [cyan]/status[/]   — Muestra el estado del sistema
  [cyan]/exit[/]     — Sale de la aplicación
  [cyan]/quit[/]     — Sale de la aplicación

[dim]Escribe cualquier otra cosa para hablar con el asistente.[/]
"""
    console.print(help_text)


def print_tools(graph_app) -> None:
    """Display available tools with their descriptions."""
    # Access tools from the graph's bound LLM
    console.print("\n[bold cyan]🔧 Herramientas disponibles:[/]\n")
    from app.tools.registry import create_tool_registry

    registry = create_tool_registry()
    for tool in registry.get_all():
        risk = tool.metadata.get("risk_level", "unknown") if tool.metadata else "unknown"
        confirm = "⚠️  " if tool.metadata and tool.metadata.get("requires_confirmation") else "   "
        console.print(f"  {confirm}[bold]{tool.name}[/] [dim]({risk})[/]")
        console.print(f"      [dim]{tool.description}[/]")
    console.print()


def print_status(settings) -> None:
    """Display system status information."""
    ollama_ok = (
        check_ollama_connection(settings.ollama_base_url)
        if settings.llm_provider == "ollama"
        else None
    )

    status_lines = [
        f"[bold]Proveedor LLM:[/]  {settings.llm_provider}",
        f"[bold]Modelo:[/]         {settings.llm_model}",
    ]
    if ollama_ok is not None:
        status_icon = "✅" if ollama_ok else "❌"
        ollama_status = "Conectado" if ollama_ok else "No disponible"
        status_lines.append(
            f"[bold]Ollama:[/]         {status_icon} {ollama_status}"
        )
    status_lines.extend([
        f"[bold]Confirmación:[/]   "
        f"{'Activada' if settings.require_confirmation else 'Desactivada'}",
        f"[bold]Nivel de log:[/]   {settings.log_level}",
        f"[bold]Dir. datos:[/]     {settings.data_dir}",
    ])

    console.print(Panel("\n".join(status_lines), title="Estado del Sistema", border_style="cyan"))


def ask_confirmation(pending_tools: list[dict]) -> bool:
    """Ask the user to confirm pending tool executions.

    Args:
        pending_tools: List of pending tool call dicts.

    Returns:
        True if the user confirms, False otherwise.
    """
    console.print("\n[bold yellow]⚠️  Se requiere confirmación para ejecutar:[/]\n")
    for call in pending_tools:
        console.print(f"  🔧 [bold]{call['name']}[/]")
        for key, value in call.get("args", {}).items():
            console.print(f"     {key}: [dim]{value}[/]")
    console.print()

    try:
        response = console.input("[bold yellow]¿Deseas continuar? (s/N): [/]").strip().lower()
        return response in ("s", "si", "sí", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def display_response(content: str) -> None:
    """Display the assistant's response with markdown formatting.

    Args:
        content: Response text to display.
    """
    if content:
        console.print()
        console.print(Panel(Markdown(content), title="🤖 ASis", border_style="green"))


def display_notifications(notifications: list) -> None:
    """Display email download notifications to the user.

    Args:
        notifications: List of DownloadNotification objects.
    """
    for notif in notifications:
        console.print()
        console.print(
            Panel(
                f"[bold]De:[/] {notif.sender}\n"
                f"[bold]Asunto:[/] {notif.subject}\n"
                f"[bold]Archivos:[/] {', '.join(notif.filenames)}\n"
                f"[bold]Hora:[/] {notif.timestamp.strftime('%H:%M:%S')}",
                title="📧 Documento descargado",
                border_style="yellow",
            )
        )


def display_folder_notifications(notifications: list) -> None:
    """Display folder monitoring notifications to the user.

    Args:
        notifications: List of FileNotification objects.
    """
    for notif in notifications:
        if notif.success:
            console.print()
            console.print(
                Panel(
                    f"[bold]Archivo:[/] {notif.filename}\n"
                    f"[bold]Origen:[/] {notif.source_folder}\n"
                    f"[bold]Destino:[/] ASIORGA/{notif.destination_folder}\n"
                    f"[bold]Hora:[/] {notif.timestamp.strftime('%H:%M:%S')}",
                    title="📁 Archivo clasificado",
                    border_style="green",
                )
            )
        else:
            console.print()
            console.print(
                Panel(
                    f"[bold]Archivo:[/] {notif.filename}\n"
                    f"[bold]Error:[/] {notif.message}\n"
                    f"[bold]Hora:[/] {notif.timestamp.strftime('%H:%M:%S')}",
                    title="❌ Error al procesar archivo",
                    border_style="red",
                )
            )


# ── Main Loop ───────────────────────────────────────────────────────────


def main() -> None:
    """Main application entry point. Runs the interactive CLI loop."""
    # Initialize configuration
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    logger.info("Starting ASis — provider=%s, model=%s", settings.llm_provider, settings.llm_model)

    # Display banner
    print_banner()

    # Check Ollama connectivity
    if settings.llm_provider == "ollama":
        if not check_ollama_connection(settings.ollama_base_url):
            console.print(
                "[bold red]❌ Error:[/] No se puede conectar con Ollama en "
                f"{settings.ollama_base_url}\n"
                "   Asegúrate de que Ollama está ejecutándose: [cyan]ollama serve[/]\n"
            )
            sys.exit(1)
        console.print("[dim]✅ Conectado a Ollama[/]\n")

    # Build graph
    try:
        graph_app = build_graph(settings)
        console.print("[dim]✅ Grafo del agente construido[/]\n")
    except LLMProviderError as e:
        console.print(f"[bold red]❌ Error inicializando LLM:[/] {e}")
        sys.exit(1)

    print_help()

    # Start email monitor
    monitor = EmailMonitor(settings)
    monitored_senders = settings.get_monitored_senders()
    if monitored_senders:
        monitor.start()
        console.print(
            f"[dim]📧 Monitor de email activo — vigilando: {', '.join(monitored_senders)}[/]\n"
        )

    # Start folder monitor
    folder_monitor = get_folder_monitor(settings)
    from app.models.llm import create_llm
    llm = create_llm(settings)
    folder_monitor.set_llm(llm)
    folder_monitor.start()
    console.print("[dim]📁 Monitor de carpetas activo — ASIORGA[/]\n")

    # Conversation state
    state = create_initial_state()
    config = {"configurable": {"thread_id": "cli-session-1"}}

    # ── Interactive loop ────────────────────────────────────────────
    while True:
        # Check for email notifications before asking for input
        notifications = monitor.get_notifications()
        if notifications:
            display_notifications(notifications)

        # Check for folder monitoring notifications
        folder_notifications = folder_monitor.get_notifications()
        if folder_notifications:
            display_folder_notifications(folder_notifications)

        try:
            user_input = console.input("\n[bold blue]Tú:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            monitor.stop()
            folder_monitor.stop()
            console.print("\n[dim]👋 ¡Hasta luego![/]")
            break

        if not user_input:
            continue

        # Handle CLI commands
        command = user_input.lower()
        if command in ("/exit", "/quit"):
            monitor.stop()
            folder_monitor.stop()
            console.print("[dim]👋 ¡Hasta luego![/]")
            break
        elif command == "/help":
            print_help()
            continue
        elif command == "/clear":
            state = create_initial_state()
            console.print("[dim]🗑️  Historial limpiado[/]")
            continue
        elif command == "/tools":
            print_tools(graph_app)
            continue
        elif command == "/status":
            print_status(settings)
            continue

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        final_state = None
        try:
            with console.status("[bold cyan]Pensando...", spinner="dots"):
                for event in graph_app.stream(state, config=config, stream_mode="values"):
                    final_state = event
        except GraphInterrupt:
            logger.debug("Graph interrupted for confirmation")
        except KeyboardInterrupt:
            console.print("\n[dim]⏹️  Operación cancelada[/]")
            continue
        except Exception as e:
            logger.error("Error during graph execution: %s", e, exc_info=True)
            console.print(f"\n[bold red]❌ Error:[/] {e}")
            console.print("[dim]El agente intentará recuperarse en la próxima interacción.[/]")
            continue

        # Handle confirmation OUTSIDE the spinner
        if final_state and final_state.get("requires_confirmation"):
            pending = final_state.get("pending_tool_calls", [])
            if pending:
                confirmed = ask_confirmation(pending)
                final_state["user_confirmed"] = confirmed

                # Add resume flag so the graph knows not to process as a fresh turn
                metadata = final_state.get("metadata", {})
                final_state["metadata"] = {**metadata, "resume": True}

                if not confirmed:
                    console.print("[dim]❌ Acción cancelada por el usuario[/]")

                status_msg = "[bold cyan]Ejecutando..." if confirmed else "[bold cyan]Cancelando..."
                try:
                    with console.status(f"{status_msg}", spinner="dots"):
                        for event in graph_app.stream(
                            final_state,
                            config=config,
                            stream_mode="values",
                        ):
                            final_state = event
                except Exception as e:
                    logger.error("Error during graph resume: %s", e, exc_info=True)
                    console.print(f"\n[bold red]❌ Error:[/] {e}")
                    continue

        # Update conversation state
        if final_state:
            state = final_state

            # Display the last AI message
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    display_response(msg.content)
                    break


if __name__ == "__main__":
    main()
