# 🤖 ASis — Asistente IA Local

**ASis** es un asistente de inteligencia artificial local (self-hosted) que interactúa con tus servicios personales mediante lenguaje natural. Construido con **LangGraph** y **LangChain**, diseñado para ser extensible, seguro y completamente privado.

## ✨ Características

- 🧠 **Agente inteligente** con razonamiento multi-paso basado en LangGraph
- 🔧 **Sistema de herramientas extensible** — añade nuevas capacidades sin modificar la arquitectura
- 🔒 **Seguridad** — confirmación del usuario para acciones sensibles, credenciales separadas del código
- 🖥️ **100% local** — tus datos nunca salen de tu máquina
- 🔄 **Multi-proveedor** — Ollama (local), OpenAI, Anthropic
- 📧 **Preparado para email** — interfaces listas para integrar IMAP, Gmail API, etc.
- 📁 **Gestión de archivos** — descarga y organización automática de adjuntos

## 📁 Estructura del Proyecto

```
ASis/
├── app/
│   ├── main.py                    # Entry point CLI
│   ├── config/
│   │   ├── settings.py            # Configuración (variables de entorno)
│   │   ├── prompts.py             # System prompts del agente
│   │   └── logging_config.py      # Logging profesional
│   ├── agent/
│   │   ├── graph.py               # StateGraph de LangGraph
│   │   ├── state.py               # AgentState (TypedDict)
│   │   ├── nodes.py               # Nodos del grafo
│   │   └── memory.py              # Capa de memoria
│   ├── tools/
│   │   ├── base.py                # Metadatos y permisos de herramientas
│   │   ├── email_tools.py         # Herramientas de email
│   │   ├── file_tools.py          # Herramientas de archivos
│   │   └── registry.py            # Auto-discovery de herramientas
│   ├── models/
│   │   └── llm.py                 # Factory de modelos LLM
│   ├── services/
│   │   ├── email_service.py       # Servicio de email (interfaz)
│   │   └── storage_service.py     # Servicio de almacenamiento local
│   └── utils/
│       └── helpers.py             # Utilidades generales
├── tests/                         # Tests unitarios
├── data/                          # Datos locales (no versionados)
├── .env.example                   # Plantilla de configuración
├── requirements.txt               # Dependencias Python
└── pyproject.toml                 # Metadata del proyecto
```

## 🏗️ Arquitectura

```
                    ┌──────────────┐
                    │ process_input │
                    └──────┬───────┘
                           │
                    ┌──────▼────────┐
              ┌────►│agent_reasoning │
              │     └──────┬────────┘
              │            │
              │     ┌──────▼────────────┐
              │     │¿Necesita tools?   │
              │     └──┬────────────┬───┘
              │   (no) │            │ (sí)
              │        │     ┌──────▼──────────────┐
              │        │     │ check_confirmation   │
              │        │     └──┬──────────────┬───┘
              │        │   (no) │              │ (sí)
              │        │        │       ┌──────▼────────────┐
              │        │        │       │ user_confirmation  │
              │        │        │       └──┬────────────┬───┘
              │        │        │     (sí) │            │ (no)
              │     ┌──▼────────▼──┐      │            │
              └─────┤tool_execution│◄─────┘            │
                    └──────────────┘                    │
                           │                           │
                    ┌──────▼──────────┐                │
                    │generate_response│◄───────────────┘
                    └──────┬──────────┘
                           │
                        ┌──▼──┐
                        │ END │
                        └─────┘
```

El agente usa un **StateGraph** de LangGraph donde:
- Cada **nodo** tiene una responsabilidad única
- El **estado** (`AgentState`) fluye entre todos los nodos
- Las **herramientas** están desacopladas del grafo
- El flujo es **condicional** — el agente razona antes de actuar

## 🚀 Instalación

### Prerrequisitos

- **Python 3.11+**
- **Ollama** instalado y ejecutándose ([ollama.ai](https://ollama.ai))

### Pasos

```bash
# 1. Clonar / navegar al proyecto
cd ASis

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env     # Windows
# cp .env.example .env       # Linux/Mac

# 5. Descargar modelo en Ollama
ollama pull llama3.1:8b

# 6. Ejecutar
python -m app.main
```

## 💬 Uso

### CLI Interactiva

```
╔══════════════════════════════════════════════════╗
║          🤖  ASis — AI Assistant  🤖             ║
║    Tu asistente inteligente de servicios locales  ║
╚══════════════════════════════════════════════════╝

Tú: Busca los últimos correos de Juan y dime cuáles tienen PDFs adjuntos

🤖 ASis:
[El agente razona, ejecuta herramientas y responde]
```

### Comandos

| Comando   | Descripción                        |
|-----------|-------------------------------------|
| `/help`   | Muestra la ayuda                    |
| `/clear`  | Limpia el historial de conversación |
| `/tools`  | Lista las herramientas disponibles  |
| `/status` | Muestra el estado del sistema       |
| `/exit`   | Sale de la aplicación               |

## 🔧 Cómo Añadir Nuevas Herramientas

Añadir herramientas es sencillo y **no requiere modificar el grafo del agente**:

### 1. Crear la herramienta

Crea un nuevo archivo en `app/tools/` o añade funciones a uno existente:

```python
# app/tools/calendar_tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.tools.base import ToolRiskLevel, tool_metadata

class GetEventsInput(BaseModel):
    date: str = Field(description="Fecha en formato YYYY-MM-DD")

@tool(
    args_schema=GetEventsInput,
    metadata=tool_metadata(risk_level=ToolRiskLevel.LOW, category="calendar"),
)
def get_calendar_events(date: str) -> str:
    """Obtener eventos del calendario para una fecha específica."""
    # Tu implementación aquí
    return f"Eventos para {date}: ..."

CALENDAR_TOOLS = [get_calendar_events]
```

### 2. Registrar en el registry

Añade una línea en `app/tools/registry.py`:

```python
def discover_tools(self) -> None:
    from app.tools.email_tools import EMAIL_TOOLS
    from app.tools.file_tools import FILE_TOOLS
    from app.tools.calendar_tools import CALENDAR_TOOLS  # ← Añadir

    self.register_many(EMAIL_TOOLS)
    self.register_many(FILE_TOOLS)
    self.register_many(CALENDAR_TOOLS)  # ← Añadir
```

¡Listo! La herramienta estará disponible para el agente automáticamente.

## 🧩 Cómo Añadir Nuevos Nodos al Grafo

Para flujos más complejos, puedes añadir nodos al grafo:

### 1. Crear la función del nodo

En `app/agent/nodes.py`:

```python
def post_process(state: AgentState) -> dict[str, Any]:
    """Nodo de post-procesamiento."""
    # Tu lógica aquí
    return {"metadata": {"post_processed": True}}
```

### 2. Añadir al grafo

En `app/agent/graph.py`:

```python
from app.agent.nodes import post_process

# Dentro de build_graph():
graph.add_node("post_process", post_process)
graph.add_edge("generate_response", "post_process")
graph.add_edge("post_process", END)
```

## 🔄 Cambiar de Modelo

Edita tu archivo `.env`:

```bash
# Modelo local con Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 🔒 Seguridad

- **Credenciales**: Siempre en `.env`, nunca en el código
- **Confirmación**: Las acciones sensibles requieren aprobación del usuario
- **Permisos**: Cada herramienta declara su nivel de riesgo
- **Local**: Todo se ejecuta en tu máquina, sin enviar datos a terceros (con Ollama)

## 📋 Roadmap

- [ ] Implementación completa del servicio de email (IMAP)
- [ ] Soporte OAuth2 para Gmail
- [ ] Memoria persistente (SQLite)
- [ ] Interfaz web
- [ ] Notificaciones automáticas
- [ ] Más herramientas (calendario, notas, etc.)
- [ ] Docker compose para despliegue

## 📄 Licencia

MIT
