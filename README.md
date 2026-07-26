# ASis — Asistente IA Local de Organizacion Personal

**ASis** es un sistema de inteligencia artificial local para la organizacion automatica de documentos, gestion de email y administracion de archivos personales. Incluye un agente conversacional con LangGraph, un monitor de archivos en tiempo real con clasificacion por embeddings, y un widget de escritorio con interfaz glassmorphism.

---

## Arquitectura General

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     ASis Architecture                        │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │   ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
  │   │  Widget   │    │  CLI (Rich)  │    │  Folder      │     │
  │   │  PyQt6    │    │  Terminal    │    │  Monitor     │     │
  │   └─────┬────┘    └──────┬───────┘    └──────┬───────┘     │
  │         │                │                    │             │
  │         └────────┬───────┘                    │             │
  │                  ▼                            ▼             │
  │         ┌────────────────┐         ┌─────────────────┐     │
  │         │  AgentBridge   │         │  FolderManager  │     │
  │         │  (LangGraph)   │         │  (ASIORGA)      │     │
  │         └───────┬────────┘         └────────┬────────┘     │
  │                 │                           │              │
  │         ┌───────▼────────┐         ┌────────▼────────┐     │
  │         │  15 Tools      │         │  Document       │     │
  │         │  Email/File/   │         │  Classifier     │     │
  │         │  Folder        │         │  (Embeddings)   │     │
  │         └───────┬────────┘         └────────┬────────┘     │
  │                 │                           │              │
  │         ┌───────▼────────┐         ┌────────▼────────┐     │
  │         │  Ollama /      │         │  ONNX Runtime   │     │
  │         │  OpenAI /      │         │  MiniLM-L6-v2   │     │
  │         │  Anthropic     │         │  (384-dim)      │     │
  │         └────────────────┘         └─────────────────┘     │
  │                                                             │
  │   ┌──────────────────────────────────────────────────┐     │
  │   │              Email Monitor (IMAP)                 │     │
  │   │  Polling cada 5 min | OAuth2 Gmail | Adjuntos    │     │
  │   └──────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. Agente Conversacional (LangGraph)

Un agente con razonamiento multi-paso que puede ejecutar herramientas, pedir confirmacion al usuario para acciones sensibles, y mantener contexto de conversacion.

**Nodos del grafo:**

| Nodo | Funcion |
|------|---------|
| `process_input` | Punto de entrada, anade system prompt, detecta reanudaciones |
| `agent_reasoning` | Invoca el LLM con herramientas vinculadas. Limite de 10 iteraciones |
| `check_confirmation` | Verifica si alguna herramienta requiere aprobacion del usuario |
| `user_confirmation` | Punto de interrupcion — el grafo pausa hasta respuesta del usuario |
| `tool_execution` | Ejecuta las herramientas pendientes y recopila resultados |
| `generate_response` | Nodo terminal con la respuesta final |

**Flujo:**

```
process_input -> agent_reasoning -> check_confirmation -> user_confirmation
                        |                    |                    |
                        v                    v                    v
                   generate_response    tool_execution    tool_execution
                        |                    |                    |
                        v                    v                    v
                       END            agent_reasoning      agent_reasoning
```

### 2. Widget de Escritorio (PyQt6)

Widget flotante con efecto glassmorphism, modo compacto/extendido, y atajo de teclado `Ctrl+Shift+A`.

| Componente | Descripcion |
|-----------|-------------|
| `StatusHeader` | Barra superior draggable con indicador de estado animado (verde/amarillo/rojo) |
| `ResponsePanel` | Panel de chat con burbujas de usuario/agente y barra de confirmacion |
| `PromptInput` | Campo de texto con boton de envio |
| `QuickActions` | Acciones rapidas: Clasificar Documento, Resumir Reunion, Redactar Email |
| `SystemMetrics` | Barras de uso de RAM y VRAM (psutil + nvidia-smi) |
| `RagDropZone` | Zona de drag-and-drop para ingestar documentos |
| `AgentBridge` | Puente sincrono entre widget y agente LangGraph |

**Tema visual:** Catppuccin Mocha con bordes redondeados (16px), fondo semitransparente con efecto DWM blur (Windows).

### 3. Monitor de Archivos (ASIORGA)

Sistema de organizacion automatica que detecta archivos nuevos en carpetas monitorizadas, extrae su contenido, los clasifica con embeddings y los mueve a la carpeta destino correcta dentro de `~/Desktop/ASIorga/`.

**Pipeline de clasificacion:**

```
Archivo nuevo
    |
    v
Extracto de contenido (PDF/DOCX/TXT/CSV/Excel/OCR)
    |
    v
Clasificacion por embeddings (all-MiniLM-L6-v2, 6-8ms)
    |
    ├── Score alto + gap suficiente --> Carpeta destino
    |
    └── Score bajo o ambiguo -----> Fallback keywords ponderadas
                                          |
                                          v
                                    Carpeta destino / Documentos
```

**Carpetas destino configuradas:**

| Carpeta | Contenido |
|---------|-----------|
| Documentos | Archivos miscelaneos, notas, texto sin clasificacion |
| Fotos | Fotografias, capturas de pantalla, imagenes |
| Facturas | Facturas, recibos, tickets, documentos de compra |
| Trabajo | Nominas, contratos laborales, informes, presentaciones |
| Personal | DNI, NIE, pasaporte, certificados, seguros, declaraciones |
| Universidad | Guias docentes, examenes, TFG/TFM, apuntes, ECTS |

**Clasificador de documentos (two-tier):**

- **Primario:** Embeddings ONNX (`Xenova/all-MiniLM-L6-v2`, 384-dim). Pre-calcula embeddings de las descripciones de carpetas al iniciar. Clasifica por cosine similarity con threshold (0.35) + gap check (0.04). Sin dependencias de PyTorch ni ChromaDB.
- **Fallback:** Reglas keyword ponderadas para archivos con contenido insuficiente o ambiguo. Patrones de nombre de archivo + palabras clave del contenido con pesos.

**Extractores de contenido soportados:**

| Formato | Metodo |
|---------|--------|
| PDF | PyMuPDF (pymupdf) |
| DOCX | python-docx |
| TXT / MD | Lectura directa |
| CSV / Excel | pandas + openpyxl |
| JPG / PNG | RapidOCR (onnxruntime) |

### 4. Monitor de Email (IMAP)

Daemon en hilo background que revisa Gmail cada 5 minutos via IMAP con OAuth2. Descarga adjuntos automaticamente de remitentes configurados.

| Servicio | Metodo | Uso |
|----------|--------|-----|
| `ImapEmailService` | IMAP + OAuth2 bearer | Lectura de emails, busqueda, descarga de adjuntos |
| `GmailSMTPService` | SMTP + XOAUTH2 | Envio de emails (texto y HTML) |
| `EmailMonitor` | Thread polling | Monitoreo automatico de remitentes configurados |
| `oauth_helper` | Google OAuth2 | Carga/refresh/autorizacion de tokens |

### 5. Sistema de Herramientas (15 tools)

**Email:**

| Herramienta | Riesgo | Descripcion |
|------------|--------|-------------|
| `search_emails` | Bajo | Buscar emails por remitente, asunto, fecha |
| `get_email_content` | Bajo | Obtener contenido completo de un email |
| `download_attachment` | Alto | Descargar adjunto especifico (requiere confirmacion) |
| `send_email` | Alto | Enviar email (requiere confirmacion) |
| `check_and_download_documents` | Medio | Buscar ultimo email no leido y descargar adjuntos |

**Archivos:**

| Herramienta | Riesgo | Descripcion |
|------------|--------|-------------|
| `save_file` | Medio | Guardar contenido en directorio de datos (requiere confirmacion) |
| `list_files` | Bajo | Listar archivos en un directorio |
| `organize_documents` | Medio | Organizar archivos por extension/fecha (requiere confirmacion) |

**Carpetas / ASIORGA:**

| Herramienta | Riesgo | Descripcion |
|------------|--------|-------------|
| `add_monitored_folder` | -- | Anadir carpeta al monitor de watchdog |
| `remove_monitored_folder` | -- | Quitar carpeta del monitor |
| `list_monitored_folders` | -- | Listar carpetas monitorizadas |
| `create_destination_folder` | Confirmar | Crear nueva categoria en ASIORGA |
| `delete_destination_folder` | Confirmar | Eliminar categoria de ASIORGA |
| `list_destination_folders` | -- | Listar categorias con descripciones |
| `list_folder_contents` | -- | Ver contenido de una carpeta ASIORGA |

---

## Estructura del Proyecto

```
ASis/
├── app/
│   ├── main.py                    # Entry point CLI (Rich)
│   ├── config/
│   │   ├── settings.py            # Configuracion (pydantic-settings + .env)
│   │   ├── prompts.py             # System prompts del agente
│   │   └── logging_config.py      # Logging con Rich + archivo UTF-8
│   ├── agent/
│   │   ├── graph.py               # StateGraph de LangGraph
│   │   ├── state.py               # AgentState (TypedDict)
│   │   ├── nodes.py               # Nodos y funciones de routing
│   │   └── memory.py              # Backend de memoria (InMemory)
│   ├── tools/
│   │   ├── base.py                # ToolRiskLevel, permisos, metadata
│   │   ├── registry.py            # Auto-discovery de herramientas
│   │   ├── email_tools.py         # 5 herramientas de email
│   │   ├── file_tools.py          # 3 herramientas de archivos
│   │   └── folder_tools.py        # 7 herramientas de carpetas/ASIORGA
│   ├── models/
│   │   └── llm.py                 # Factory: Ollama / OpenAI / Anthropic
│   ├── services/
│   │   ├── email_service.py       # Interfaz abstracta de email
│   │   ├── imap_service.py        # IMAP via imapclient + OAuth2
│   │   ├── smtp_service.py        # Gmail SMTP con XOAUTH2
│   │   ├── oauth_helper.py        # Flujo OAuth2 compartido
│   │   ├── email_monitor.py       # Daemon de polling IMAP
│   │   ├── storage_service.py     # Almacenamiento local + colisiones
│   │   ├── folder_monitor.py      # Watchdog + clasificacion automatica
│   │   ├── folder_manager.py      # CRUD de ASIORGA + mover archivos
│   │   ├── document_extractor.py  # Extraccion multi-formato
│   │   ├── document_classifier.py # Embeddings ONNX + keywords fallback
│   │   └── image_classifier.py    # Clasificador heuristico de imagenes
│   ├── utils/
│   │   └── helpers.py             # Utilidades generales
│   └── widget/
│       ├── __main__.py            # Entry point widget (Ctrl+Shift+A)
│       ├── dashboard.py           # Widget principal glassmorphism
│       ├── dwm.py                 # Windows DWM blur/dark mode
│       ├── styles.py              # Tema Catppuccin Mocha + QSS
│       ├── api/
│       │   ├── agent_bridge.py    # Puente widget <-> LangGraph
│       │   └── llm_client.py      # Cliente HTTP directo para Ollama
│       └── components/
│           ├── status_header.py   # Header draggable + status dot
│           ├── prompt_input.py    # Campo de texto + boton enviar
│           ├── response_panel.py  # Chat bubbles + confirmacion
│           ├── quick_actions.py   # Acciones rapidas + menu
│           ├── system_metrics.py  # Barras RAM/VRAM
│           └── rag_drop.py        # Zona drag-and-drop
├── config/
│   ├── destination_folders.json   # 6 categorias ASIORGA
│   ├── monitored_folders.json     # Carpetas monitorizadas
│   └── monitored_senders.json     # Remitentes de email
├── scripts/
│   └── authorize_gmail.py         # Autorizacion OAuth2 de Gmail
├── tests/
│   ├── test_tools.py              # Tests de registry y metadata
│   ├── test_llm.py                # Tests de factory LLM y settings
│   └── test_graph.py              # Tests de estado del agente
├── data/                          # Datos locales (gitignored)
├── .env.example                   # Plantilla de configuracion
├── requirements.txt               # 23 dependencias
└── pyproject.toml                 # Metadata + ruff + pytest config
```

---

## Instalacion

### Prerrequisitos

- **Python 3.11+**
- **Ollama** instalado y ejecutando ([ollama.ai](https://ollama.ai))
- **Git**

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/joseparedes14/ASis.git
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
ollama pull qwen3.5:latest

# 6. Ejecutar (una de dos opciones)
python -m app.main          # CLI terminal
python -m app.widget        # Widget de escritorio
```

---

## Configuracion

### Variables de Entorno

| Variable | Por defecto | Descripcion |
|----------|-------------|-------------|
| `LLM_PROVIDER` | `ollama` | Proveedor: `ollama`, `openai`, `anthropic` |
| `LLM_MODEL` | `qwen3.5:latest` | Modelo a usar |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL de Ollama |
| `EMAIL_ADDRESS` | -- | Direccion Gmail |
| `EMAIL_OAUTH_CLIENT_ID` | -- | Google OAuth client ID |
| `EMAIL_OAUTH_CLIENT_SECRET` | -- | Google OAuth client secret |
| `EMAIL_OAUTH_CREDENTIALS_FILE` | `./credentials.json` | Archivo credenciales OAuth |
| `EMAIL_OAUTH_TOKEN_FILE` | `./data/oauth_token.json` | Token refresh OAuth |
| `EMAIL_MONITOR_INTERVAL` | `5` | Intervalo de polling (minutos) |
| `ASIORGA_ROOT` | `~/Desktop/ASIorga` | Raiz de carpetas destino |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `REQUIRE_CONFIRMATION` | `true` | Confirmar acciones sensibles |

### Cambiar de Modelo

```bash
# Modelo local con Ollama
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:latest

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Uso

### CLI Terminal

```bash
python -m app.main

# Comandos disponibles:
# /help    - Muestra la ayuda
# /clear   - Limpia historial de conversacion
# /tools   - Lista herramientas disponibles
# /status  - Estado del sistema
# /exit    - Salir
```

### Widget de Escritorio

```bash
python -m app.widget
# Atajo: Ctrl+Shift+A para mostrar/ocultar
```

El widget opera en dos modos:
- **Compacto:** Barra flotante con status y boton de expandir
- **Extendido:** Panel completo con chat, acciones rapidas y metricas del sistema

---

## Testing

```bash
pytest tests/ -v
```

**Linting:**

```bash
ruff check app/ --fix
ruff format app/
```

---

## Dependencias Principales

| Paquete | Uso |
|---------|-----|
| `langchain` + `langgraph` | Framework de agente y grafo de estados |
| `langchain-ollama` | Integracion con Ollama |
| `PyQt6` | Widget de escritorio |
| `watchdog` | Monitoreo de sistemas de archivos |
| `onnxruntime` | Inferencia ONNX para embeddings y OCR |
| `huggingface_hub` | Descarga de modelo de embeddings |
| `pymupdf` | Extraccion de texto de PDF |
| `rapidocr-onnxruntime` | OCR para imagenes |
| `imapclient` + `google-auth` | IMAP con OAuth2 para Gmail |
| `rich` | Terminal CLI enriquecida |
| `psutil` | Metricas del sistema |

---

## Seguridad

- **Credenciales:** Siempre en `.env` o archivos gitignored, nunca en el codigo fuente
- **Confirmacion:** Herramientas de riesgo alto requieren aprobacion explicita del usuario
- **Permisos:** Cada herramienta declara su nivel de riesgo (`LOW`, `MEDIUM`, `HIGH`)
- **Local:** Con Ollama, todos los datos permanecen en la maquina del usuario
- **Git:** `.gitignore` protege `.env`, `credentials.json`, tokens OAuth, logs

---

## Licencia

MIT
