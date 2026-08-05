# ASis — Asistente IA local de organización personal

**ASis** es un asistente personal de inteligencia artificial que funciona 100 % en tu equipo. No es solo un chatbot: es un asistente que actúa sobre tus servicios y archivos reales. Le dices en lenguaje natural qué hacer y él:

- te responde como un chat normal (modelo local vía Ollama, o nube con OpenAI/Anthropic),
- busca y lee tus emails de Gmail, descarga adjuntos y envía correos por ti,
- **organiza tus documentos automáticamente**: vigila carpetas (Descargas, correos, etc.), detecta archivos nuevos, los clasifica y los mueve a una estructura ordenada en tu escritorio,
- **recuerda de tus correcciones**: si mueves un archivo a mano porque lo clasificó mal, ASis aprende de ello,
- responde preguntas sobre tus propios documentos gracias a un índice de búsqueda semántica (RAG).

Todo se controla desde un **widget flotante de escritorio** con estilo glassmorphism, sin depender de la nube si no quieres.

---

## Funcionalidades principales

### 1. Widget de escritorio glassmorphism

Interfaz flotante y semitransparente (Windows DWM blur) que se coloca en la esquina de tu pantalla. Tiene dos modos: un **círculo compacto** que no estorba y un **panel expandido** con chat, acciones rápidas, métricas de RAM/VRAM y zona de drag-and-drop. Atajo global: `Ctrl+Shift+A`. Se ajusta solo al cambiar de pantalla con distinto DPI.

```markdown
<!-- ZONA FOTO: captura del widget expandido (chat + acciones rápidas + métricas) -->
![Widget ASis expandido](docs/screenshots/widget_expandido.png)
```

Desde aquí lanzas acciones rápidas (clasificar documento, resumir reunión, redactar email), arrastras un archivo para que lo analice o activas la consulta a tus documentos de ASIORGA.

### 2. Agente conversacional con herramientas

Un agente construido con **LangGraph** (grafo de estados) que razona en varios pasos, decide qué herramienta necesita y ejecuta acciones sobre tu correo y tus archivos. Tiene un **sistema de confirmación**: las acciones sensibles (enviar email, descargar adjuntos, crear/eliminar carpetas) se pausan y te piden aprobación antes de ejecutarse, con una barra de confirmación dentro del propio widget.

```markdown
<!-- ZONA FOTO: barra de confirmación pidiendo aprobación para enviar un email -->
![Confirmación de acciones sensibles](docs/screenshots/confirmacion.png)
```

Herramientas disponibles (15 en total):

| Categoría | Herramientas |
|-----------|--------------|
| Email | `search_emails`, `get_email_content`, `download_attachment`, `send_email`, `check_and_download_documents` |
| Archivos | `save_file`, `list_files`, `organize_documents` |
| Carpetas / ASIORGA | `add_monitored_folder`, `remove_monitored_folder`, `list_monitored_folders`, `create_destination_folder`, `delete_destination_folder`, `list_destination_folders`, `list_folder_contents` |

### 3. Organización automática de documentos (ASIORGA)

El corazón del proyecto. **ASis** vigila carpetas que tú elijas (p. ej. `Descargas`) y, en cuanto aparece un archivo nuevo, lo procesa en cadena:

1. **Extrae** el contenido del archivo (PDF, DOCX, TXT, CSV, Excel y OCR de imágenes).
2. **Clasifica** con embeddings multilingües ONNX (*paraphrase-multilingual-MiniLM-L12-v2*, 384 dimensiones, sin PyTorch) usando k-NN con max-pooling sobre fragmentos del documento, umbral dinámico y una segunda pasada con resumen LLM si hay dudas.
3. **Mueve** el archivo a la carpeta correcta de `~/Desktop/ASIorga/` (Documentos, Fotos, Facturas, Trabajo, Personal, Universidad...).

El sistema **aprende de ti**: si mueves un archivo a mano, el monitor lo detecta, actualiza los textos de referencia de la carpeta y añade/elimina el vector exacto del documento en el índice k-NN. También detecta eliminaciones y mantiene los centroides reconstruidos.

```markdown
<!-- ZONA FOTO: estructura de carpetas ASIORGA en el escritorio con documentos clasificados -->
![Carpetas ASIORGA](docs/screenshots/asiorga.png)
```

### 4. Consulta tus documentos (RAG)

Cada documento clasificado se indexa automáticamente en un índice **FAISS** con fragmentos (chunks) de 512 caracteres. Puedes preguntar en lenguaje natural sobre tu documentación — *"¿cuánto pagué de luz en enero?"* — y ASis recupera los fragmentos más relevantes y responde citando los archivos fuente, sin inventarse nada.

```markdown
<!-- ZONA FOTO: pregunta al widget y respuesta con las fuentes citadas -->
![Consulta RAG sobre ASIORGA](docs/screenshots/rag_query.png)
```

### 5. Monitor de email

Un hilo en segundo plano revisa tu Gmail cada 5 minutos (configurable) vía IMAP con OAuth2 y **descarga automáticamente los adjuntos** de los remitentes que configures. Cuando el agente detecta documentos nuevos, te lo notifica en el widget.

---

## Cómo funciona por dentro

```
        ┌──────────────────────────────────────────────────────┐
        │                    Widget (PyQt6)                    │
        │  glassmorphism · chat · acciones rápidas · métricas  │
        └──────────────────────┬───────────────────────────────┘
                               │ AgentBridge
        ┌──────────────────────▼───────────────────────────────┐
        │              Agente LangGraph (grafo de estados)     │
        │  process_input → agent_reasoning → (confirmación)    │
        │         → tool_execution → generate_response         │
        └──────────┬──────────────────────┬────────────────────┘
                   │                      │
        ┌──────────▼───────────┐  ┌───────▼─────────────────────────┐
        │  15 herramientas      │  │  Email (IMAP/SMTP + OAuth2)    │
        │  email · archivos ·  │  │  búsqueda · descarga · envío    │
        │  carpetas            │  │  monitor de remitentes          │
        └──────────────────────┘  └────────────────────────────────┘
                   │
        ┌──────────▼─────────────────────────────────────────────┐
        │                Monitor de carpetas (watchdog)          │
        │  extracción → embeddings k-NN → clasificación → move   │
        │  aprende de correcciones y eliminaciones               │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                          ┌────────▼─────────┐      ┌──────────────────┐
                          │  RAG (FAISS)     │──────│  ASIORGA         │
                          │  índice + LLM    │      │  ~/Desktop/ASIorga│
                          └──────────────────┘      └──────────────────┘
```

- **LLM**: `app/models/llm.py` — factory que soporta **Ollama** (local, por defecto), **OpenAI** y **Anthropic**. Solo cambias variables de entorno.
- **Clasificador**: `app/services/document_classifier.py` — embeddings ONNX (sin PyTorch), índice k-NN con memoria de sesión para eliminar vectores exactos, clasificación en dos pasos (chunks → resumen LLM) y umbral dinámico.
- **Monitor de carpetas**: `app/services/folder_monitor.py` — watchdog con debounce, detección de correcciones manuales y de eliminaciones.
- **RAG**: `app/services/rag_service.py` — FAISS + recorte de texto + respuestas con fuentes citadas.

---

## Estructura del proyecto

```
ASis/
├── app/
│   ├── agent/                  # Grafo LangGraph, nodos, estado y memoria
│   ├── config/                 # settings (pydantic-settings), prompts, logging
│   ├── models/                 # Factory LLM (Ollama / OpenAI / Anthropic)
│   ├── tools/                  # 15 herramientas con auto-discovery
│   ├── services/               # Email, monitor, clasificador, RAG, extracción...
│   ├── utils/
│   └── widget/                 # Interfaz de escritorio (PyQt6)
│       ├── api/                # AgentBridge y cliente LLM HTTP
│       └── components/         # Header, chat, acciones, métricas, drag&drop
├── config/
│   ├── destination_folders.json   # Categorías de ASIORGA (con descripciones)
│   ├── monitored_folders.json     # Carpetas que vigilar
│   └── monitored_senders.json     # Remitentes de email monitorizados
├── scripts/authorize_gmail.py     # Autorización OAuth2 de Gmail (una sola vez)
├── data/                          # Datos locales y logs (ignorado por git)
├── tests/
├── .env.example                   # Plantilla de configuración
└── requirements.txt
```

---

## Requisitos

- **Windows** (recomendado) — el widget usa blur DWM y DPI awareness de Windows
- **Python 3.11+**
- **Ollama** instalado y en ejecución con el modelo `llama3.1:8b` descargado
  (es el modelo por defecto del agente y el que usa el clasificador para los resúmenes)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/joseparedes14/ASis.git
cd ASis

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env

# 5. Descargar el modelo en Ollama
ollama pull llama3.1:8b

# 6. Arrancar el widget
python -m app.widget
```

> El widget es el único punto de ejecución de la aplicación. `python -m app.widget` lanza todo: el agente, el monitor de carpetas, el monitor de email y el índice RAG.

---

## Configuración

Configuración vía archivo `.env` (mira `.env.example`):

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `LLM_PROVIDER` | `ollama` | Proveedor: `ollama`, `openai`, `anthropic` |
| `LLM_MODEL` | `llama3.1:8b` | Modelo a usar |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `EMAIL_ADDRESS` | — | Tu dirección Gmail |
| `EMAIL_OAUTH_CLIENT_ID` / `_SECRET` | — | OAuth2 de Google (opcional si usas `credentials.json`) |
| `EMAIL_OAUTH_CREDENTIALS_FILE` | `./credentials.json` | Archivo de credenciales OAuth |
| `EMAIL_OAUTH_TOKEN_FILE` | `./data/oauth_token.json` | Token de refresco |
| `EMAIL_MONITOR_INTERVAL` | `5` | Intervalo de revisión de email (minutos) |
| `ASIORGA_ROOT` | `~/Desktop/ASIorga` | Raíz de carpetas destino (variable del código) |
| `REQUIRE_CONFIRMATION` | `true` | Exigir confirmación en acciones sensibles |

---

## Cómo dejar la aplicación totalmente funcional

Esta es la **lista de comprobación** para que ASis quede al 100 % operativo. Sin estos pasos la aplicación arranca, pero el agente no podrá tocar tu correo ni clasificar documentos.

### 1. Arrancar el motor del modelo (obligatorio)

1. Instala **Ollama** desde [ollama.ai](https://ollama.ai) y déjalo ejecutándose.
2. Descarga el modelo que usa tanto el agente como el clasificador:

```bash
ollama pull llama3.1:8b
```

3. Comprueba que responde: `ollama list` debe mostrar el modelo. El widget te lo indicará con el semáforo de estado (verde = online).

### 2. Crear el `.env`

```bash
copy .env.example .env
```

Y edita al menos estas variables:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
EMAIL_ADDRESS=tu-correo@gmail.com
```

Sin `EMAIL_ADDRESS`, las herramientas de email fallarán. El resto del `.env` puede quedarse por defecto.

### 3. Autorizar Gmail (OAuth2) — para leer, descargar y enviar email

ASis accede a tu Gmail con OAuth2 (IMAP para leer y SMTP para enviar). Es un proceso de una sola vez:

1. Ve a la **Google Cloud Console** ([console.cloud.google.com](https://console.cloud.google.com)).
2. Crea un proyecto (o usa uno existente) y **activa la Gmail API**.
3. En *APIs & Services → Credentials*, crea un **OAuth 2.0 Client ID** de tipo *Desktop app*.
4. Descarga el JSON resultante y guárdalo como **`credentials.json`** en la raíz del proyecto.
5. Ejecuta el script de autorización (abrirá el navegador para que apruebes los permisos):

```bash
python scripts/authorize_gmail.py
```

6. Al terminar se creará `data/oauth_token.json` (el token se refresca solo). El script además comprueba que SMTP e IMAP conectan correctamente.

> El `credentials.json` y `data/oauth_token.json` están protegidos por `.gitignore`. Nunca los subas a GitHub.

### 4. Configurar qué remitentes vigilar (email)

Edita `config/monitored_senders.json` y añade los remitentes cuyos adjuntos quieres que ASis descargue automáticamente:

```json
{
  "senders": [
    { "email": "facturas@tu-empresa.com", "description": "Facturas", "enabled": true }
  ]
}
```

Con esto, el monitor de email revisa el buzón cada 5 minutos y baja los adjuntos de esos remitentes a `data/attachments/<remitente>/`.

### 5. Configurar qué carpetas monitorizar (ASIORGA)

Dos opciones:

- **Desde el widget**: menú *Acciones rápidas → Añadir carpeta a monitorizar* y selecciona la carpeta (p. ej. `Descargas`).
- **A mano**: edita `config/monitored_folders.json` con rutas absolutas:

```json
{
  "folders": [
    "C:\\Users\\TuUsuario\\Downloads"
  ]
}
```

Cuando aparezca un archivo nuevo en esas carpetas, ASis lo clasificará y lo moverá a ASIORGA.

### 6. Revisar las categorías destino de ASIORGA

ASis crea automáticamente `~/Desktop/ASIorga/` y sus carpetas (Documentos, Fotos, Facturas, Trabajo, Personal, Universidad) a partir de `config/destination_folders.json`. Puedes:

- añadir categorías por chat: *"crea una carpeta Proyectos para documentación de mis proyectos"*,
- o editar el JSON directamente con la descripción y textos de ejemplo de cada categoría.

En el primer arranque, ASis calcula los embeddings de las descripciones para empezar a clasificar y construye el índice RAG en segundo plano.

### 7. Arrancar el widget

```bash
python -m app.widget
```

Verifica el semáforo del widget (debe estar verde = LLM conectado) y prueba una clasificación dejando caer un documento en la zona de drag-and-drop.

### 8. Mantenimiento

- **Tests**: `pytest tests/ -v`
- **Lint**: `ruff check app/ --fix` y `ruff format app/`
- **Logs**: `data/logs/asis.log` (agente) y `data/logs/widget.log` (interfaz)

---

## Seguridad

- Todo lo sensible vive en `.env`, `credentials.json` y `data/` — archivos gitignoreados.
- Las herramientas declaran su nivel de riesgo (`LOW` / `MEDIUM` / `HIGH`); las de riesgo alto requieren **confirmación explícita** del usuario en el widget.
- Con Ollama, tus documentos y emails nunca salen de tu máquina.
- El agente nunca filtra datos sensibles innecesariamente y pregunta antes de acciones destructivas.

---

## Licencia

MIT
