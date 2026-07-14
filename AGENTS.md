# AGENTS.md — Instrucciones para Agentes de Código

## Reglas Generales

- Responde en el mismo idioma que el usuario.
- Sé conciso y directo.
- Antes de modificar código existente, lee el archivo completo para entender el contexto.

---

## Commits y Push a GitHub

Cuando el usuario te pida hacer un **commit** o **push**, **SIEMPRE** ejecuta esta secuencia ANTES de commitear:

### 1. Verificar seguridad

```bash
git status
```

Revisa que **NO** aparezcan archivos sensibles en la lista de "Changes to be committed":
- `.env`
- `credentials.json`
- `data/oauth_token.json`
- `token.json`
- `secrets.json`
- Cualquier archivo `.pem`, `.key`, `.log`
- Cualquier `.json` dentro de `data/`

### 2. Verificar el .gitignore

```bash
git check-ignore .env credentials.json data/oauth_token.json token.json secrets.json
```

Todos deben aparecer en la salida. Si alguno no aparece, **NO hagas commit** hasta corregir el `.gitignore`.

### 3. Verificar que no hay secrets en el código fuente

Busca patrones peligrosos en archivos `.py`:

```bash
rg "GOCSPX|sk-ant-|sk-|ya29\.|1\/\/|password\s*=\s*['\"][^'\"${]" app/ scripts/ tests/ --glob "*.py"
```

Si encuentra algo que no sea un placeholder o una carga desde `.env`/settings → **NO hagas commit**.

### 4. Decisión

| Resultado | Acción |
|---|---|
| Todo limpio | Haz `git add .`, `git commit` y `git push` sin preguntar |
| Archivo sensible detectado | **NO hagas commit**. Explica al usuario qué archivo es el problema y por qué no se puede subir |
| `.gitignore` incompleto | Corrige el `.gitignore` primero, luego commit |

### 5. Mensajes de commit

Usa el formato Conventional Commits:
- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `refactor:` reestructuración sin cambiar comportamiento
- `docs:` cambios en documentación
- `test:` añadir o modificar tests
- `chore:` tareas de mantenimiento

Ejemplo: `feat: add Gmail SMTP OAuth2 email sending tool`

---

## Archivos que NUNCA se deben commitear

| Archivo | Razón |
|---|---|
| `.env` | Contiene Client Secret y credenciales |
| `credentials.json` | Credenciales OAuth2 de Google |
| `data/oauth_token.json` | Refresh token de acceso a Gmail |
| `token.json` | Token de autenticación |
| `secrets.json` | Cualquier secreto futuro |
| `*.pem`, `*.key` | Claves criptográficas |
| `*.log` | Pueden contener emails y datos sensibles |
| `data/*.json` | Tokens y backups de credenciales |

---

## Configuración del Proyecto

- **Python**: 3.11+
- **Framework de agentes**: LangGraph
- **Gestión de dependencias**: `requirements.txt`
- **Configuración**: `pydantic-settings` con `.env`
- **Tests**: `pytest` + `pytest-asyncio`
- **Linting**: `ruff` (configurado en `pyproject.toml`)
