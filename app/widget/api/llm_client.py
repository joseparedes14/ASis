"""REST client for local LLM servers (Ollama, LM Studio, etc.)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("asis.widget.llm_client")


class AgentStatus(Enum):
    ONLINE = "online"
    THINKING = "thinking"
    OFFLINE = "offline"


@dataclass
class RolePreset:
    name: str
    system_prompt: str
    temperature: float


ROLE_PRESETS: dict[str, RolePreset] = {
    "General": RolePreset(
        name="General",
        system_prompt=(
            "Eres ASis, un asistente IA local util y amigable. "
            "Responde de forma clara, concisa y en el idioma del usuario."
        ),
        temperature=0.7,
    ),
    "Coder": RolePreset(
        name="Coder",
        system_prompt=(
            "Eres un programador experto senior. Escribe codigo limpio, "
            "eficiente y bien documentado. Explica tu razonamiento tecnico. "
            "Usa buenas practicas y patrones de diseno."
        ),
        temperature=0.2,
    ),
    "Escritor": RolePreset(
        name="Escritor",
        system_prompt=(
            "Eres un escritor creativo y talentoso. Escribe contenido "
            "atractivo, bien estructurado y con estilo literario. "
            "Adapta el tono al contexto solicitado."
        ),
        temperature=0.9,
    ),
    "Traductor": RolePreset(
        name="Traductor",
        system_prompt=(
            "Eres un traductor experto multilingue. Traduce con precision "
            "manteniendo el tono, contexto y estilo del texto original. "
            "Si el idioma destino no se especifica, traduce al espanol."
        ),
        temperature=0.3,
    ),
}

TEMPERATURE_PRESETS: dict[str, float] = {
    "Preciso": 0.2,
    "Balance": 0.5,
    "Normal": 0.7,
    "Creativo": 1.0,
}


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    active_role: str = "General"
    custom_temperature: float | None = None

    @property
    def temperature(self) -> float:
        if self.custom_temperature is not None:
            return self.custom_temperature
        preset = ROLE_PRESETS.get(self.active_role)
        return preset.temperature if preset else 0.7

    @property
    def system_prompt(self) -> str:
        preset = ROLE_PRESETS.get(self.active_role)
        return preset.system_prompt if preset else ROLE_PRESETS["General"].system_prompt


class LLMClient:
    """Synchronous HTTP client for local LLM servers."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.status = AgentStatus.OFFLINE
        self.current_model: str = ""
        self._request_timeout = 30

    def check_status(self) -> AgentStatus:
        """Ping the server and update status."""
        if self._try_ollama():
            return self.status
        if self._try_openai_compat():
            return self.status
        self.status = AgentStatus.OFFLINE
        self.current_model = ""
        return self.status

    def get_models(self) -> list[str]:
        """Get list of available model names."""
        models = self._ollama_models()
        if models:
            return models
        models = self._openai_models()
        return models or []

    def get_running_model(self) -> str:
        """Get the currently loaded model name."""
        running = self._ollama_running()
        if running:
            self.current_model = running
            return running
        return self.current_model

    def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a chat message and return the response text.

        Automatically selects the correct API endpoint.
        """
        sys_prompt = system_prompt or self.config.system_prompt
        temp = temperature if temperature is not None else self.config.temperature

        response = self._ollama_chat(prompt, sys_prompt, temp)
        if response is not None:
            return response

        response = self._openai_chat(prompt, sys_prompt, temp)
        if response is not None:
            return response

        return "[Error] No se pudo conectar con el servidor LLM. Verifica que este ejecutandose."

    def summarize_clipboard(self, text: str) -> str:
        """Send clipboard text with a summarization prompt."""
        prompt = (
            "Resume el siguiente texto de forma concisa y clara. "
            "Preserva los puntos clave:\n\n"
            f"{text}"
        )
        return self.send_message(prompt)

    def analyze_document(self, filename: str, content: str) -> str:
        """Analyze a dropped document."""
        prompt = (
            f"Analiza el siguiente documento ('{filename}') y proporciona:\n"
            "1. Un resumen breve\n"
            "2. Puntos clave\n"
            "3. Categorias o temas relevantes\n\n"
            f"Contenido:\n{content[:8000]}"
        )
        return self.send_message(prompt)

    # ── Private: Ollama API ──────────────────────────────────────────

    def _try_ollama(self) -> bool:
        try:
            data = self._http_get(f"{self.config.base_url}/api/tags")
            if data and "models" in data:
                self.status = AgentStatus.ONLINE
                running = self._ollama_running()
                self.current_model = running or (
                    data["models"][0]["name"] if data["models"] else ""
                )
                return True
        except Exception:
            pass
        return False

    def _ollama_models(self) -> list[str]:
        try:
            data = self._http_get(f"{self.config.base_url}/api/tags")
            if data and "models" in data:
                return [m["name"] for m in data["models"]]
        except Exception:
            pass
        return []

    def _ollama_running(self) -> str:
        try:
            data = self._http_get(f"{self.config.base_url}/api/ps")
            if data and "models" in data and data["models"]:
                return data["models"][0].get("name", "")
        except Exception:
            pass
        return ""

    def _ollama_chat(self, prompt: str, system: str, temperature: float) -> str | None:
        try:
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            }
            data = self._http_post(
                f"{self.config.base_url}/api/chat", payload
            )
            if data and "message" in data:
                return data["message"].get("content", "")
        except Exception as e:
            logger.debug("Ollama chat failed: %s", e)
        return None

    # ── Private: OpenAI-compatible API (LM Studio) ───────────────────

    def _try_openai_compat(self) -> bool:
        base = self.config.base_url
        if "11434" in base:
            alt_url = base.replace("11434", "1234")
        else:
            alt_url = base
        try:
            data = self._http_get(f"{alt_url}/v1/models")
            if data and "data" in data:
                self.status = AgentStatus.ONLINE
                self.config.base_url = alt_url
                if data["data"]:
                    self.current_model = data["data"][0].get("id", "")
                return True
        except Exception:
            pass
        return False

    def _openai_models(self) -> list[str]:
        try:
            data = self._http_get(f"{self.config.base_url}/v1/models")
            if data and "data" in data:
                return [m["id"] for m in data["data"]]
        except Exception:
            pass
        return []

    def _openai_chat(self, prompt: str, system: str, temperature: float) -> str | None:
        try:
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "stream": False,
            }
            data = self._http_post(
                f"{self.config.base_url}/v1/chat/completions", payload
            )
            if data and "choices" in data and data["choices"]:
                return data["choices"][0].get("message", {}).get("content", "")
        except Exception as e:
            logger.debug("OpenAI-compat chat failed: %s", e)
        return None

    # ── Private: HTTP helpers ────────────────────────────────────────

    def _http_get(self, url: str) -> Any:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._request_timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, url: str, payload: dict) -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._request_timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
