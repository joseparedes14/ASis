"""
Document classification service using LLM.

Analyzes extracted document content and classifies it into
the appropriate destination folder based on folder descriptions.
"""

import concurrent.futures
from typing import Optional

from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Timeout for LLM classification (seconds)
CLASSIFICATION_TIMEOUT = 30

CLASSIFICATION_PROMPT = """\
Eres un asistente de clasificación de documentos. Analiza el siguiente contenido \
y clasifícalo en una de las carpetas destino disponibles.

## Carpetas destino disponibles:
{folders}

## Archivo a clasificar:
- Nombre: {filename}
- Tipo: {file_type}
- Tamaño: {file_size}

## Contenido extraído (primeros 2000 caracteres):
{content}

## Instrucciones:
1. Analiza el contenido del archivo.
2. Determina qué carpetas destino coincide mejor con el tipo de documento.
3. Responde ÚNICAMENTE con el nombre exacto de la carpeta destino (sin paths, sin explicaciones).
4. Si el contenido no encaja en ninguna carpeta, responde con el nombre de la carpeta más \
general disponible (ej: "Documentos").

Respuesta:"""


class DocumentClassifier:
    """Classifies documents using the configured LLM.

    Sends extracted content to the LLM with the list of available
    destination folders and their descriptions, and receives back
    the recommended folder name.
    """

    def __init__(self, llm=None) -> None:
        """Initialize the classifier.

        Args:
            llm: LangChain LLM instance. If None, will use a simple
                 keyword-based fallback classifier.
        """
        self._llm = llm

    def set_llm(self, llm) -> None:
        """Set or update the LLM instance."""
        self._llm = llm

    def classify(
        self,
        content: str,
        filename: str,
        file_type: str,
        folder_descriptions: str,
        file_size: str = "unknown",
    ) -> Optional[str]:
        """Classify a document into a destination folder.

        Args:
            content: Extracted text content of the document.
            filename: Original filename.
            file_type: File extension/type.
            folder_descriptions: Formatted string of available folders.
            file_size: File size as string.

        Returns:
            Destination folder name, or None if classification failed.
        """
        if not content or not content.strip():
            logger.warning("Empty content for %s, using fallback", filename)
            return self._fallback_classify(filename)

        if self._llm is None:
            logger.info("No LLM available, using fallback classifier")
            return self._fallback_classify(filename)

        try:
            # Truncate content for LLM context
            truncated = content[:2000]
            if len(content) > 2000:
                truncated += "\n... [contenido truncado]"

            prompt = CLASSIFICATION_PROMPT.format(
                folders=folder_descriptions,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                content=truncated,
            )

            # Call LLM with timeout to prevent hanging
            from langchain_core.messages import HumanMessage

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._llm.invoke, [HumanMessage(content=prompt)])
                response = future.result(timeout=CLASSIFICATION_TIMEOUT)

            result = response.content.strip()

            # Validate the result is a valid folder name
            if self._validate_folder_name(result, folder_descriptions):
                logger.info("Classified %s → %s", filename, result)
                return result
            else:
                logger.warning(
                    "LLM returned invalid folder '%s' for %s, using fallback",
                    result,
                    filename,
                )
                return self._fallback_classify(filename)

        except concurrent.futures.TimeoutError:
            logger.warning(
                "LLM classification timed out for %s after %ds, using fallback",
                filename,
                CLASSIFICATION_TIMEOUT,
            )
            return self._fallback_classify(filename)
        except Exception as e:
            logger.error("Classification failed for %s: %s", filename, e)
            return self._fallback_classify(filename)

    def _validate_folder_name(self, name: str, folder_descriptions: str) -> bool:
        """Validate that the folder name exists in the available folders."""
        # Extract folder names from the descriptions string
        for line in folder_descriptions.split("\n"):
            if line.startswith("- "):
                folder_name = line.split(":")[0].replace("- ", "").strip()
                if folder_name.lower() == name.lower():
                    return True
        return False

    def _fallback_classify(self, filename: str) -> str:
        """Simple keyword-based fallback classifier.

        Used when LLM is unavailable or classification fails.
        """
        name_lower = filename.lower()

        # Image files → Fotos
        if any(ext in name_lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
            return "Fotos"

        # Common invoice/bill keywords
        invoice_keywords = [
            "factura", "invoice", "receipt", "recibo", "ticket",
            "compra", "pago", "bill", "cargo",
        ]
        if any(kw in name_lower for kw in invoice_keywords):
            return "Facturas"

        # Work-related keywords
        work_keywords = [
            "informe", "report", "presentación", "presentation",
            "proyecto", "project", "trabajo", "work",
        ]
        if any(kw in name_lower for kw in work_keywords):
            return "Trabajo"

        # Default
        return "Documentos"
