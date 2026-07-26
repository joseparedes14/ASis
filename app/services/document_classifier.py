"""
Document classification service using embeddings.

Uses a local ONNX embedding model (all-MiniLM-L6-v2, 384-dim) to compute
semantic similarity between document content and destination folder descriptions.
Falls back to weighted keyword scoring if embeddings are unavailable.
"""

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.config.logging_config import get_logger

logger = get_logger(__name__)

# Model repo on HuggingFace (ONNX format, no PyTorch needed)
_MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
_MODEL_FILES = ["onnx/model.onnx", "tokenizer.json"]
_EMBED_DIM = 384
_DEFAULT_THRESHOLD = 0.35


# ── Keyword rules for fallback ───────────────────────────────────────
FOLDER_RULES: dict[str, dict] = {
    "Universidad": {
        "filename_patterns": [
            r"^GD_\d+",
            r"(?i)guia_docente",
            r"(?i)tema\d+",
            r"(?i)(tfg|tfm|pfc)",
            r"(?i)(examen|parcial|final)_",
        ],
        "filename_keywords": [
            ("universidad", 5), ("campus", 5), ("asignatura", 5),
            ("guia_docente", 5), ("tema", 3), ("alumno", 4),
            ("matricula", 4), ("calificacion", 4), ("profesor", 3),
            ("facultad", 3), ("grado", 3), ("master", 3),
            ("clase", 3), ("apuntes", 3), ("ejercicio", 3),
        ],
        "content_keywords": [
            ("universidad", 5), ("university", 5), ("campus", 4),
            ("alumno", 4), ("matricula", 4), ("expediente academico", 5),
            ("calificacion", 4), ("asignatura", 5), ("guia docente", 5),
            ("programa asignatura", 5), ("temario", 4),
            ("profesor", 3), ("titulacion", 4), ("facultad", 4),
            ("cuatrimestre", 4), ("semestre", 4),
            ("examen", 4), ("trabajo final", 4),
            ("tfm", 5), ("tfg", 5), ("pfc", 5),
            ("credito", 3), ("ects", 4), ("plan de estudios", 4),
        ],
    },
    "Facturas": {
        "filename_patterns": [
            r"(?i)factura[_\-]?\d",
            r"(?i)(fac|inv|bill|receipt)[_\-]?\d",
            r"(?i)ticket[_\-]?\d",
        ],
        "filename_keywords": [
            ("factura", 5), ("invoice", 5), ("recibo", 5),
            ("ticket", 4), ("compra", 3),
        ],
        "content_keywords": [
            ("factura", 5), ("invoice", 5), ("recibo", 5),
            ("ticket", 4), ("base imponible", 5), ("cuota", 4),
            ("iva", 5), ("importe", 4), ("total a pagar", 5),
            ("factura electronica", 5), ("numero de factura", 5),
            ("nif/cif", 4),
        ],
    },
    "Trabajo": {
        "filename_patterns": [
            r"(?i)(informe|report|memoria)[_\-]?\d",
            r"(?i)(ppt|pptx|presentacion)",
        ],
        "filename_keywords": [
            ("informe", 4), ("report", 4), ("presentacion", 4),
            ("memoria", 3), ("propuesta", 3), ("presupuesto", 4),
            ("reunion", 3), ("meeting", 3),
        ],
        "content_keywords": [
            ("informe", 3), ("presentacion", 3), ("presentacion", 3),
            ("memoria", 3), ("propuesta", 3), ("presupuesto", 4),
            ("reunion", 3), ("meeting", 3), ("cliente", 3),
            ("empresa", 3), ("nomina", 5), ("nomina", 5),
            ("contrato laboral", 5), ("salario", 4), ("sueldo", 4),
        ],
    },
    "Personal": {
        "filename_patterns": [
            r"(?i)(dni|nie|pasaporte)",
            r"(?i)certificado[_\-]",
            r"(?i)(seguro|insurance)[_\-]",
        ],
        "filename_keywords": [
            ("contrato", 4), ("certificado", 4), ("seguro", 4),
            ("impuesto", 4), ("declaracion", 4),
        ],
        "content_keywords": [
            ("contrato", 4), ("certificado", 4), ("dni", 5),
            ("nie", 5), ("pasaporte", 5), ("licencia", 4),
            ("seguro", 4), ("impuesto", 4), ("declaracion", 4),
            ("tributario", 5), ("hacienda", 5), ("ayuntamiento", 4),
            ("registro civil", 5), ("nacimiento", 4),
        ],
    },
}

DEFAULT_FOLDER = "Documentos"


class _EmbeddingEngine:
    """Lightweight ONNX embedding engine using HuggingFace models.

    Downloads model files on first use and caches them locally.
    No PyTorch required — uses onnxruntime + tokenizers directly.
    """

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._ready = False

    def _ensure_loaded(self) -> bool:
        """Lazy-load model files on first use."""
        if self._ready:
            return True

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=_MODEL_REPO, filename="onnx/model.onnx")
            tokenizer_path = hf_hub_download(repo_id=_MODEL_REPO, filename="tokenizer.json")

            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._session = ort.InferenceSession(model_path)
            self._ready = True
            logger.info("[EMBED] Modelo ONNX cargado: %s", _MODEL_REPO)
            return True

        except Exception as e:
            logger.error("[EMBED] Error cargando modelo ONNX: %s", e)
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Returns:
            numpy array of shape (len(texts), 384)
        """
        if not self._ensure_loaded():
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)

        # Tokenize
        encoded = self._tokenizer.encode_batch(texts)

        # Prepare ONNX inputs
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        # Create token type ids (all zeros for single sentences)
        token_type_ids = np.zeros_like(input_ids)

        # Run inference
        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        # Mean pooling over sequence length
        token_embeddings = outputs[0]  # (batch, seq_len, 384)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (token_embeddings * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        embeddings = summed / counts

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.embed([text])[0]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and matrix b.

    Args:
        a: shape (384,)
        b: shape (N, 384)

    Returns:
        array of shape (N,) with similarity scores.
    """
    return np.dot(b, a)


class DocumentClassifier:
    """Classifies documents using embedding similarity with keyword fallback.

    Primary method: embed document content, compare against pre-embedded
    folder descriptions via cosine similarity. Instant and reliable.

    Fallback: weighted keyword scoring from filename + content patterns.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm  # kept for interface compatibility, not used
        self._engine = _EmbeddingEngine()
        self._folder_names: list[str] = []
        self._folder_embeddings: Optional[np.ndarray] = None
        self._threshold = _DEFAULT_THRESHOLD
        self._min_gap = 0.04  # best must beat 2nd-best by this margin

    def set_llm(self, llm) -> None:
        self._llm = llm

    def load_folders(self, folders: list[dict]) -> None:
        """Pre-embed destination folder descriptions.

        Args:
            folders: list of {"name": str, "description": str} dicts
        """
        if not folders:
            return

        texts = [f"{f['name']}: {f['description']}" for f in folders]
        self._folder_names = [f["name"] for f in folders]

        start = time.time()
        self._folder_embeddings = self._engine.embed(texts)
        elapsed = (time.time() - start) * 1000

        logger.info(
            "[CLASSIFY] %d carpetas embebidas en %.0fms",
            len(folders), elapsed,
        )

    def classify(
        self,
        content: str,
        filename: str,
        file_type: str,
        folder_descriptions: str,
        file_size: str = "unknown",
    ) -> Tuple[Optional[str], Optional[str]]:
        """Classify a document and suggest a clean filename.

        Uses embedding similarity first, falls back to keyword scoring.

        Returns:
            Tuple of (Destination folder name, Suggested filename).
        """
        logger.info(
            "[CLASSIFY] Inicio — archivo: %s | content_len: %d",
            filename, len(content or ""),
        )

        # Fast path: images
        if file_type in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
            suggested = self._suggest_name(filename, content, "Fotos")
            logger.info("[CLASSIFY] %s -> Fotos (imagen)", filename)
            return "Fotos", suggested

        # ── Primary: embedding classification ────────────────────────
        if self._folder_embeddings is not None and content and content.strip():
            snippet = content.strip()[:500]
            start = time.time()
            doc_emb = self._engine.embed_single(snippet)
            sims = _cosine_similarity(doc_emb, self._folder_embeddings)
            elapsed = (time.time() - start) * 1000

            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_folder = self._folder_names[best_idx]

            sorted_idx = np.argsort(sims)[::-1]
            second_score = float(sims[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
            gap = best_score - second_score

            logger.info(
                "[CLASSIFY] Embeddings (%.0fms): mejor='%s' score=%.3f gap=%.3f | todas=%s",
                elapsed, best_folder, best_score, gap,
                {n: f"{s:.3f}" for n, s in zip(self._folder_names, sims)},
            )

            if best_score >= self._threshold and gap >= self._min_gap:
                suggested = self._suggest_name(filename, content, best_folder)
                logger.info(
                    "[CLASSIFY] %s -> %s (embed=%.3f gap=%.3f) nombre='%s'",
                    filename, best_folder, best_score, gap, suggested,
                )
                return best_folder, suggested
            else:
                logger.info(
                    "[CLASSIFY] Score %.3f (gap %.3f) insuficiente, usando keywords",
                    best_score, gap,
                )

        # ── Fallback: keyword scoring ────────────────────────────────
        return self._classify_by_keywords(filename, content)

    def _classify_by_keywords(
        self, filename: str, content: str = ""
    ) -> Tuple[str, str]:
        """Weighted keyword-based classification (fallback)."""
        name_lower = filename.lower()
        content_lower = (content or "").lower()[:3000]

        scores: dict[str, float] = defaultdict(float)

        for folder, rules in FOLDER_RULES.items():
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename):
                    scores[folder] += 10.0

            for keyword, weight in rules.get("filename_keywords", []):
                if keyword in name_lower:
                    scores[folder] += weight

            for keyword, weight in rules.get("content_keywords", []):
                if keyword in content_lower:
                    scores[folder] += weight

        if scores:
            best_folder = max(scores, key=scores.get)
            best_score = scores[best_folder]
            if best_score >= 5.0:
                suggested = self._suggest_name(filename, content, best_folder)
                logger.info(
                    "[CLASSIFY] Keywords: %s -> %s (score=%.1f)",
                    filename, best_folder, best_score,
                )
                return best_folder, suggested

        suggested = self._suggest_name(filename, content, DEFAULT_FOLDER)
        logger.info("[CLASSIFY] %s -> %s (fallback)", filename, DEFAULT_FOLDER)
        return DEFAULT_FOLDER, suggested

    def _suggest_name(self, filename: str, content: str, folder: str) -> str:
        """Suggest a clean, descriptive filename."""
        stem = Path(filename).stem
        suffix = Path(filename).suffix

        # If filename has descriptive text, clean and keep it
        if any(c.isalpha() for c in stem) and not stem.isdigit():
            clean = re.sub(r'[^\w\s-]', '', stem).strip()
            clean = re.sub(r'\s+', '_', clean)
            if clean and len(clean) >= 3:
                return f"{clean}{suffix}"

        # Filename is numeric/generic — try to extract from content
        if content:
            skip_words = {
                "http", "www.", "page", "página", "created by",
                "microsoft", "adobe", "pdf version", "outputintent",
            }
            for line in content.split("\n"):
                line = line.strip()
                if len(line) < 5 or line.isdigit():
                    continue
                if any(s in line.lower() for s in skip_words):
                    continue
                clean = re.sub(r'[^\w\s-]', '', line[:60]).strip()
                clean = re.sub(r'\s+', '_', clean)
                if clean and len(clean) >= 3:
                    return f"{clean}{suffix}"

        return f"{folder}_{stem}{suffix}"
