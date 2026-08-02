import csv
import json
import re
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.config.logging_config import get_logger

logger = get_logger(__name__)


def _json_default(obj):
    """Serialize numpy scalars/arrays for json.dumps."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def _strip_accents(text: str) -> str:
    """Remove combining diacritics (á -> a) for fuzzy name matching."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents and all non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]+", "", _strip_accents(name or "").lower())


def _significant_tokens(name: str) -> set[str]:
    """Meaningful tokens of a filename (no accents, no digits, no particles)."""
    toks = {
        t
        for t in re.split(r"[^a-z0-9]+", _strip_accents(name or "").lower())
        if t
    }
    return {t for t in toks if len(t) > 2 and not t.isdigit()}


def _rename_match(log_name: str, phys_name: str) -> bool:
    """True when most meaningful tokens of the log name appear in the
    physical file name (detects manual moves that also renamed the file)."""
    lt = _significant_tokens(log_name)
    pt = _significant_tokens(phys_name)
    if not lt or not pt:
        return False
    shared = lt & pt
    return len(shared) >= 1 and (len(shared) / len(lt)) >= 0.6

_MODEL_REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_DIM = 384
_DEFAULT_THRESHOLD = 0.35
_MIN_GAP = 0.05
_MIN_CONFIDENCE = 0.40
_SUMMARY_MODEL = "llama3.1:8b"
_SUMMARY_PROMPT = """Extract: document type, main topic, key entities.
Return ONLY the raw summary in the SAME LANGUAGE as the document.
No greetings, introductions, explanations or meta-text of any kind.
Just the facts.

Document text:
{text}"""
_CACHE_DIR = Path("data")
_NPY_CACHE = _CACHE_DIR / "folder_embeddings.npy"
_META_CACHE = _CACHE_DIR / "folder_embeddings.json"
_LOG_FILE = _CACHE_DIR / "classification_log.csv"
_MAX_CHUNKS = 8
_CHUNK_SIZE = 300
_THRESHOLD_WINDOW = 50
_THRESHOLD_PERCENTILE = 25
_KNN_K = 5
_MAX_DOCS_PER_FOLDER = 50

FOLDER_RULES: dict[str, dict] = {
    "Universidad": {
        "filename_patterns": [
            r"^GD_\d+",
            r"(?i)guia_docente",
            r"(?i)tema[\s._-]?\d",
            r"(?i)(tfg|tfm|pfc)",
            r"(?i)(examen|parcial|final)_",
        ],
        "filename_keywords": [
            ("universidad", 5), ("campus", 5), ("asignatura", 5),
            ("guia_docente", 5), ("tema", 3), ("curso", 4), ("alumno", 4),
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
            ("cuatrimestre", 4), ("semestre", 4), ("curso", 3),
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
            ("informe", 3), ("presentacion", 3),
            ("memoria", 3), ("propuesta", 3), ("presupuesto", 4),
            ("reunion", 3), ("meeting", 3), ("cliente", 3),
            ("empresa", 3), ("nomina", 5),
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


@dataclass
class ClassificationResult:
    folder: Optional[str]
    suggested_name: Optional[str]
    confidence: float = 0.0
    method: str = "unknown"
    scores: dict[str, float] = field(default_factory=dict)
    gap: float = 0.0
    threshold_used: float = _DEFAULT_THRESHOLD
    all_scores_raw: dict[str, float] = field(default_factory=dict)
    summary: str = ""


class _EmbeddingEngine:
    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._ready = False

    def _ensure_loaded(self) -> bool:
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
        if not self._ensure_loaded():
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (token_embeddings * mask_expanded).sum(axis=1)
        counts = mask_expanded.sum(axis=1).clip(min=1e-9)
        embeddings = summed / counts

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(b, a)


class FolderEmbeddingCache:
    def __init__(self, npy_path: Path = _NPY_CACHE, meta_path: Path = _META_CACHE) -> None:
        self._npy_path = npy_path
        self._meta_path = meta_path

    def load(self) -> tuple[Optional[list[str]], Optional[np.ndarray], Optional[list[str]], Optional[dict]]:
        if not self._npy_path.exists() or not self._meta_path.exists():
            return None, None, None, None
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            embeddings = np.load(str(self._npy_path))
            folder_names = meta.get("folder_names", [])
            labels = meta.get("labels", [])
            logger.info("[CACHE] Embeddings cargados de cache (%d vectores)", len(labels))
            return folder_names, embeddings, labels, meta
        except Exception as e:
            logger.warning("[CACHE] Error cargando cache: %s", e)
            return None, None, None, None

    def save(
        self, folder_names: list[str], embeddings: np.ndarray, labels: list[str],
        extra_meta: Optional[dict] = None,
    ) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {"folder_names": folder_names, "labels": labels, "version": 3}
        if extra_meta:
            meta.update(extra_meta)
        try:
            self._meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            np.save(str(self._npy_path), embeddings)
            logger.info("[CACHE] Embeddings guardados en cache (%d vectores, %s)",
                        len(labels), str(self._npy_path))
        except Exception as e:
            logger.warning("[CACHE] Error guardando cache: %s", e)

    def invalidate(self) -> None:
        self._npy_path.unlink(missing_ok=True)
        self._meta_path.unlink(missing_ok=True)
        logger.info("[CACHE] Cache de embeddings invalidada")


class ClassificationLog:
    def __init__(self, log_file: Path = _LOG_FILE) -> None:
        self._log_file = log_file
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._log_file.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "filename", "file_path", "predicted_folder",
                    "confidence", "threshold", "gap", "method", "scores_json",
                    "success", "corrected_folder", "summary",
                ])

    def log(self, result: ClassificationResult, file_path: str = "") -> None:
        try:
            with open(self._log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    result.suggested_name or Path(file_path).name,
                    file_path,
                    result.folder or "None",
                    f"{result.confidence:.4f}",
                    f"{result.threshold_used:.4f}",
                    f"{result.gap:.4f}",
                    result.method,
                    json.dumps(result.all_scores_raw, ensure_ascii=False, default=_json_default),
                    "",
                    "",
                    result.summary,
                ])
        except Exception as e:
            logger.warning("[LOG] Error escribiendo log: %s", e)

    def record_correction(self, file_path: str, corrected_folder: str) -> Optional[tuple[str, str]]:
        """Register a manual correction.

        Matches the physical file against the classification log using the
        destination filename (stored in the log), its original basename, and
        normalized/token-based fallbacks so manual moves keep being detected
        even when the file was renamed.

        Returns:
            (predicted_folder, summary) tuple if a correction was recorded,
            or None if no correction was needed.
        """
        try:
            if not self._log_file.exists():
                return None

            basename = Path(file_path).name
            norm_basename = _normalize_name(basename)

            rows: list[list[str]] = []
            candidates: list[list[str]] = []
            with open(self._log_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    rows.append(header)
                for row in reader:
                    rows.append(row)
                    if len(row) >= 11 and not row[9]:
                        candidates.append(row)

            def _matches(row: list[str], token_fallback: bool) -> bool:
                log_path = row[2]
                log_filename = row[1]
                log_basename = Path(log_path).name
                if log_basename == basename or log_filename == basename:
                    return True
                norm_log = _normalize_name(log_basename)
                norm_file = _normalize_name(log_filename)
                if norm_log == norm_basename or norm_file == norm_basename:
                    return True
                if token_fallback:
                    if _rename_match(log_filename, basename):
                        return True
                    if _rename_match(log_basename, basename):
                        return True
                return False

            predicted_folder = None
            summary = ""
            found_exact = False

            # Pass 1: exact / normalized matches
            for row in candidates:
                if _matches(row, token_fallback=False):
                    found_exact = True
                    predicted = row[3]
                    if predicted != corrected_folder:
                        row[9] = "False"
                        row[10] = corrected_folder
                        predicted_folder = predicted
                        summary = row[11] if len(row) > 11 else ""

            # Pass 2: token-based fallback for renamed files, only if nothing
            # matched by exact/normalized name (avoids false positives).
            if predicted_folder is None and not found_exact:
                for row in candidates:
                    if _matches(row, token_fallback=True):
                        predicted = row[3]
                        if predicted != corrected_folder:
                            row[9] = "False"
                            row[10] = corrected_folder
                            predicted_folder = predicted
                            summary = row[11] if len(row) > 11 else ""
                        break

            if predicted_folder is None:
                return None

            with open(self._log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return (predicted_folder, summary)
        except Exception as e:
            logger.warning("[LOG] Error registrando correccion: %s", e)
            return None

    def get_orphaned_entries(
        self, active_basenames: set[str],
    ) -> list[tuple[str, str, str, str]]:
        """Find log entries whose file no longer exists in any ASIORGA folder.

        Marks found entries as deleted so they are not returned again.
        Only processes entries not already corrected or deleted.

        Returns:
            List of (file_path, predicted_folder, summary, filename) tuples.
        """
        entries: list[tuple[str, str, str, str]] = []
        try:
            if not self._log_file.exists():
                return entries
            rows: list[list[str]] = []
            with open(self._log_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    rows.append(header)
                for row in reader:
                    if len(row) < 4:
                        rows.append(row)
                        continue
                    if row[9]:
                        rows.append(row)
                        continue
                    log_basename = Path(row[2]).name if row[2] else ""
                    log_filename = row[1] if len(row) > 1 else ""
                    still_active = (
                        (log_basename in active_basenames)
                        or (log_filename in active_basenames)
                    )
                    if not still_active:
                        row[9] = "Deletion"
                        summary = row[11] if len(row) > 11 else ""
                        filename = row[1]
                        entries.append((row[2], row[3], summary, filename))
                    rows.append(row)
            with open(self._log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return entries
        except Exception as e:
            logger.warning("[LOG] Error buscando entradas huerfanas: %s", e)
            return entries

    def get_stats(self) -> dict:
        total = 0
        correct = 0
        corrected = 0
        by_method: dict[str, int] = {}
        if not self._log_file.exists():
            return {"total": 0, "accuracy": 0.0, "corrected": 0, "by_method": {}}
        try:
            with open(self._log_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) < 11:
                        continue
                    total += 1
                    method = row[7]
                    by_method[method] = by_method.get(method, 0) + 1
                    if row[9] == "True":
                        correct += 1
                    elif row[9] == "False":
                        corrected += 1
        except Exception:
            pass
        return {
            "total": total,
            "accuracy": correct / max(total, 1),
            "corrected": corrected,
            "by_method": by_method,
        }


_SYNTHETIC_CENTROID_PROMPT = """\
Generate 15 representative document titles or short excerpts (5–25 words each) \
that belong in the folder "{name}". Each one must be a plausible real document.
Description: {description}
{seed_context}
Return one per line. Do NOT number them. Do NOT use markdown."""


class DocumentClassifier:
    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._engine = _EmbeddingEngine()
        self._summary_llm = self._init_summary_llm()
        self._folder_names: list[str] = []
        self._knn_embeddings: Optional[np.ndarray] = None
        self._knn_labels: list[str] = []
        self._threshold = _DEFAULT_THRESHOLD
        self._min_gap = _MIN_GAP
        self._recent_scores: deque[float] = deque(maxlen=_THRESHOLD_WINDOW)
        self._cache = FolderEmbeddingCache()
        self._log = ClassificationLog()

    @property
    def classification_log(self) -> ClassificationLog:
        return self._log

    def set_llm(self, llm) -> None:
        self._llm = llm

    def _init_summary_llm(self):
        try:
            return ChatOllama(
                model=_SUMMARY_MODEL,
                base_url="http://localhost:11434",
                temperature=0.0,
                num_ctx=2048,
            )
        except Exception:
            logger.warning("[CLASSIFY] No se pudo crear summary LLM")
            return None

    def reload_folders(self) -> None:
        from app.services.folder_manager import FolderManager
        folders = FolderManager().list_destinations()
        if folders:
            self.load_folders(folders)

    def load_folders(self, folders: list[dict]) -> None:
        if not folders:
            return

        self._folder_names = []
        for f in folders:
            name = f["name"]
            if name in (DEFAULT_FOLDER, "Fotos"):
                continue
            self._folder_names.append(name)

        cached_names, cached_emb, cached_labels, cached_meta = self._cache.load()
        if cached_names == self._folder_names and cached_emb is not None and cached_labels is not None:
            self._knn_embeddings = cached_emb
            self._knn_labels = cached_labels
            logger.info("[CLASSIFY] Cargado de cache (%d vectores)", len(self._knn_labels))
            return

        # Fast path initialization from seeds
        folder_map = {f["name"]: f for f in folders}
        all_embs = []
        all_labels = []

        start = time.time()
        for name in self._folder_names:
            f = folder_map.get(name, {})
            description = f.get("description", name)
            seed_texts = f.get("seed_texts") or []

            local_texts = (
                [f"{name}: {t}" for t in seed_texts]
                if seed_texts
                else [f"{name}: {description}"]
            )
            embs = self._engine.embed(local_texts)
            for emb in embs:
                norm = np.linalg.norm(emb)
                if norm > 1e-9:
                    emb = emb / norm
                all_embs.append(emb)
                all_labels.append(name)

        if all_embs:
            self._knn_embeddings = np.stack(all_embs, axis=0)
            self._knn_labels = all_labels
            self._cache.save(self._folder_names, self._knn_embeddings, self._knn_labels)
        
        elapsed = (time.time() - start) * 1000
        logger.info(
            "[CLASSIFY] %d vectores inicializados en %.0fms (fast path)",
            len(all_labels), elapsed,
        )

    def _compute_dynamic_threshold(self) -> float:
        if len(self._recent_scores) < 10:
            return self._threshold
        arr = np.array(list(self._recent_scores))
        dyn = float(np.percentile(arr, _THRESHOLD_PERCENTILE))
        return max(dyn, _DEFAULT_THRESHOLD * 0.8)

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return [""]
        words = text.split()
        if len(words) <= _CHUNK_SIZE:
            return [text]
        chunks: list[str] = []
        step = len(words) // _MAX_CHUNKS if len(words) > _MAX_CHUNKS * _CHUNK_SIZE else _CHUNK_SIZE
        step = max(step, _CHUNK_SIZE // 2)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + _CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
            if len(chunks) >= _MAX_CHUNKS:
                break
        if not chunks:
            chunks = [text[:1000]]
        return chunks

    def _compute_confidence(
        self, best_score: float, second_score: float, gap: float, folder_probs: dict[str, float]
    ) -> float:
        gap_conf = np.clip(gap / 0.15, 0.0, 1.0)
        score_conf = np.clip((best_score - 0.25) / 0.4, 0.0, 1.0)

        probs = np.array(list(folder_probs.values()))
        probs = probs / (np.sum(probs) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(probs) + 1e-9)
        entropy_conf = 1.0 - (entropy / max(max_entropy, 1e-9)) if max_entropy > 0 else 0.5

        combined = 0.35 * score_conf + 0.35 * gap_conf + 0.30 * entropy_conf
        return float(np.clip(combined, 0.0, 1.0))

    def _extract_classification_summary(self, content: str) -> Optional[str]:
        if self._summary_llm is None or not content:
            return None
        text = content[:2000]
        prompt = _SUMMARY_PROMPT.format(text=text)
        try:
            response = self._summary_llm.invoke([HumanMessage(content=prompt)])
            summary = (response.content or "").strip()
            summary = self._clean_summary(summary)
            if len(summary) >= 20:
                return summary
            return None
        except Exception as e:
            logger.warning("[CLASSIFY] Error extrayendo resumen: %s", e)
            return None

    @staticmethod
    def _clean_summary(text: str) -> str:
        text = re.sub(
            r"^(?:here\s+is|here'?s|based\s+on|i\s+(?:think|believe|would\s+say)"
            r"|this\s+(?:appears|seems|looks\s+like|document|text)"
            r"|the\s+(?:document|text|content|following)"
            r"|analizando|a\s+continuaci[óo]n|te\s+presento|te\s+muestro"
            r"|resumen|summary|classification|an[áa]lisis)"
            r"[\s,:;.!-]*",
            "", text, flags=re.I,
        ).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_document_embedding(
        self, content: str, summary: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        # Keeps original behavior: returns single averaged vector if no summary
        if not content or not content.strip():
            return None
        if summary is None:
            summary = self._extract_classification_summary(content)
        if summary:
            emb = self._engine.embed_single(summary)
            return emb
        chunks = self._chunk_text(content.strip())
        if not chunks:
            return None
        chunk_embs = self._engine.embed(chunks)
        return np.mean(chunk_embs, axis=0)
        
    def get_chunk_embeddings(self, content: str) -> Optional[np.ndarray]:
        if not content or not content.strip():
            return None
        chunks = self._chunk_text(content.strip())
        if not chunks:
            return None
        return self._engine.embed(chunks)

    def rebuild_centroids(self, folders: list[dict], extract_fn=None) -> bool:
        """Reconstruye el índice k-NN desde cero escaneando los archivos físicos.
        Mantenemos el nombre 'rebuild_centroids' por compatibilidad con folder_monitor.
        """
        if not folders:
            return False

        from app.services.folder_manager import ASIORGA_ROOT

        new_names: list[str] = []
        all_embs: list[np.ndarray] = []
        all_labels: list[str] = []

        for f in folders:
            name = f["name"]
            if name in (DEFAULT_FOLDER, "Fotos"):
                continue
            new_names.append(name)

            seed_texts = f.get("seed_texts") or []
            description = f.get("description", name)

            if seed_texts:
                seeds_emb = self._engine.embed([f"{name}: {t}" for t in seed_texts])
                for emb in seeds_emb:
                    all_embs.append(emb)
                    all_labels.append(name)
            else:
                desc_emb = self._engine.embed_single(f"{name}: {description}")
                all_embs.append(desc_emb)
                all_labels.append(name)
            
            # Limit the number of physical files to embed per folder to avoid huge matrices
            folder_doc_count = 0
            if extract_fn:
                folder_path = ASIORGA_ROOT / name
                if folder_path.is_dir():
                    # Sort by modification time to get newest first
                    try:
                        children = sorted(folder_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    except Exception:
                        children = sorted(folder_path.iterdir())
                        
                    for child in children:
                        if not child.is_file():
                            continue
                        if folder_doc_count >= _MAX_DOCS_PER_FOLDER:
                            break
                        try:
                            content = extract_fn(child) or ""
                            if content.strip():
                                emb = self.get_document_embedding(content)
                                if emb is not None:
                                    all_embs.append(emb)
                                    all_labels.append(name)
                                    folder_doc_count += 1
                        except Exception:
                            continue

        if not all_embs:
            return False

        self._folder_names = new_names
        self._knn_embeddings = np.stack(all_embs, axis=0)
        
        # Normalize
        norms = np.linalg.norm(self._knn_embeddings, axis=1, keepdims=True).clip(min=1e-9)
        self._knn_embeddings = self._knn_embeddings / norms
        
        self._knn_labels = all_labels
        self._cache.save(self._folder_names, self._knn_embeddings, self._knn_labels)
        logger.info(
            "[CLASSIFY] Índice k-NN reconstruido (%d carpetas, %d vectores)",
            len(new_names), len(all_labels),
        )
        return True

    def add_document_embedding(self, folder_name: str, document_embedding: np.ndarray) -> bool:
        if folder_name not in self._folder_names:
            logger.warning("[CLASSIFY] Carpeta '%s' no encontrada para k-NN", folder_name)
            return False
            
        norm = np.linalg.norm(document_embedding)
        if norm > 1e-9:
            document_embedding = document_embedding / norm
            
        if self._knn_embeddings is None:
            self._knn_embeddings = np.array([document_embedding])
            self._knn_labels = [folder_name]
        else:
            self._knn_embeddings = np.vstack([self._knn_embeddings, document_embedding])
            self._knn_labels.append(folder_name)
            
        # Prune if too many
        indices = [i for i, lbl in enumerate(self._knn_labels) if lbl == folder_name]
        if len(indices) > _MAX_DOCS_PER_FOLDER:
            # Remove the oldest for this folder (the first one found)
            idx_to_remove = indices[0]
            self._knn_embeddings = np.delete(self._knn_embeddings, idx_to_remove, axis=0)
            self._knn_labels.pop(idx_to_remove)
            
        self._cache.save(self._folder_names, self._knn_embeddings, self._knn_labels)
        logger.info("[CLASSIFY] Añadido vector a '%s' (total %d en k-NN)", folder_name, len(self._knn_labels))
        return True
        
    def remove_document_embedding(self, folder_name: str, document_embedding: np.ndarray) -> bool:
        if self._knn_embeddings is None or folder_name not in self._knn_labels:
            return False
            
        norm = np.linalg.norm(document_embedding)
        if norm > 1e-9:
            document_embedding = document_embedding / norm
            
        # Find the most similar embedding in this folder to remove it
        folder_indices = [i for i, lbl in enumerate(self._knn_labels) if lbl == folder_name]
        if not folder_indices:
            return False
            
        folder_embs = self._knn_embeddings[folder_indices]
        sims = document_embedding @ folder_embs.T
        best_match_idx = int(np.argmax(sims))
        best_match_sim = sims[best_match_idx]
        
        if best_match_sim > 0.95:  # Very close match
            global_idx_to_remove = folder_indices[best_match_idx]
            self._knn_embeddings = np.delete(self._knn_embeddings, global_idx_to_remove, axis=0)
            self._knn_labels.pop(global_idx_to_remove)
            self._cache.save(self._folder_names, self._knn_embeddings, self._knn_labels)
            logger.info("[CLASSIFY] Vector removido de '%s'", folder_name)
            return True
            
        return False
        
    def update_folder_centroid(self, folder_name: str, document_embedding: np.ndarray, remove: bool = False) -> bool:
        """Backward compatibility con folder_monitor.py"""
        if remove:
            return self.remove_document_embedding(folder_name, document_embedding)
        else:
            return self.add_document_embedding(folder_name, document_embedding)

    def _predict_knn(self, query_embs: np.ndarray) -> tuple[str, float, float, float, dict[str, float]]:
        """
        Calcula KNN con Max-Pooling sobre chunks.
        query_embs: Shape (num_chunks, 384)
        Retorna: (best_folder, best_score, gap, confidence, all_scores_raw)
        """
        if self._knn_embeddings is None or len(self._knn_labels) == 0:
            return DEFAULT_FOLDER, 0.0, 0.0, 0.0, {}

        # 1. Similitud de cada chunk contra toda la base de datos
        # Shape: (num_chunks, num_db_vectors)
        sim_matrix = query_embs @ self._knn_embeddings.T

        # 2. Score máximo de chunk para cada vector en la DB (Max-Pooling)
        # Shape: (num_db_vectors,)
        db_scores = np.max(sim_matrix, axis=0)

        # 3. K-Nearest Neighbors
        k = min(_KNN_K, len(db_scores))
        top_k_indices = np.argsort(db_scores)[::-1][:k]
        
        # Agrupar scores por carpeta usando los K mejores
        folder_scores = defaultdict(list)
        for idx in top_k_indices:
            lbl = self._knn_labels[idx]
            folder_scores[lbl].append(db_scores[idx])
            
        # El score final de cada carpeta es el promedio de sus scores en el top-K
        # Si una carpeta tiene más votos, su score será más alto, o penalizamos las que no están?
        # Mejor: usar el score máximo o ponderado de la carpeta en el top-K
        final_scores = {folder: np.mean(scores) for folder, scores in folder_scores.items()}
        
        # Rellenar con 0 para el resto
        for name in self._folder_names:
            if name not in final_scores:
                final_scores[name] = 0.0
                
        sorted_folders = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        best_folder, best_score = sorted_folders[0]
        second_score = sorted_folders[1][1] if len(sorted_folders) > 1 else 0.0
        gap = best_score - second_score
        
        # Softmax over all top max scores per folder for confidence
        # Instead of just top-K, get the max score for EACH folder across the whole DB
        all_folder_max = defaultdict(float)
        for i, lbl in enumerate(self._knn_labels):
            all_folder_max[lbl] = max(all_folder_max[lbl], db_scores[i])
        for name in self._folder_names:
            if name not in all_folder_max:
                all_folder_max[name] = 0.0
                
        # Confidence is calculated using the global max per folder
        global_sorted = sorted(all_folder_max.values(), reverse=True)
        g_best = global_sorted[0] if global_sorted else 0.0
        g_second = global_sorted[1] if len(global_sorted) > 1 else 0.0
        g_gap = g_best - g_second
        
        confidence = self._compute_confidence(g_best, g_second, g_gap, all_folder_max)
        
        return (
            best_folder,
            float(best_score),
            float(g_gap),
            confidence,
            {k: float(v) for k, v in all_folder_max.items()},
        )


    def classify(
        self,
        content: str,
        filename: str,
        file_type: str,
        folder_descriptions: str,
        file_size: str = "unknown",
        source_path: str = "",
    ) -> ClassificationResult:
        log_path = source_path or filename
        logger.info(
            "[CLASSIFY] Inicio — archivo: %s | content_len: %d | path: %s",
            filename, len(content or ""), log_path,
        )

        if file_type in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
            suggested = self._suggest_name(filename, content, "Fotos")
            result = ClassificationResult(
                folder="Fotos",
                suggested_name=suggested,
                confidence=0.95,
                method="image",
                scores={"Fotos": 1.0},
                gap=0.0,
                threshold_used=self._threshold,
                all_scores_raw={"Fotos": 1.0},
                summary="",
            )
            self._log.log(result, file_path=log_path)
            logger.info("[CLASSIFY] %s -> Fotos (imagen)", filename)
            return result

        # Step 1: Keywords
        kw_folder, kw_suggested = self._classify_by_keywords(filename, content)
        if kw_folder != DEFAULT_FOLDER:
            conf = 0.5
            scores_map = {}
            if self._folder_names:
                scores_map = self._keyword_score_map(filename, content)
            result = ClassificationResult(
                folder=kw_folder,
                suggested_name=kw_suggested,
                confidence=conf,
                method="keyword",
                scores=scores_map,
                gap=0.0,
                threshold_used=self._threshold,
                all_scores_raw=scores_map,
                summary="",
            )
            doc_emb = self.get_document_embedding(content)
            if doc_emb is not None:
                self.add_document_embedding(kw_folder, doc_emb)
            self._log.log(result, file_path=log_path)
            logger.info(
                "[CLASSIFY] %s -> %s (keyword) nombre='%s'",
                filename, kw_folder, kw_suggested,
            )
            return result

        # Step 2: Two-Pass Embedding Classification
        if self._knn_embeddings is not None and content and content.strip():
            start = time.time()
            dynamic_th = self._compute_dynamic_threshold()
            
            # --- FASE 1: Chunks Max-Pooling (Sin LLM) ---
            chunk_embs = self.get_chunk_embeddings(content)
            best_folder = DEFAULT_FOLDER
            best_score = 0.0
            gap = 0.0
            confidence = 0.0
            all_scores_raw = {}
            method = "knn_chunks"
            summary = ""
            
            if chunk_embs is not None:
                best_folder, best_score, gap, confidence, all_scores_raw = self._predict_knn(chunk_embs)
                
            # Evaluamos si la Fase 1 es suficientemente segura.
            # Además de score y gap, exigimos una confianza mínima para
            # no comprometer la clasificación con poca evidencia.
            gap_ok = gap >= self._min_gap
            is_confident = (
                best_score >= dynamic_th
                and gap_ok
                and confidence >= _MIN_CONFIDENCE
            )

            # --- FASE 2: LLM Summary (Si hay dudas) ---
            if not is_confident and self._summary_llm is not None:
                logger.info("[CLASSIFY] Fase 1 (%.3f conf=%.2f) insuficiente. Iniciando Fase 2 (LLM Summary)...", best_score, confidence)
                summary = self._extract_classification_summary(content) or ""
                if summary:
                    summary_emb = self._engine.embed_single(summary)
                    if summary_emb is not None:
                        # Pasamos summary_emb como un solo "chunk"
                        best_folder, best_score, gap, confidence, all_scores_raw = self._predict_knn(np.array([summary_emb]))
                        method = "knn_summary"
                        is_confident = (
                            best_score >= dynamic_th
                            and gap >= self._min_gap
                            and confidence >= _MIN_CONFIDENCE
                        )

            elapsed = (time.time() - start) * 1000

            self._recent_scores.append(best_score)
            
            logger.info(
                "[CLASSIFY] %s (%.0fms): mejor='%s' score=%.3f gap=%.3f "
                "confianza=%.2f th=%.3f",
                method, elapsed, best_folder, best_score, gap,
                confidence, dynamic_th,
            )

            if is_confident:
                suggested = self._suggest_name(filename, content, best_folder)
                result = ClassificationResult(
                    folder=best_folder,
                    suggested_name=suggested,
                    confidence=confidence,
                    method=method,
                    scores={best_folder: best_score, "gap": gap},
                    gap=gap,
                    threshold_used=dynamic_th,
                    all_scores_raw=all_scores_raw,
                    summary=summary,
                )
                
                # Para añadir a la BD k-NN, usamos el vector final (el resumen si lo hay, o el promedio de chunks)
                doc_emb = self.get_document_embedding(content, summary=summary)
                if doc_emb is not None:
                    self.add_document_embedding(best_folder, doc_emb)
                    
                self._log.log(result, file_path=log_path)
                logger.info(
                    "[CLASSIFY] %s -> %s (%s=%.3f gap=%.3f conf=%.2f) nombre='%s'",
                    filename, best_folder, method, best_score, gap, confidence, suggested,
                )
                return result
            else:
                logger.info(
                    "[CLASSIFY] Score %.3f (gap %.3f) insuficiente tras todas las fases (th=%.3f)",
                    best_score, gap, dynamic_th,
                )

        # Fallback
        suggested = self._suggest_name(filename, content, DEFAULT_FOLDER)
        logger.info("[CLASSIFY] %s -> %s (fallback)", filename, DEFAULT_FOLDER)
        result = ClassificationResult(
            folder=DEFAULT_FOLDER,
            suggested_name=suggested,
            confidence=0.2,
            method="fallback",
            scores={},
            gap=0.0,
            threshold_used=self._threshold,
            all_scores_raw={},
            summary="",
        )
        self._log.log(result, file_path=log_path)
        return result

    def _keyword_score_map(self, filename: str, content: str) -> dict[str, float]:
        name_lower = filename.lower()
        content_lower = (content or "").lower()[:3000]
        scores: dict[str, float] = defaultdict(float)
        for folder, rules in FOLDER_RULES.items():
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename):
                    scores[folder] += 10.0
            for keyword, weight in rules.get("filename_keywords", []):
                if re.search(r"(?:^|[\W_])" + re.escape(keyword) + r"(?=[\W_]|$)", name_lower):
                    scores[folder] += weight
            for keyword, weight in rules.get("content_keywords", []):
                if re.search(r"\b" + re.escape(keyword) + r"\b", content_lower):
                    scores[folder] += weight
        if not scores:
            return {DEFAULT_FOLDER: 0.0}
        total = max(sum(scores.values()), 1)
        return {k: v / total for k, v in scores.items()}

    def _classify_by_keywords(self, filename: str, content: str = "") -> tuple[str, str]:
        name_lower = filename.lower()
        content_lower = (content or "").lower()[:3000]

        scores: dict[str, float] = defaultdict(float)

        for folder, rules in FOLDER_RULES.items():
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename):
                    scores[folder] += 10.0
            for keyword, weight in rules.get("filename_keywords", []):
                if re.search(r"(?:^|[\W_])" + re.escape(keyword) + r"(?=[\W_]|$)", name_lower):
                    scores[folder] += weight
            for keyword, weight in rules.get("content_keywords", []):
                if re.search(r"\b" + re.escape(keyword) + r"\b", content_lower):
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
        stem = Path(filename).stem
        suffix = Path(filename).suffix

        if any(c.isalpha() for c in stem) and not stem.isdigit():
            clean = re.sub(r'[^\w\s-]', '', stem).strip()
            clean = re.sub(r'\s+', '_', clean)
            if clean and len(clean) >= 3:
                return f"{clean}{suffix}"

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


def hashlib_md5(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()
