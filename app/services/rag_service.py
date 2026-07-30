import os
import pickle
import threading
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.logging_config import get_logger
from app.config.settings import get_settings
from app.models.llm import create_llm
from app.services.document_classifier import _EmbeddingEngine

logger = get_logger(__name__)

_RAG_DIR = Path("data/rag_index")
_FAISS_PATH = _RAG_DIR / "faiss.index"
_META_PATH = _RAG_DIR / "metadata.pkl"
_ID_MAP_PATH = _RAG_DIR / "file_id_map.pkl"
_EMBED_DIM = 384
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 128
_TOP_K = 5

_RAG_PROMPT = """\
Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.

Contexto de los documentos:
{context}

Pregunta: {query}

Instrucciones:
- Si la respuesta está en el contexto, responde de forma clara y concisa.
- Si NO encuentras la respuesta en el contexto, di exactamente: \
"No encontré información sobre eso en tus documentos de ASIORGA."
- No inventes información ni uses conocimiento previo.
- Si es relevante, menciona qué documentos contienen la información.\
"""


class RagVectorStore:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.file_id_map: dict[str, set[int]] = {}
        self._next_id = 0
        self._engine = _EmbeddingEngine()
        os.makedirs(_RAG_DIR, exist_ok=True)

    def build_from_documents(self, documents: list[dict]) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        all_texts: list[str] = []
        all_meta: list[dict] = []

        for doc in documents:
            if not doc["text"].strip():
                continue
            chunks = splitter.split_text(doc["text"])
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                all_texts.append(chunk_text)
                all_meta.append({
                    "file_path": doc["file_path"],
                    "folder": doc["folder"],
                    "text": chunk_text,
                    "chunk_index": i,
                })

        if not all_texts:
            logger.warning("No texts to index")
            return

        embeddings = self._engine.embed(all_texts)
        dim = embeddings.shape[1]
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        for idx_val, meta in zip(ids, all_meta):
            meta["id"] = int(idx_val)
            self.metadata.append(meta)
            fp = meta["file_path"]
            if fp not in self.file_id_map:
                self.file_id_map[fp] = set()
            self.file_id_map[fp].add(int(idx_val))

        self._next_id += len(embeddings)
        self.save()
        logger.info(
            "Built index with %d chunks from %d docs",
            len(all_texts), len(documents),
        )

    def add_file(self, file_path: Path, folder: str, text: str) -> None:
        if not text.strip():
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)

        texts_to_embed: list[str] = []
        metas: list[dict] = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            texts_to_embed.append(chunk_text)
            metas.append({
                "file_path": str(file_path),
                "folder": folder,
                "text": chunk_text,
                "chunk_index": i,
            })

        if not texts_to_embed:
            return

        embeddings = self._engine.embed(texts_to_embed)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

        fp_str = str(file_path)
        for idx_val, meta in zip(ids, metas):
            meta["id"] = int(idx_val)
            self.metadata.append(meta)
            if fp_str not in self.file_id_map:
                self.file_id_map[fp_str] = set()
            self.file_id_map[fp_str].add(int(idx_val))

        self._next_id += len(embeddings)
        self.save()
        logger.info("Indexed %s (%d chunks)", file_path.name, len(texts_to_embed))

    def remove_file(self, file_path: str) -> bool:
        fp = str(file_path)
        ids = self.file_id_map.pop(fp, None)
        if not ids:
            logger.warning("File not found in index: %s", fp)
            return False

        ids_array = np.array(list(ids), dtype=np.int64)
        self.index.remove_ids(ids_array)
        self.metadata = [m for m in self.metadata if m["id"] not in ids]
        self.save()
        logger.info("Removed %s from index (%d chunks)", Path(fp).name, len(ids))
        return True

    def search(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        query_emb = self._engine.embed_single(query)
        query_emb = query_emb.reshape(1, -1)

        distances, indices = self.index.search(query_emb, top_k)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = next((m for m in self.metadata if m["id"] == idx), None)
            if meta:
                results.append({
                    **meta,
                    "distance": float(dist),
                })

        return results

    def query(self, query: str, top_k: int = _TOP_K) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No se encontraron documentos relevantes en ASIORGA."

        context_parts: list[str] = []
        for r in results:
            folder = r.get("folder", "?")
            source = Path(r["file_path"]).name
            context_parts.append(f"[{folder}/{source}]\n{r['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = _RAG_PROMPT.format(context=context, query=query)

        try:
            settings = get_settings()
            llm = create_llm(settings)
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content if hasattr(response, "content") else str(response)

            sources = list(set(r["file_path"] for r in results))
            source_lines = "\n".join(f"- {Path(s).name}" for s in sources[:5])
            return f"{answer}\n\n---\n📄 Fuentes:\n{source_lines}"
        except Exception as e:
            logger.error("LLM query failed: %s", e)
            return f"Contexto recuperado ({len(results)} chunks), pero falló la generación: {e}"

    def save(self) -> None:
        if self.index is None:
            return
        _RAG_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(_FAISS_PATH))
        with open(_META_PATH, "wb") as f:
            pickle.dump({"metadata": self.metadata, "next_id": self._next_id}, f)
        with open(_ID_MAP_PATH, "wb") as f:
            pickle.dump(self.file_id_map, f)
        logger.info("Saved index (%d vectors)", self.index.ntotal)

    def load(self) -> bool:
        if not _FAISS_PATH.exists() or not _META_PATH.exists():
            logger.info("No saved index found")
            return False
        try:
            self.index = faiss.read_index(str(_FAISS_PATH))
            with open(_META_PATH, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self._next_id = data["next_id"]
            if _ID_MAP_PATH.exists():
                with open(_ID_MAP_PATH, "rb") as f:
                    self.file_id_map = pickle.load(f)
            else:
                for m in self.metadata:
                    fp = m["file_path"]
                    if fp not in self.file_id_map:
                        self.file_id_map[fp] = set()
                    self.file_id_map[fp].add(m["id"])
            logger.info("Loaded index (%d vectors)", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("Failed to load index: %s", e)
            return False

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0


_SUPPORTED_EXTS = {
    ".pdf", ".docx", ".txt", ".md",
    ".csv", ".xlsx", ".xls",
    ".jpg", ".jpeg", ".png",
}


class _AsiorgaHandler:
    def __init__(self, indexer: "RagDocumentIndexer") -> None:
        self._indexer = indexer

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in _SUPPORTED_EXTS:
            return
        self._indexer.index_new(file_path)


class RagDocumentIndexer:
    def __init__(self, vector_store: RagVectorStore):
        self._store = vector_store
        self._extractor = None
        self._lock = threading.Lock()
        self._observer: Optional[Any] = None
        self._watcher_started = False

    def _get_extractor(self):
        if self._extractor is None:
            from app.services.document_extractor import DocumentExtractor
            self._extractor = DocumentExtractor()
        return self._extractor

    def index_all(self) -> None:
        with self._lock:
            from app.services.folder_manager import ASIORGA_ROOT, FolderManager
            fm = FolderManager()
            folders = fm.list_destinations()

            total = 0
            for folder in folders:
                folder_name = folder["name"]
                folder_path = ASIORGA_ROOT / folder_name
                if not folder_path.is_dir():
                    continue

                for file_path in sorted(folder_path.iterdir()):
                    if not file_path.is_file():
                        continue
                    if file_path.suffix.lower() not in {
                        ".pdf", ".docx", ".txt", ".md",
                        ".csv", ".xlsx", ".xls",
                        ".jpg", ".jpeg", ".png",
                    }:
                        continue

                    if str(file_path) in self._store.file_id_map:
                        continue

                    text = self._get_extractor().extract(file_path)
                    if text and text.strip():
                        self._store.add_file(file_path, folder_name, text)
                        total += 1

            logger.info("Indexed %d new files from ASIORGA folders", total)

    def index_new(self, file_path: Path) -> None:
        with self._lock:
            fp_str = str(file_path)
            if fp_str in self._store.file_id_map:
                return

            folder = file_path.parent.name
            if not folder:
                folder = "Documentos"

            text = self._get_extractor().extract(file_path)
            if text and text.strip():
                self._store.add_file(file_path, folder, text)
                logger.info("Auto-indexed new file: %s", file_path.name)

    def remove_file(self, file_path: str) -> None:
        with self._lock:
            self._store.remove_file(file_path)

    def start_watcher(self) -> None:
        if self._watcher_started:
            return

        from watchdog.observers import Observer

        try:
            from app.services.folder_manager import ASIORGA_ROOT, FolderManager
            fm = FolderManager()
            folders = fm.list_destinations()

            self._observer = Observer()
            handler = _AsiorgaHandler(self)

            for folder in folders:
                folder_path = ASIORGA_ROOT / folder["name"]
                if folder_path.is_dir():
                    self._observer.schedule(handler, str(folder_path), recursive=False)
                    logger.info("RAG watcher monitoring: %s", folder_path)

            if self._observer._watches:
                self._observer.start()
                self._watcher_started = True
                logger.info("RAG watcher started — watching %d ASIORGA folders", len(folders))
            else:
                self._observer = None
        except Exception as e:
            logger.warning("RAG watcher could not start: %s", e)

    def stop_watcher(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None
            self._watcher_started = False
            logger.info("RAG watcher stopped")

    def sync_with_filesystem(self) -> None:
        with self._lock:
            removed = 0
            for fp in list(self._store.file_id_map.keys()):
                if not Path(fp).exists():
                    self._store.remove_file(fp)
                    removed += 1
            if removed:
                logger.info("Removed %d orphaned files from index", removed)


_instance: Optional[RagDocumentIndexer] = None
_instance_lock = threading.Lock()


def get_rag_indexer() -> RagDocumentIndexer:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                store = RagVectorStore()
                store.load()
                _instance = RagDocumentIndexer(store)
    return _instance


def get_rag_vector_store() -> RagVectorStore:
    return get_rag_indexer()._store
