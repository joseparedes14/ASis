"""Tests for the document classifier service.

Uses frozen test data to verify embedding, keyword fallback,
confidence scoring, chunking, and classification log behavior.
"""

import numpy as np
import pytest

from app.services.document_classifier import (
    DEFAULT_FOLDER,
    ClassificationLog,
    ClassificationResult,
    DocumentClassifier,
    FolderEmbeddingCache,
    _cosine_similarity,
)

FROZEN_FOLDER_CONFIG = [
    {"name": "Documentos", "description": "Varios miscelanea",
     "seed_texts": ["notas variadas"]},
    {"name": "Facturas", "description": "Facturas y recibos",
     "seed_texts": ["factura con IVA", "recibo de compra"]},
    {"name": "Trabajo", "description": "Laboral y empresa",
     "seed_texts": ["nomina mensual", "contrato laboral"]},
    {"name": "Universidad", "description": "Universidad y estudios",
     "seed_texts": ["examen parcial", "guia docente"]},
    {"name": "Personal", "description": "DNI y documentos personales",
     "seed_texts": ["dni documento", "certificado"]},
]

FROZEN_FACTURA_CONTENT = (
    "FACTURA ELECTRONICA N\xba 2024-00123\n"
    "Proveedor: Suministros SL\n"
    "Base imponible: 1.200,00 \u20ac\n"
    "IVA 21%: 252,00 \u20ac\n"
    "Total a pagar: 1.452,00 \u20ac\n"
    "NIF/CIF: B-12345678\n"
    "Fecha de emisi\xf3n: 15/03/2024"
)

FROZEN_NOMINA_CONTENT = (
    "NOMINA DEL MES DE MARZO 2024\n"
    "Empresa: Tecnolog\xeda SA\n"
    "Salario base: 2.500,00 \u20ac\n"
    "Complementos: 300,00 \u20ac\n"
    "IRPF: 15%\n"
    "Seguridad Social: 250,00 \u20ac\n"
    "Total liquido: 2.335,00 \u20ac"
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear folder embedding cache before each test."""
    cache = FolderEmbeddingCache()
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def classifier():
    clf = DocumentClassifier()
    clf.load_folders(FROZEN_FOLDER_CONFIG)
    return clf


# ── Chunking ──────────────────────────────────────────────────────


def test_chunk_short_text(classifier):
    chunks = classifier._chunk_text("Hola mundo")
    assert len(chunks) == 1
    assert chunks[0] == "Hola mundo"


def test_chunk_long_text(classifier):
    long_text = "palabra " * 2000
    chunks = classifier._chunk_text(long_text)
    assert 1 <= len(chunks) <= 8
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_empty_text(classifier):
    assert classifier._chunk_text("") == [""]
    assert classifier._chunk_text("   ") == [""]


# ── Cosine similarity ─────────────────────────────────────────────


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = _cosine_similarity(a, b)
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([[0.0, 1.0]])
    sims = _cosine_similarity(a, b)
    assert np.isclose(sims[0], 0.0)


# ── Keywords / fallback ───────────────────────────────────────────


def test_keyword_classify_factura(classifier):
    folder, name = classifier._classify_by_keywords(
        "factura_2024_001.pdf", FROZEN_FACTURA_CONTENT
    )
    assert folder == "Facturas"
    assert name.endswith(".pdf")


def test_keyword_classify_empty(classifier):
    folder, name = classifier._classify_by_keywords("random_file.txt", "")
    assert folder == DEFAULT_FOLDER


def test_keyword_classify_nomina_by_content(classifier):
    folder, name = classifier._classify_by_keywords(
        "documento.pdf", FROZEN_NOMINA_CONTENT
    )
    assert folder in ("Trabajo", "Personal")


# ── Full classification pipeline (with ONNX if available) ──────────


def test_classify_image(classifier):
    result = classifier.classify(
        content="",
        filename="vacaciones.jpg",
        file_type=".jpg",
        folder_descriptions="",
    )
    assert result.folder == "Fotos"
    assert result.method == "image"
    assert result.confidence > 0.9


def test_classify_empty_content(classifier):
    result = classifier.classify(
        content="",
        filename="test.pdf",
        file_type=".pdf",
        folder_descriptions="",
    )
    assert result.folder == DEFAULT_FOLDER
    assert result.method in ("keyword", "fallback")


def test_classify_factura_content(classifier):
    result = classifier.classify(
        content=FROZEN_FACTURA_CONTENT,
        filename="doc.pdf",
        file_type=".pdf",
        folder_descriptions="",
    )
    assert result.folder == "Facturas"
    assert result.method in ("embedding", "keyword")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.all_scores_raw) > 0


def test_classify_result_dataclass(classifier):
    result = classifier.classify(
        content="contenido de prueba generico",
        filename="test.pdf",
        file_type=".pdf",
        folder_descriptions="",
    )
    assert isinstance(result, ClassificationResult)
    assert isinstance(result.folder, (str, type(None)))
    assert isinstance(result.suggested_name, (str, type(None)))
    assert isinstance(result.confidence, float)
    assert isinstance(result.method, str)
    assert isinstance(result.scores, dict)
    assert isinstance(result.all_scores_raw, dict)


# ── Confidence scoring ────────────────────────────────────────────


def test_confidence_perfect():
    clf = DocumentClassifier()
    all_scores = np.array([0.95, 0.30, 0.25, 0.20, 0.15])
    conf = clf._compute_confidence(0.95, 0.30, all_scores)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5


def test_confidence_low():
    clf = DocumentClassifier()
    all_scores = np.array([0.30, 0.29, 0.28, 0.27, 0.26])
    conf = clf._compute_confidence(0.30, 0.29, all_scores)
    assert conf < 0.4


# ── Classification log ────────────────────────────────────────────


def test_classification_log(tmp_path):
    log_path = tmp_path / "log.csv"
    log = ClassificationLog(log_path)

    result = ClassificationResult(
        folder="Facturas",
        suggested_name="factura.pdf",
        confidence=0.85,
        method="embedding",
        scores={"Facturas": 0.7, "second": 0.3, "gap": 0.4},
        gap=0.4,
        threshold_used=0.35,
        all_scores_raw={"Facturas": 0.7, "Documentos": 0.3},
    )

    log.log(result, file_path="/tmp/test.pdf")
    stats = log.get_stats()
    assert stats["total"] == 1
    assert stats["by_method"].get("embedding") == 1


def test_classification_log_stats(tmp_path):
    log_path = tmp_path / "log.csv"
    log = ClassificationLog(log_path)

    for i in range(5):
        result = ClassificationResult(
            folder="Facturas" if i % 2 == 0 else "Trabajo",
            suggested_name=f"file_{i}.pdf",
            confidence=0.7,
            method="embedding",
            scores={},
            gap=0.1,
            threshold_used=0.35,
            all_scores_raw={},
        )
        log.log(result, file_path=f"/tmp/file_{i}.pdf")

    stats = log.get_stats()
    assert stats["total"] == 5
    assert stats["by_method"].get("embedding") == 5


# ── Embedding cache ───────────────────────────────────────────────


def test_embedding_cache_roundtrip(tmp_path):
    cache = FolderEmbeddingCache(
        npy_path=tmp_path / "test.npy",
        meta_path=tmp_path / "test.json",
    )

    names = ["Facturas", "Trabajo"]
    embs = np.random.randn(2, 384).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms
    cache.save(names, embs)

    loaded_names, loaded_embs, loaded_meta = cache.load()
    assert loaded_names == names
    assert loaded_embs is not None
    assert np.allclose(loaded_embs, embs)
    assert loaded_meta is not None


def test_embedding_cache_invalidate(tmp_path):
    cache = FolderEmbeddingCache(
        npy_path=tmp_path / "test.npy",
        meta_path=tmp_path / "test.json",
    )

    names = ["Facturas"]
    embs = np.random.randn(1, 384).astype(np.float32)
    cache.save(names, embs)
    assert cache._npy_path.exists()
    assert cache._meta_path.exists()

    cache.invalidate()
    assert not cache._npy_path.exists()
    assert not cache._meta_path.exists()


# ── Folder loading ────────────────────────────────────────────────


def test_load_folders_with_seed_texts(classifier):
    assert classifier._folder_embeddings is not None
    assert classifier._folder_embeddings.shape[0] == len(FROZEN_FOLDER_CONFIG) - 1
    assert classifier._folder_embeddings.shape[1] == 384


def test_load_empty_folders():
    clf = DocumentClassifier()
    clf.load_folders([])
    assert clf._folder_embeddings is None
    assert clf._folder_names == []


# ── Suggest name ──────────────────────────────────────────────────


def test_suggest_name_preserves_descriptive_name(classifier):
    name = classifier._suggest_name("informe_ventas_2024.pdf", "", "Trabajo")
    assert name == "informe_ventas_2024.pdf"


def test_suggest_name_fallback(classifier):
    name = classifier._suggest_name("12345.pdf", "", "Facturas")
    assert "Facturas" in name


def test_suggest_name_from_content(classifier):
    content = "RESUMEN EJECUTIVO: El proyecto ha finalizado con exito."
    name = classifier._suggest_name("12345.pdf", content, "Trabajo")
    assert "RESUMEN" in name or "resumen" in name.lower()
