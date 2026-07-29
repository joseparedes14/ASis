"""Offline evaluation of the document classifier against ASIORGA structure.

Usage:
    python scripts/eval_classifier.py

Iterates over files in each ASIORGA folder, treats the folder name as
ground truth, runs the classifier, and reports accuracy, confusion
matrix, and per-folder metrics.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.logging_config import get_logger
from app.services.document_classifier import DocumentClassifier
from app.services.document_extractor import DocumentExtractor

logger = get_logger("eval_classifier")


def load_folders_from_config() -> list[dict]:
    config_path = Path("config/destination_folders.json")
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("folders", [])


def collect_ground_truth(asiorga_root: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for folder_path in asiorga_root.iterdir():
        if not folder_path.is_dir():
            continue
        ground_truth = folder_path.name
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                samples.append((file_path, ground_truth))
    return samples


def main() -> None:
    asiorga_root = Path.home() / "Desktop" / "ASIorga"
    if not asiorga_root.exists():
        logger.error("ASIORGA root not found at %s", asiorga_root)
        logger.info("Create the folder structure first by running the app.")
        return

    samples = collect_ground_truth(asiorga_root)
    if not samples:
        logger.warning("No files found in ASIORGA folders for evaluation.")
        return

    folders = load_folders_from_config()

    classifier = DocumentClassifier()
    classifier.load_folders(folders)
    extractor = DocumentExtractor()

    results: list[dict] = []
    correct = 0
    confusion: dict[tuple[str, str], int] = Counter()

    for file_path, ground_truth in samples:
        content = extractor.extract(file_path) or ""
        suffix = file_path.suffix.lower()

        result = classifier.classify(
            content=content,
            filename=file_path.name,
            file_type=suffix,
            folder_descriptions="",
            file_size=f"{file_path.stat().st_size / 1024:.1f} KB",
        )

        predicted = result.folder or "Documentos"
        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1

        confusion[(ground_truth, predicted)] += 1

        results.append({
            "file": str(file_path),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": result.confidence,
            "method": result.method,
            "scores": result.all_scores_raw,
        })

    total = len(results)
    accuracy = correct / max(total, 1)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Evaluacion del Clasificador de Documentos")
    print(f"  Archivos: {total}  Correctos: {correct}  Accuracy: {accuracy:.2%}")
    print(f"{sep}")

    print("\n  {:<20} {:>6} {:>9} {:>9} {:>11} {:>10}".format(
        "Carpeta", "Total", "Correctos", "Accuracy", "Conf. Media", "Metodo"))
    print(f"  {'-' * 66}")

    by_folder: dict[str, dict] = {}
    for r in results:
        gt = r["ground_truth"]
        if gt not in by_folder:
            by_folder[gt] = {"total": 0, "correct": 0, "conf_sum": 0.0, "methods": Counter()}
        by_folder[gt]["total"] += 1
        by_folder[gt]["correct"] += 1 if r["correct"] else 0
        by_folder[gt]["conf_sum"] += r["confidence"]
        by_folder[gt]["methods"][r["method"]] += 1

    for folder in sorted(by_folder.keys()):
        d = by_folder[folder]
        acc = d["correct"] / max(d["total"], 1)
        avg_conf = d["conf_sum"] / max(d["total"], 1)
        main_method = d["methods"].most_common(1)[0][0] if d["methods"] else "-"
        print("  {:<20} {:>6} {:>9} {:>8.1%} {:>10.3f} {:>10}".format(
            folder, d["total"], d["correct"], acc, avg_conf, main_method))

    print("\n  Resultados por metodo:")
    method_stats: dict[str, dict] = {}
    for r in results:
        m = r["method"]
        if m not in method_stats:
            method_stats[m] = {"total": 0, "correct": 0}
        method_stats[m]["total"] += 1
        if r["correct"]:
            method_stats[m]["correct"] += 1
    for m in sorted(method_stats.keys()):
        d = method_stats[m]
        acc = d["correct"] / max(d["total"], 1)
        print(f"    {m:<15} {d['correct']:>4}/{d['total']:<4} ({acc:.1%})")

    print("\n  Matriz de confusion:")
    all_labels = sorted(set(k[0] for k in confusion) | set(k[1] for k in confusion))
    header = "    " + "{:<18}".format("") + "".join("{:<14}".format(lb) for lb in all_labels)
    print(header)
    for gt in all_labels:
        row = f"    {gt:<18}"
        for pred in all_labels:
            count = confusion.get((gt, pred), 0)
            row += f"{count:<14}"
        print(row)

    print("\n  Archivos mal clasificados:")
    for r in results:
        if not r["correct"]:
            score_str = json.dumps(
                {k: f"{v:.3f}" for k, v in r.get("scores", {}).items()},
                ensure_ascii=False,
            )
            print(f"    \u2717 {r['file']}")
            print(f"      Real: {r['ground_truth']} \u2192 Predicho: {r['predicted']} "
                  f"(conf={r['confidence']:.2f}, metodo={r['method']})")
            print(f"      Scores: {score_str}")

    print(f"\n{sep}\n")

    stats = classifier.classification_log.get_stats()
    print(f"  Log stats: {stats}")


if __name__ == "__main__":
    main()
