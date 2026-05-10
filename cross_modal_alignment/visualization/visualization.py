#!/usr/bin/env python3
"""UMAP-only representation analysis for CLIP, BLIP, and BLIP-2.

The script reuses retrieval embeddings. It does not reload any model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_OUTPUT_DIRS = {
    "clip": "outputs_clip",
    "blip": "outputs_blip",
    "blip2": "outputs_blip2",
}


@dataclass
class RetrievalArtifacts:
    model: str
    image_embeds: np.ndarray
    text_embeds: np.ndarray
    metadata: dict[str, Any]


@dataclass
class PairBatch:
    image_indices: np.ndarray
    caption_indices: np.ndarray
    image_vectors: np.ndarray
    text_vectors: np.ndarray
    pair_cosine: np.ndarray
    query_indices: list[int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP-only embedding visualization.")
    parser.add_argument("--retrieval-root", type=Path, default=Path("cross_modal_alignment/retrieval"))
    parser.add_argument("--instances", type=Path, default=Path("~/data/coco/annotations/instances_val2017.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("cross_modal_alignment/visualization/outputs"))
    parser.add_argument("--models", default="clip,blip,blip2")
    parser.add_argument("--num-points", type=int, default=1000, help="Total modality points. 1000 means 500 image-caption pairs.")
    parser.add_argument("--semantic-per-class", type=int, default=300)
    parser.add_argument("--caption-choice", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=10)
    return parser.parse_args()


def parse_models(value: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    for model in models:
        if model not in MODEL_OUTPUT_DIRS:
            raise ValueError(f"Unsupported model: {model}")
    return models


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, eps)


def load_artifacts(retrieval_root: Path, model: str) -> RetrievalArtifacts:
    root = retrieval_root / MODEL_OUTPUT_DIRS[model]
    embedding_path = root / "embeddings.npz"
    metadata_path = root / "metadata.json"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing {embedding_path}. Run retrieval first.")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run retrieval first.")

    arrays = np.load(embedding_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return RetrievalArtifacts(
        model=model,
        image_embeds=arrays["image_embeds"].astype(np.float32),
        text_embeds=arrays["text_embeds"].astype(np.float32),
        metadata=metadata,
    )


def check_metadata_compatible(reference: dict[str, Any], candidate: dict[str, Any], model: str) -> None:
    if reference["image_ids"] != candidate["image_ids"]:
        raise ValueError(f"{model} image_ids do not match reference metadata.")
    if reference["captions"] != candidate["captions"]:
        raise ValueError(f"{model} captions do not match reference metadata.")


def first_caption_indices(metadata: dict[str, Any]) -> np.ndarray:
    return np.asarray([int(caption_ids[0]) for caption_ids in metadata["image_to_caption_indices"]], dtype=np.int64)


def choose_caption_indices(
    metadata: dict[str, Any],
    image_indices: np.ndarray,
    caption_choice: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if caption_choice == "first":
        first = first_caption_indices(metadata)
        return first[image_indices]

    chosen = []
    for image_index in image_indices:
        chosen.append(int(rng.choice(metadata["image_to_caption_indices"][int(image_index)])))
    return np.asarray(chosen, dtype=np.int64)


def make_pair_batch(artifacts: RetrievalArtifacts, image_indices: np.ndarray, caption_indices: np.ndarray) -> PairBatch:
    text_vectors = l2_normalize(artifacts.text_embeds[caption_indices])
    raw_image_vectors = artifacts.image_embeds[image_indices]
    query_indices: list[int | None]

    if raw_image_vectors.ndim == 3:
        image_query_vectors = l2_normalize(raw_image_vectors)
        query_scores = np.einsum("nqd,nd->nq", image_query_vectors, text_vectors)
        best_queries = query_scores.argmax(axis=1)
        image_vectors = raw_image_vectors[np.arange(len(image_indices)), best_queries]
        pair_cosine = query_scores[np.arange(len(image_indices)), best_queries]
        query_indices = [int(item) for item in best_queries]
    else:
        image_vectors = raw_image_vectors
        image_norm = l2_normalize(image_vectors)
        pair_cosine = np.sum(image_norm * text_vectors, axis=1)
        query_indices = [None for _ in image_indices]

    return PairBatch(
        image_indices=image_indices,
        caption_indices=caption_indices,
        image_vectors=l2_normalize(image_vectors.astype(np.float32)),
        text_vectors=text_vectors.astype(np.float32),
        pair_cosine=pair_cosine.astype(np.float32),
        query_indices=query_indices,
    )


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def load_main_object_labels(instances_path: Path) -> dict[int, str]:
    if not instances_path.exists():
        raise FileNotFoundError(f"Missing {instances_path}")
    with instances_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    categories = {int(item["id"]): str(item["name"]) for item in raw["categories"]}
    best: dict[int, tuple[float, str]] = {}
    for ann in raw["annotations"]:
        image_id = int(ann["image_id"])
        area = float(ann.get("area", 0.0))
        label = categories[int(ann["category_id"])]
        if image_id not in best or area > best[image_id][0]:
            best[image_id] = (area, label)
    return {image_id: label for image_id, (_, label) in best.items()}


def top_two_semantic_indices(
    metadata: dict[str, Any],
    labels_by_image_id: dict[int, str],
    per_class: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    image_ids = [int(item) for item in metadata["image_ids"]]
    labels = [labels_by_image_id.get(image_id, "unknown") for image_id in image_ids]
    counts = Counter(label for label in labels if label != "unknown")
    top_labels = [label for label, _ in counts.most_common(2)]
    if len(top_labels) < 2:
        raise ValueError("Could not find two semantic categories from COCO instances.")

    chosen = []
    for label in top_labels:
        candidates = np.asarray([idx for idx, item in enumerate(labels) if item == label], dtype=np.int64)
        if len(candidates) == 0:
            continue
        sample_size = min(per_class, len(candidates))
        chosen.extend(rng.choice(candidates, size=sample_size, replace=False).tolist())
    return top_labels, np.asarray(sorted(chosen), dtype=np.int64)


def run_umap(vectors: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    return reducer.fit_transform(vectors).astype(np.float32)


def save_3d_html(path: Path, rows: list[dict[str, Any]], color: str, symbol: str, title: str) -> None:
    import plotly.express as px

    fig = px.scatter_3d(
        rows,
        x="x",
        y="y",
        z="z",
        color=color,
        symbol=symbol,
        hover_data=["image_id", "caption", "image_path", "main_object", "query_index"],
        title=title,
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.78})
    fig.update_layout(template="plotly_white")
    fig.write_html(str(path), include_plotlyjs="cdn")


def make_point_rows(
    artifacts: RetrievalArtifacts,
    pair_batch: PairBatch,
    coords: np.ndarray,
    labels_by_image_id: dict[int, str],
    semantic_override: list[str] | None = None,
) -> list[dict[str, Any]]:
    n = len(pair_batch.image_indices)
    rows = []
    for i in range(n):
        image_index = int(pair_batch.image_indices[i])
        caption_index = int(pair_batch.caption_indices[i])
        image_id = int(artifacts.metadata["image_ids"][image_index])
        main_object = labels_by_image_id.get(image_id, "unknown")
        semantic_label = semantic_override[i] if semantic_override is not None else main_object
        shared = {
            "model": artifacts.model,
            "pair_index": i,
            "image_index": image_index,
            "caption_index": caption_index,
            "image_id": image_id,
            "image_path": artifacts.metadata["image_paths"][image_index],
            "caption": artifacts.metadata["captions"][caption_index],
            "main_object": main_object,
            "semantic_label": semantic_label,
            "query_index": pair_batch.query_indices[i],
            "pair_cosine": float(pair_batch.pair_cosine[i]),
        }
        rows.append(
            {
                **shared,
                "modality": "image",
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            }
        )
        rows.append(
            {
                **shared,
                "modality": "text",
                "x": float(coords[n + i, 0]),
                "y": float(coords[n + i, 1]),
                "z": float(coords[n + i, 2]),
            }
        )
    return rows


def cross_modal_nn_rate(vectors: np.ndarray, labels: list[str], metric: str) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(vectors)
    nearest = nn.kneighbors(vectors, return_distance=False)[:, 1]
    label_array = np.asarray(labels)
    return float(np.mean(label_array[nearest] != label_array) * 100.0)


def silhouette_or_nan(vectors: np.ndarray, labels: list[str], metric: str) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(vectors, labels, metric=metric))


def semantic_knn_purity(vectors: np.ndarray, labels: list[str], k: int, metric: str) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(vectors) <= k:
        return float("nan")
    label_array = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(vectors)
    neighbors = nn.kneighbors(vectors, return_distance=False)[:, 1:]
    purities = []
    for idx, neighbor_indices in enumerate(neighbors):
        purities.append(float(np.mean(label_array[neighbor_indices] == label_array[idx])))
    return float(np.mean(purities) * 100.0)


def modality_metrics(pair_batch: PairBatch, coords: np.ndarray) -> dict[str, float]:
    vectors = np.vstack([pair_batch.image_vectors, pair_batch.text_vectors])
    labels = ["image"] * len(pair_batch.image_indices) + ["text"] * len(pair_batch.image_indices)
    return {
        "original_modality_silhouette": silhouette_or_nan(vectors, labels, metric="cosine"),
        "original_cross_modal_nearest_neighbor_rate": cross_modal_nn_rate(vectors, labels, metric="cosine"),
        "umap_modality_silhouette": silhouette_or_nan(coords, labels, metric="euclidean"),
        "umap_cross_modal_nearest_neighbor_rate": cross_modal_nn_rate(coords, labels, metric="euclidean"),
    }


def semantic_metrics(pair_batch: PairBatch, coords: np.ndarray, labels: list[str], k: int) -> dict[str, float]:
    image_coords = coords[: len(pair_batch.image_indices)]
    text_coords = coords[len(pair_batch.image_indices) :]
    all_vectors = np.vstack([pair_batch.image_vectors, pair_batch.text_vectors])
    all_coords = np.vstack([image_coords, text_coords])
    all_labels = labels + labels
    return {
        "original_image_semantic_silhouette": silhouette_or_nan(pair_batch.image_vectors, labels, metric="cosine"),
        "original_text_semantic_silhouette": silhouette_or_nan(pair_batch.text_vectors, labels, metric="cosine"),
        "original_all_semantic_silhouette": silhouette_or_nan(all_vectors, all_labels, metric="cosine"),
        "original_image_knn_purity": semantic_knn_purity(pair_batch.image_vectors, labels, k, metric="cosine"),
        "original_text_knn_purity": semantic_knn_purity(pair_batch.text_vectors, labels, k, metric="cosine"),
        "umap_image_semantic_silhouette": silhouette_or_nan(image_coords, labels, metric="euclidean"),
        "umap_text_semantic_silhouette": silhouette_or_nan(text_coords, labels, metric="euclidean"),
        "umap_all_semantic_silhouette": silhouette_or_nan(all_coords, all_labels, metric="euclidean"),
        "umap_image_knn_purity": semantic_knn_purity(image_coords, labels, k, metric="euclidean"),
        "umap_text_knn_purity": semantic_knn_purity(text_coords, labels, k, metric="euclidean"),
    }


def pair_cosine_analysis(
    artifacts_by_model: dict[str, RetrievalArtifacts],
    image_indices: np.ndarray,
    caption_indices: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    rows = []
    summary = {}
    reference = next(iter(artifacts_by_model.values()))
    for model, artifacts in artifacts_by_model.items():
        pairs = make_pair_batch(artifacts, image_indices, caption_indices)
        summary[model] = summarize_values(pairs.pair_cosine)
        for i, score in enumerate(pairs.pair_cosine):
            image_index = int(image_indices[i])
            caption_index = int(caption_indices[i])
            rows.append(
                {
                    "model": model,
                    "pair_index": i,
                    "image_index": image_index,
                    "caption_index": caption_index,
                    "image_id": int(reference.metadata["image_ids"][image_index]),
                    "image_path": reference.metadata["image_paths"][image_index],
                    "caption": reference.metadata["captions"][caption_index],
                    "pair_cosine": float(score),
                    "query_index": pairs.query_indices[i],
                }
            )
    write_csv(output_dir / "pair_cosine_by_pair.csv", rows)
    write_json(output_dir / "pair_cosine_metrics.json", summary)
    return summary


def run_model_umap(
    artifacts: RetrievalArtifacts,
    pair_batch: PairBatch,
    labels_by_image_id: dict[int, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vectors = np.vstack([pair_batch.image_vectors, pair_batch.text_vectors])
    coords = run_umap(vectors, args)
    rows = make_point_rows(artifacts, pair_batch, coords, labels_by_image_id)
    write_csv(output_dir / "modality_umap_points.csv", rows)
    save_3d_html(
        output_dir / "modality_umap_3d.html",
        rows,
        color="modality",
        symbol="modality",
        title=f"{artifacts.model.upper()} UMAP 3D: image/text modality",
    )
    return modality_metrics(pair_batch, coords)


def run_semantic_umap(
    artifacts: RetrievalArtifacts,
    pair_batch: PairBatch,
    labels: list[str],
    labels_by_image_id: dict[int, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vectors = np.vstack([pair_batch.image_vectors, pair_batch.text_vectors])
    coords = run_umap(vectors, args)
    rows = make_point_rows(artifacts, pair_batch, coords, labels_by_image_id, semantic_override=labels)
    write_csv(output_dir / "semantic_top2_umap_points.csv", rows)
    save_3d_html(
        output_dir / "semantic_top2_umap_3d.html",
        rows,
        color="semantic_label",
        symbol="modality",
        title=f"{artifacts.model.upper()} UMAP 3D: top-2 semantic categories",
    )

    image_rows = [row for row in rows if row["modality"] == "image"]
    text_rows = [row for row in rows if row["modality"] == "text"]
    save_3d_html(
        output_dir / "semantic_top2_image_umap_3d.html",
        image_rows,
        color="semantic_label",
        symbol="semantic_label",
        title=f"{artifacts.model.upper()} UMAP 3D: top-2 semantic image points",
    )
    save_3d_html(
        output_dir / "semantic_top2_text_umap_3d.html",
        text_rows,
        color="semantic_label",
        symbol="semantic_label",
        title=f"{artifacts.model.upper()} UMAP 3D: top-2 semantic text points",
    )
    return semantic_metrics(pair_batch, coords, labels, args.knn_k)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models)
    rng = np.random.default_rng(args.seed)

    artifacts_by_model = {model: load_artifacts(args.retrieval_root, model) for model in models}
    reference = artifacts_by_model[models[0]]
    for model in models[1:]:
        check_metadata_compatible(reference.metadata, artifacts_by_model[model].metadata, model)

    labels_by_image_id = load_main_object_labels(args.instances)
    num_images = len(reference.metadata["image_ids"])
    all_image_indices = np.arange(num_images, dtype=np.int64)
    all_caption_indices = first_caption_indices(reference.metadata)

    # Question 1: same-pair cosine across all images using the first COCO caption.
    pair_summary = pair_cosine_analysis(artifacts_by_model, all_image_indices, all_caption_indices, args.output_dir)

    # Question 2: 1000 points = 500 image-caption pairs, shared across all models.
    if args.num_points % 2 != 0:
        raise ValueError("--num-points must be even because each pair contributes one image point and one text point.")
    modality_pair_count = min(args.num_points // 2, num_images)
    modality_image_indices = np.sort(rng.choice(num_images, size=modality_pair_count, replace=False).astype(np.int64))
    modality_caption_indices = choose_caption_indices(reference.metadata, modality_image_indices, args.caption_choice, rng)

    # Question 3: top-2 COCO main-object categories, shared across all models.
    semantic_labels, semantic_image_indices = top_two_semantic_indices(
        reference.metadata,
        labels_by_image_id,
        args.semantic_per_class,
        rng,
    )
    semantic_caption_indices = choose_caption_indices(reference.metadata, semantic_image_indices, args.caption_choice, rng)
    image_ids = [int(reference.metadata["image_ids"][int(i)]) for i in semantic_image_indices]
    semantic_point_labels = [labels_by_image_id.get(image_id, "unknown") for image_id in image_ids]

    write_json(
        args.output_dir / "selection.json",
        {
            "seed": args.seed,
            "models": models,
            "caption_choice": args.caption_choice,
            "umap": {
                "n_components": 3,
                "n_neighbors": args.umap_n_neighbors,
                "min_dist": args.umap_min_dist,
                "metric": "cosine",
            },
            "pair_cosine_pairs": int(len(all_image_indices)),
            "modality_points": int(args.num_points),
            "modality_pairs": int(modality_pair_count),
            "semantic_top2_labels": semantic_labels,
            "semantic_per_class": int(args.semantic_per_class),
            "semantic_selected_pairs": int(len(semantic_image_indices)),
        },
    )

    summary: dict[str, Any] = {"pair_cosine": pair_summary, "models": {}}
    for model, artifacts in artifacts_by_model.items():
        model_dir = args.output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        modality_pairs = make_pair_batch(artifacts, modality_image_indices, modality_caption_indices)
        semantic_pairs = make_pair_batch(artifacts, semantic_image_indices, semantic_caption_indices)

        modality_result = run_model_umap(artifacts, modality_pairs, labels_by_image_id, model_dir, args)
        semantic_result = run_semantic_umap(
            artifacts,
            semantic_pairs,
            semantic_point_labels,
            labels_by_image_id,
            model_dir,
            args,
        )

        model_summary = {
            "pair_cosine_all_first_caption": pair_summary[model],
            "modality_umap": modality_result,
            "semantic_top2_umap": semantic_result,
        }
        summary["models"][model] = model_summary
        write_json(model_dir / "metrics.json", model_summary)

    write_json(args.output_dir / "summary.json", summary)
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
