#!/usr/bin/env python3
"""COCO image-text retrieval for CLIP, BLIP, and BLIP-2.

The script extracts image/text retrieval embeddings, computes the image-text
similarity matrix, and reports Text-to-Image and Image-to-Text Recall@K.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class CocoRetrievalData:
    image_ids: list[int]
    image_paths: list[Path]
    captions: list[str]
    caption_image_indices: np.ndarray
    image_to_caption_indices: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-text retrieval on COCO val2017.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["clip", "blip", "blip2"],
        help="Run exactly one retrieval model.",
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, default=None)
    parser.add_argument("--blip-dir", type=Path, default=None)
    parser.add_argument("--blip2-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use the first N COCO val images. 0 means all 5000 images.",
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future dataloader use.")
    parser.add_argument(
        "--no-save-similarity",
        dest="save_similarity",
        action="store_false",
        default=True,
        help="Do not save the full similarity matrix.",
    )
    parser.add_argument("--reuse-embeddings", action="store_true", help="Load existing embeddings if present.")
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=True,
        help="Only load model files from local directories.",
    )
    parser.add_argument(
        "--allow-remote-files",
        dest="local_files_only",
        action="store_false",
        help="Allow transformers to download missing model files.",
    )
    return parser.parse_args()


def load_coco_retrieval_data(
    annotation_path: Path,
    image_root: Path,
    max_images: int = 0,
) -> CocoRetrievalData:
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    images = sorted(raw["images"], key=lambda x: x["id"])
    if max_images and max_images > 0:
        images = images[:max_images]

    image_ids = [int(x["id"]) for x in images]
    image_id_to_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
    image_paths = [image_root / x["file_name"] for x in images]

    captions: list[str] = []
    caption_image_indices: list[int] = []
    image_to_caption_indices: list[list[int]] = [[] for _ in image_ids]

    annotations = sorted(raw["annotations"], key=lambda x: (x["image_id"], x["id"]))
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_id_to_index:
            continue
        caption_index = len(captions)
        image_index = image_id_to_index[image_id]
        captions.append(str(ann["caption"]).strip())
        caption_image_indices.append(image_index)
        image_to_caption_indices[image_index].append(caption_index)

    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        shown = "\n".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} image files are missing. First missing files:\n{shown}")

    return CocoRetrievalData(
        image_ids=image_ids,
        image_paths=image_paths,
        captions=captions,
        caption_image_indices=np.asarray(caption_image_indices, dtype=np.int64),
        image_to_caption_indices=image_to_caption_indices,
    )


def read_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def batches(items: list[Any], batch_size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def resolve_model_path(args: argparse.Namespace, model_name: str) -> Path:
    explicit = {
        "clip": args.clip_dir,
        "blip": args.blip_dir,
        "blip2": args.blip2_dir,
    }[model_name]
    if explicit is not None:
        return explicit
    defaults = {
        "clip": "clip-vit-base-patch32",
        "blip": "blip-itm-base-coco",
        "blip2": "blip2-itm-vit-g-coco",
    }
    return args.model_root / defaults[model_name]


def choose_dtype(model_name: str, requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.float32


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


class RetrievalModel:
    name: str

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype, local_files_only: bool):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only

    def load(self) -> None:
        raise NotImplementedError

    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, captions: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, image_embeds: np.ndarray, text_embeds: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(image_embeds).float()
        text = torch.from_numpy(text_embeds).float()
        if image.ndim == 3:
            sims = torch.einsum("iqd,td->iqt", image, text).amax(dim=1)
        else:
            sims = image @ text.t()
        return sims.cpu().numpy().astype(np.float32)


class ClipRetrievalModel(RetrievalModel):
    name = "clip"

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            self.model_path, local_files_only=self.local_files_only, torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        feats = []
        for _, path_batch in tqdm(list(batches(image_paths, batch_size)), desc=f"{self.name} image"):
            images = read_images(path_batch)
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            image_feat = self.model.get_image_features(**inputs)
            image_feat = F.normalize(image_feat, dim=-1)
            feats.append(image_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.inference_mode()
    def encode_texts(self, captions: list[str], batch_size: int) -> np.ndarray:
        feats = []
        for _, text_batch in tqdm(list(batches(captions, batch_size)), desc=f"{self.name} text"):
            inputs = self.processor(text=text_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            text_feat = self.model.get_text_features(**inputs)
            text_feat = F.normalize(text_feat, dim=-1)
            feats.append(text_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


class BlipRetrievalModel(RetrievalModel):
    name = "blip"

    def load(self) -> None:
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        self.model = BlipForImageTextRetrieval.from_pretrained(
            self.model_path, local_files_only=self.local_files_only, torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        feats = []
        for _, path_batch in tqdm(list(batches(image_paths, batch_size)), desc=f"{self.name} image"):
            images = read_images(path_batch)
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            image_feat = self.model.vision_proj(vision_outputs.last_hidden_state[:, 0, :])
            image_feat = F.normalize(image_feat, dim=-1)
            feats.append(image_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.inference_mode()
    def encode_texts(self, captions: list[str], batch_size: int) -> np.ndarray:
        feats = []
        for _, text_batch in tqdm(list(batches(captions, batch_size)), desc=f"{self.name} text"):
            inputs = self.processor(text=text_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            text_outputs = self.model.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            text_feat = self.model.text_proj(text_outputs.last_hidden_state[:, 0, :])
            text_feat = F.normalize(text_feat, dim=-1)
            feats.append(text_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


class Blip2RetrievalModel(RetrievalModel):
    name = "blip2"

    def load(self) -> None:
        import transformers
        from transformers import AutoProcessor

        if not hasattr(transformers, "Blip2ForImageTextRetrieval"):
            raise RuntimeError(
                "Current transformers does not provide Blip2ForImageTextRetrieval. "
                "For BLIP-2 retrieval, switch alignment to a newer transformers version "
                "(transformers==4.45.2 was verified in the blip2 environment). "
                "Note: old BLIP caption models need transformers<4.27, so these tasks "
                "cannot share one perfect transformers version."
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        self.model = transformers.Blip2ForImageTextRetrieval.from_pretrained(
            self.model_path, local_files_only=self.local_files_only, torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path], batch_size: int) -> np.ndarray:
        feats = []
        for _, path_batch in tqdm(list(batches(image_paths, batch_size)), desc=f"{self.name} image"):
            images = read_images(path_batch)
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            image_feat = self.model.vision_projection(query_outputs.last_hidden_state)
            image_feat = F.normalize(image_feat, dim=-1)
            feats.append(image_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.inference_mode()
    def encode_texts(self, captions: list[str], batch_size: int) -> np.ndarray:
        feats = []
        for _, text_batch in tqdm(list(batches(captions, batch_size)), desc=f"{self.name} text"):
            inputs = self.processor(text=text_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = move_batch(inputs, self.device, self.dtype)
            query_embeds = self.model.embeddings(input_ids=inputs["input_ids"])
            text_outputs = self.model.qformer(
                query_embeds=query_embeds,
                query_length=0,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            text_feat = self.model.text_projection(text_outputs.last_hidden_state[:, 0, :])
            text_feat = F.normalize(text_feat, dim=-1)
            feats.append(text_feat.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


MODEL_CLASSES = {
    "clip": ClipRetrievalModel,
    "blip": BlipRetrievalModel,
    "blip2": Blip2RetrievalModel,
}


def recall_metrics(
    similarity: np.ndarray,
    caption_image_indices: np.ndarray,
    image_to_caption_indices: list[list[int]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    sim = torch.from_numpy(similarity)
    caption_targets = torch.from_numpy(caption_image_indices)
    max_k = max(ks)

    text_to_image_scores = sim.t()
    t2i_top_k = min(max_k, text_to_image_scores.shape[1])
    t2i_top = torch.topk(text_to_image_scores, k=t2i_top_k, dim=1).indices
    t2i_correct = t2i_top.eq(caption_targets[:, None])

    image_to_text_scores = sim
    i2t_top_k = min(max_k, image_to_text_scores.shape[1])
    i2t_top = torch.topk(image_to_text_scores, k=i2t_top_k, dim=1).indices

    text_to_image_results: dict[int, float] = {}
    image_to_text_results: dict[int, float] = {}
    for k in ks:
        t2i_k = min(k, t2i_top_k)
        i2t_k = min(k, i2t_top_k)
        text_to_image_results[k] = float(t2i_correct[:, :t2i_k].any(dim=1).float().mean().item() * 100.0)

        hits = []
        for image_index, positive_caption_indices in enumerate(image_to_caption_indices):
            positives = set(positive_caption_indices)
            hit = any(int(idx) in positives for idx in i2t_top[image_index, :i2t_k])
            hits.append(hit)
        image_to_text_results[k] = float(np.mean(hits) * 100.0)

    results: dict[str, float] = {}
    for k in ks:
        results[f"text_to_image_R@{k}"] = text_to_image_results[k]
    for k in ks:
        results[f"image_to_text_R@{k}"] = image_to_text_results[k]

    return results


def save_metadata(data: CocoRetrievalData, path: Path) -> None:
    metadata = {
        "image_ids": data.image_ids,
        "image_paths": [str(x) for x in data.image_paths],
        "captions": data.captions,
        "caption_image_indices": data.caption_image_indices.tolist(),
        "image_to_caption_indices": data.image_to_caption_indices,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run_model(args: argparse.Namespace, data: CocoRetrievalData) -> dict[str, Any]:
    model_name = args.model
    device = torch.device(args.device)
    dtype = choose_dtype(model_name, args.dtype, device)
    model_path = resolve_model_path(args, model_name)
    model_output_dir = args.output_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = model_output_dir / "embeddings.npz"
    metadata_path = model_output_dir / "metadata.json"
    metrics_path = model_output_dir / "metrics.json"
    similarity_path = model_output_dir / "similarity.npy"

    started = time.time()
    if args.reuse_embeddings and embedding_path.exists():
        arrays = np.load(embedding_path)
        image_embeds = arrays["image_embeds"]
        text_embeds = arrays["text_embeds"]
        print(f"[{model_name}] reused embeddings from {embedding_path}")
        model = None
    else:
        model_cls = MODEL_CLASSES[model_name]
        model = model_cls(model_path, device=device, dtype=dtype, local_files_only=args.local_files_only)
        print(f"[{model_name}] loading {model_path} on {device} ({dtype})")
        model.load()
        image_embeds = model.encode_images(data.image_paths, args.image_batch_size)
        text_embeds = model.encode_texts(data.captions, args.text_batch_size)
        np.savez_compressed(
            embedding_path,
            image_embeds=image_embeds.astype(np.float32),
            text_embeds=text_embeds.astype(np.float32),
        )

    if model is None:
        similarity = RetrievalModel(Path("."), device, dtype, True).similarity(image_embeds, text_embeds)
    else:
        similarity = model.similarity(image_embeds, text_embeds)

    metrics = recall_metrics(similarity, data.caption_image_indices, data.image_to_caption_indices)
    elapsed_sec = time.time() - started
    payload = {
        "model": model_name,
        "model_path": str(model_path),
        "num_images": len(data.image_paths),
        "num_captions": len(data.captions),
        "image_embedding_shape": list(image_embeds.shape),
        "text_embedding_shape": list(text_embeds.shape),
        "dtype": str(dtype),
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "metrics": metrics,
    }
    save_metadata(data, metadata_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.save_similarity:
        np.save(similarity_path, similarity)
    print(f"[{model_name}] metrics: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"[{model_name}] wrote {metrics_path}")
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco_retrieval_data(args.annotations, args.image_root, args.max_images)
    print(f"Loaded {len(data.image_paths)} images and {len(data.captions)} captions")

    run_model(args, data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
