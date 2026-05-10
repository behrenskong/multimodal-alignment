#!/usr/bin/env python3
"""Generate and evaluate COCO captions for BLIP and BLIP-2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_MODEL_DIRS = {
    "blip": "blip-image-captioning-base",
    "blip2": "blip2-opt-2.7b-coco",
}

DEFAULT_PROMPTS = {
    # This script uses Hugging Face BlipForConditionalGeneration. The HF
    # captioning model card recommends unconditional captioning for the normal
    # image-captioning path; the LAVIS "a picture of " prompt makes HF BLIP
    # behave like short phrase completion.
    "blip": "",
    "blip2": "a photo of",
}


@dataclass
class CocoCaptionData:
    image_ids: list[int]
    image_paths: list[Path]
    gt_captions: dict[int, list[str]]


@dataclass
class GenerationConfig:
    num_beams: int
    max_length: int
    min_length: int
    length_penalty: float
    repetition_penalty: float
    do_sample: bool
    top_p: float
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate BLIP/BLIP-2 captions on COCO val2017.")
    parser.add_argument("--model", required=True, choices=["blip", "blip2"], help="Run exactly one captioning model.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--blip-dir", type=Path, default=None)
    parser.add_argument("--blip2-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="Use first N COCO val images. 0 means all images.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. auto uses float32.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt prefix. Defaults: BLIP uses HF unconditional captioning, BLIP2 uses LAVIS COCO prompt 'a photo of'.",
    )
    parser.add_argument("--no-prompt", action="store_true", help="Generate without a text prompt prefix.")
    parser.add_argument("--keep-prompt", action="store_true", help="Do not strip prompt text from decoded captions.")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true", help="Use nucleus sampling instead of deterministic beams.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip-eval", action="store_true", help="Only generate captions; do not run pycocoevalcap.")
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse output-dir/predictions.json if it exists.")
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--pycocoevalcap-root", type=Path, default=Path("../pycocoevalcap"))
    parser.add_argument(
        "--java-home",
        type=Path,
        default=None,
        help="Java home for SPICE. Defaults to local Java 8 if present.",
    )
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


def load_coco_caption_data(annotation_path: Path, image_root: Path, max_images: int) -> CocoCaptionData:
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    images = sorted(raw["images"], key=lambda x: int(x["id"]))
    if max_images and max_images > 0:
        images = images[:max_images]

    image_ids = [int(item["id"]) for item in images]
    selected_ids = set(image_ids)
    image_paths = [image_root / item["file_name"] for item in images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        shown = "\n".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} image files are missing. First missing files:\n{shown}")

    gt_captions: dict[int, list[str]] = {image_id: [] for image_id in image_ids}
    for ann in sorted(raw["annotations"], key=lambda x: (int(x["image_id"]), int(x["id"]))):
        image_id = int(ann["image_id"])
        if image_id in selected_ids:
            gt_captions[image_id].append(str(ann["caption"]).strip())

    return CocoCaptionData(image_ids=image_ids, image_paths=image_paths, gt_captions=gt_captions)


def batches(indices: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def read_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def resolve_model_path(args: argparse.Namespace) -> Path:
    explicit = args.blip_dir if args.model == "blip" else args.blip2_dir
    if explicit is not None:
        return explicit
    return args.model_root / DEFAULT_MODEL_DIRS[args.model]


def choose_dtype(requested: str) -> torch.dtype:
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


def clean_caption(caption: str, prompt: str, keep_prompt: bool) -> str:
    text = " ".join(caption.strip().split())
    prompt_text = " ".join(prompt.strip().split())
    if prompt_text and not keep_prompt and text.lower().startswith(prompt_text.lower()):
        text = text[len(prompt_text) :].lstrip(" ,.:;")
    return text


class CaptionModel:
    def __init__(
        self,
        model_name: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        prompt: str,
        keep_prompt: bool,
        local_files_only: bool,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.prompt = prompt
        self.keep_prompt = keep_prompt
        self.local_files_only = local_files_only

    def load(self) -> None:
        if self.model_name == "blip":
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_path, local_files_only=self.local_files_only, torch_dtype=self.dtype
            )
            self._patch_blip_batched_beam_search()
        elif self.model_name == "blip2":
            from transformers import AutoProcessor, Blip2ForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path, local_files_only=self.local_files_only, torch_dtype=self.dtype
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model.to(self.device)
        self.model.eval()

    def _patch_blip_batched_beam_search(self) -> None:
        """Expand BLIP image states together with beam-search text states."""
        original_expand = self.model.text_decoder._expand_inputs_for_generation

        def expand_inputs_for_generation(
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: torch.LongTensor | None = None,
            **model_kwargs: Any,
        ):
            input_ids, model_kwargs = original_expand(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs,
            )
            if input_ids is None:
                return input_ids, model_kwargs
            for key in ("encoder_hidden_states", "encoder_attention_mask"):
                value = model_kwargs.get(key)
                if isinstance(value, torch.Tensor) and value.shape[0] * expand_size == input_ids.shape[0]:
                    model_kwargs[key] = value.repeat_interleave(expand_size, dim=0)
            return input_ids, model_kwargs

        self.model.text_decoder._expand_inputs_for_generation = expand_inputs_for_generation

    @torch.inference_mode()
    def generate(self, image_paths: list[Path], batch_size: int, gen_cfg: GenerationConfig) -> list[str]:
        captions: list[str] = []
        index_list = list(range(len(image_paths)))
        for batch_indices in tqdm(list(batches(index_list, batch_size)), desc=f"{self.model_name} caption"):
            path_batch = [image_paths[i] for i in batch_indices]
            images = read_images(path_batch)
            processor_kwargs: dict[str, Any] = {
                "images": images,
                "return_tensors": "pt",
            }
            if self.prompt:
                processor_kwargs["text"] = [self.prompt] * len(images)
                processor_kwargs["padding"] = True

            inputs = self.processor(**processor_kwargs)
            inputs = move_batch(inputs, self.device, self.dtype)

            generate_kwargs: dict[str, Any] = {
                "num_beams": gen_cfg.num_beams,
                "max_length": gen_cfg.max_length,
                "min_length": gen_cfg.min_length,
                "length_penalty": gen_cfg.length_penalty,
                "repetition_penalty": gen_cfg.repetition_penalty,
                "do_sample": gen_cfg.do_sample,
            }
            if gen_cfg.do_sample:
                generate_kwargs["top_p"] = gen_cfg.top_p
                generate_kwargs["temperature"] = gen_cfg.temperature
            else:
                generate_kwargs["early_stopping"] = True

            outputs = self.model.generate(**inputs, **generate_kwargs)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            captions.extend(clean_caption(text, self.prompt, self.keep_prompt) for text in decoded)
        return captions


def install_local_pycocoevalcap(root: Path) -> None:
    if root.exists():
        package_parent = root.parent if root.name == "pycocoevalcap" else root
        sys.path.insert(0, str(package_parent))


def configure_java_for_spice(java_home: Path | None) -> None:
    # SPICE is old: Java 8 is the most reliable path because Java 15+ removed
    # Nashorn, which SPICE still expects. /tmp is also noexec-like here, so
    # send Java native extraction into this project directory.
    java_tmp = Path(__file__).resolve().parent / ".java_tmp"
    java_tmp.mkdir(parents=True, exist_ok=True)

    default_java8 = Path("/usr/lib/jvm/java-8-openjdk-amd64")
    if java_home is None and (default_java8 / "bin" / "java").exists():
        java_home = default_java8

    opts = [f"-Djava.io.tmpdir={java_tmp}"]
    if java_home is not None and (java_home / "bin" / "java").exists():
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    else:
        # Best-effort fallback for Java 9+. This still cannot restore Nashorn
        # on Java 15+, but helps on Java 9-14.
        opts.extend(
            [
                "--add-opens=java.base/java.lang=ALL-UNNAMED",
                "--add-opens=java.base/java.math=ALL-UNNAMED",
                "--add-opens=java.base/java.util=ALL-UNNAMED",
                "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
                "--add-opens=java.base/java.io=ALL-UNNAMED",
                "--add-opens=java.base/java.net=ALL-UNNAMED",
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "--add-opens=java.base/java.text=ALL-UNNAMED",
            ]
        )

    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    merged = existing.split()
    for opt in opts:
        if opt not in merged:
            merged.append(opt)
    os.environ["JAVA_TOOL_OPTIONS"] = " ".join(merged).strip()


def evaluate_predictions(
    annotation_path: Path,
    predictions_path: Path,
    pycocoevalcap_root: Path,
    java_home: Path | None,
) -> dict[str, Any]:
    install_local_pycocoevalcap(pycocoevalcap_root)
    configure_java_for_spice(java_home)
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO

    coco = COCO(str(annotation_path))
    coco_result = coco.loadRes(str(predictions_path))
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    scores = {metric: float(score) for metric, score in coco_eval.eval.items()}
    required = {
        "BLEU-4": scores.get("Bleu_4"),
        "CIDEr": scores.get("CIDEr"),
        "METEOR": scores.get("METEOR"),
        "ROUGE-L": scores.get("ROUGE_L"),
        "SPICE": scores.get("SPICE"),
    }
    return {"all": scores, "required": required}


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_examples(data: CocoCaptionData, predictions: list[dict[str, Any]], num_examples: int) -> list[dict[str, Any]]:
    image_id_to_path = {image_id: str(path) for image_id, path in zip(data.image_ids, data.image_paths)}
    examples = []
    for item in predictions[:num_examples]:
        image_id = int(item["image_id"])
        examples.append(
            {
                "image_id": image_id,
                "image_path": image_id_to_path[image_id],
                "prediction": item["caption"],
                "ground_truth": data.gt_captions.get(image_id, []),
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt = "" if args.no_prompt else (args.prompt if args.prompt is not None else DEFAULT_PROMPTS[args.model])
    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype)
    model_path = resolve_model_path(args)
    gen_cfg = GenerationConfig(
        num_beams=args.num_beams,
        max_length=args.max_length,
        min_length=args.min_length,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    data = load_coco_caption_data(args.annotations, args.image_root, args.max_images)
    print(f"Loaded {len(data.image_paths)} images from {args.annotations}")

    predictions_path = args.output_dir / "predictions.json"
    examples_path = args.output_dir / "examples.json"
    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "metrics.json"

    started = time.time()
    if args.reuse_predictions and predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"[{args.model}] reused predictions from {predictions_path}")
    else:
        caption_model = CaptionModel(
            args.model,
            model_path=model_path,
            device=device,
            dtype=dtype,
            prompt=prompt,
            keep_prompt=args.keep_prompt,
            local_files_only=args.local_files_only,
        )
        print(f"[{args.model}] loading {model_path} on {device} ({dtype})")
        print(f"[{args.model}] prompt={prompt!r}")
        caption_model.load()
        captions = caption_model.generate(data.image_paths, args.batch_size, gen_cfg)
        predictions = [
            {"image_id": image_id, "caption": caption}
            for image_id, caption in zip(data.image_ids, captions)
        ]
        write_json(predictions_path, predictions)

    examples = build_examples(data, predictions, args.num_examples)
    write_json(examples_path, examples)
    write_json(
        metadata_path,
        {
            "model": args.model,
            "model_path": str(model_path),
            "image_root": str(args.image_root),
            "annotations": str(args.annotations),
            "num_images": len(data.image_paths),
            "image_ids": data.image_ids,
            "prompt": prompt,
            "keep_prompt": args.keep_prompt,
            "dtype": str(dtype),
            "device": str(device),
            "generation": gen_cfg.__dict__,
        },
    )

    metrics: dict[str, Any] = {}
    if not args.skip_eval:
        metrics = evaluate_predictions(args.annotations, predictions_path, args.pycocoevalcap_root, args.java_home)

    payload = {
        "model": args.model,
        "model_path": str(model_path),
        "num_images": len(data.image_paths),
        "num_predictions": len(predictions),
        "prompt": prompt,
        "dtype": str(dtype),
        "device": str(device),
        "generation": gen_cfg.__dict__,
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
    }
    write_json(metrics_path, payload)
    print(f"[{args.model}] wrote {predictions_path}")
    print(f"[{args.model}] wrote {metrics_path}")
    if metrics:
        print(f"[{args.model}] required metrics: {json.dumps(metrics['required'], ensure_ascii=False)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
