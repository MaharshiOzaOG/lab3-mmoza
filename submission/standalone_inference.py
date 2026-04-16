#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from src.model.config import ModelConfig
from src.model.language_model import TransformerLanguageModel
from src.tokenizer.loading import load_tokenizer


def maybe_set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: Any) -> int | None:
    if model is None or not hasattr(model, "parameters"):
        return None
    return int(sum(p.numel() for p in model.parameters()))


def total_artifact_size_bytes(paths: list[str | Path]) -> int:
    total = 0
    for path in paths:
        resolved = (PACKAGE_DIR / path).resolve() if not Path(path).is_absolute() else Path(path)
        if resolved.exists() and resolved.is_file():
            total += resolved.stat().st_size
    return total


def maybe_strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_runtime(device: str) -> dict[str, Any]:
    tokenizer_path = PACKAGE_DIR / "tokenizer.json"
    checkpoint_path = PACKAGE_DIR / "best_model.pt"
    config_path = PACKAGE_DIR / "model_config.json"

    tokenizer = load_tokenizer(str(tokenizer_path))
    model_config = ModelConfig.load(str(config_path))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = maybe_strip_module_prefix(state_dict)

    model = TransformerLanguageModel(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "submission_name": "lab3_small_3ep_baseline",
        "dtype": "float32",
        "artifact_paths": [
            "standalone_inference.py",
            "best_model.pt",
            "tokenizer.json",
            "model_config.json",
        ],
        "device": device,
    }


def generate_with_runtime(
    runtime: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    maybe_set_seed(seed)

    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = runtime["device"]

    input_ids_list = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    top_k_arg = top_k if top_k > 0 else None
    top_p_arg = top_p if top_p < 1.0 else None
    do_sample = (temperature != 1.0) or (top_k_arg is not None) or (top_p_arg is not None)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_arg,
            top_p=top_p_arg,
            do_sample=do_sample,
        )

    full_ids = generated[0].tolist()
    response_ids = full_ids[len(input_ids_list):]

    generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "response_text": response_text,
        "num_generated_tokens": len(response_ids),
        "parameter_count": count_parameters(model),
        "artifact_paths": runtime["artifact_paths"],
        "dtype": runtime.get("dtype", "float32"),
        "extra": {
            "checkpoint": "best_model.pt",
            "tokenizer": "tokenizer.json",
            "config": "model_config.json",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone inference entry point for Moodle submission")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0 disables)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to also save the JSON result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maybe_set_seed(args.seed)
    device = resolve_device(args.device)

    runtime = load_runtime(device=device)

    start_time = time.perf_counter()
    generation = generate_with_runtime(
        runtime=runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    wall_time_sec = time.perf_counter() - start_time

    num_generated_tokens = int(generation["num_generated_tokens"])
    seconds_per_generated_token = (
        wall_time_sec / num_generated_tokens if num_generated_tokens > 0 else None
    )
    tokens_per_second = (
        num_generated_tokens / wall_time_sec if wall_time_sec > 0 and num_generated_tokens > 0 else None
    )

    artifact_paths = generation.get("artifact_paths") or runtime.get("artifact_paths") or []
    parameter_count = generation.get("parameter_count")
    if parameter_count is None:
        parameter_count = count_parameters(runtime.get("model"))

    result = {
        "submission_name": runtime.get("submission_name", PACKAGE_DIR.name),
        "prompt": args.prompt,
        "generated_text": generation["generated_text"],
        "response_text": generation["response_text"],
        "num_generated_tokens": num_generated_tokens,
        "wall_time_sec": wall_time_sec,
        "seconds_per_generated_token": seconds_per_generated_token,
        "tokens_per_second": tokens_per_second,
        "parameter_count": parameter_count,
        "artifact_size_bytes": total_artifact_size_bytes(artifact_paths),
        "device": device,
        "dtype": generation.get("dtype", runtime.get("dtype")),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "extra": generation.get("extra", {}),
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")

    sys.stdout.write(json_text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())