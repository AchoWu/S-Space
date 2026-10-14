"""Dump COCO-1800 query-minus-reference states for offline axis comparison.

The formal training run stores validation margins only for its own axes. This
script stores the raw post-block differences ``h_query - h_reference`` once,
so every candidate axis set can be scored offline without another forward.

Run from the project root with the model environment, e.g.::

    PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 .envs/qwen36/bin/python \
        spectral_subspace/dump_validation_states.py \
        --config configs/sspace/train/pilot_qwen35_4b_spectral.json \
        --output outputs/spectral_pilot/qwen35_4b_validation_states

It lives outside ``sspace/`` on purpose: files under ``sspace/`` enter the
launch fingerprint, and adding one would invalidate completed formal outputs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image

from sspace.core.config import load_distributed_config
from sspace.core.data import load_construction_split
from sspace.core.extraction.execution import (
    LayerwisePostBlockStates,
    move_model_inputs,
)
from sspace.core.models.runtime import load_model_runtime
from sspace.core.prompts.rendering import render_prompt_pair
from sspace.core.prompts.templates import PROMPT_STYLE_ORDER
from sspace.determinism import configure_deterministic_cuda
from sspace.run_records import atomic_json, canonical_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configure_torch(seed: int) -> None:
    """Mirror the numerical settings of ``sspace.core.cli``."""
    configure_deterministic_cuda()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated post-block IDs, or 'all' (default)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    config = load_distributed_config(args.config, PROJECT_ROOT)
    _configure_torch(config.seed)
    _, samples = load_construction_split(config.validation_dataset, "validation")
    selected = config.validation_selection.resolve(len(samples))
    jobs = [(index, style) for index in selected for style in PROMPT_STYLE_ORDER]

    torch.cuda.set_device(0)
    runtime = load_model_runtime(config.model_adapter, config.model_path, "cuda:0")
    if args.layers == "all":
        layers = list(range(runtime.spec.layer_count))
    else:
        layers = sorted({int(value) for value in args.layers.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= runtime.spec.layer_count:
        raise ValueError(f"Unsupported layers {layers}")

    shape = (len(jobs), 2, len(layers), runtime.spec.hidden_size)
    identity = {
        "config": config.to_dict(),
        "layers": layers,
        "shape": list(shape),
    }
    fingerprint = canonical_fingerprint(identity)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "differences.npy"
    jobs_path = output / "jobs.jsonl"
    checkpoint_path = output / "checkpoint.json"

    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Completed dump was produced by a different config")
        print(f"already complete: {output}")
        return
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint was produced by a different config")
        next_job = int(checkpoint["next_job"])
        states = np.load(states_path, mmap_mode="r+")
        lines = jobs_path.read_text(encoding="utf-8").splitlines()[:next_job]
        jobs_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    else:
        if states_path.exists() or jobs_path.exists():
            raise FileExistsError(f"Dump files exist without a checkpoint: {output}")
        states = open_memmap(states_path, mode="w+", dtype=np.float32, shape=shape)
        states[:] = np.nan
        next_job = 0
        jobs_path.touch()

    def checkpoint() -> None:
        states.flush()
        atomic_json(checkpoint_path, {"fingerprint": fingerprint, "next_job": next_job})

    if next_job == 0:
        checkpoint()
    batch_size = runtime.spec.orientation_batch_size
    with (
        jobs_path.open("a", encoding="utf-8") as records,
        LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor,
    ):
        while next_job < len(jobs):
            sample_index, style = jobs[next_job]
            sample = samples[sample_index]
            prompts = render_prompt_pair(
                style, sample.group, sample.query, sample.reference
            )
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            differences = []
            for start in range(0, len(prompts), batch_size):
                prepared = runtime.prepare(
                    runtime.processor, image, prompts[start : start + batch_size]
                )
                differences.append(
                    extractor.extract(
                        move_model_inputs(prepared.inputs, runtime.model),
                        prepared.positions,
                    )
                )
            value = torch.cat(differences, dim=0)[:, layers, :]
            states[next_job] = value.numpy()
            records.write(
                json.dumps(
                    {
                        "job_index": next_job,
                        "global_sample_index": sample_index,
                        "sample_id": sample.sample_id,
                        "style": style,
                        "group": sample.group,
                        "label": sample.label,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            next_job += 1
            if next_job % args.checkpoint_every == 0:
                records.flush()
                checkpoint()
                print(f"validation states {next_job}/{len(jobs)}", flush=True)
        records.flush()
        checkpoint()
    if not np.isfinite(states).all():
        raise FloatingPointError("Dumped validation states are non-finite")
    atomic_json(
        output / "manifest.json",
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "model_adapter": config.model_adapter,
            "layers": layers,
            "shape": list(shape),
            "orientations": ["original", "swapped"],
            "value": "h_query - h_reference (post-block, float32)",
        },
    )
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
