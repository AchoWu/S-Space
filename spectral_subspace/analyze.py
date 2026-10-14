"""Offline Phase-0 comparison of mean-gradient axes and spectral variants.

Inputs are a finished formal training run (its ``shards/`` hold every
per-sample role gradient) and a COCO-1800 state dump written by
``dump_validation_states.py``. Nothing here loads a model, so it runs on CPU
with only NumPy::

    PYTHONPATH=. .envs/spatialmqa/bin/python spectral_subspace/analyze.py \
        --run-dir outputs/spectral_pilot/qwen35_4b_coco600_final_logit_axes \
        --validation-states outputs/spectral_pilot/qwen35_4b_validation_states \
        --released-artifact \
            sspace/core/artifacts/pretrained/qwen35_4b_coco6000_final_logit_axes_v2 \
        --out outputs/spectral_pilot/qwen35_4b_analysis

Axis sets (each gives one unit vector per H/V/D per layer):

* ``mean``: the released method, ``unit(mean_g a)``.
* ``raw_svd``: project the means onto the top-3 uncentered principal
  subspace of all per-prompt gradients.
* ``sample_svd``: the same, but the subspace comes from per-sample means
  (five prompt styles averaged), which suppresses wording noise.
* ``cross_svd``: the top-3 eigenvectors of the original x swapped
  cross-covariance of per-sample means, which keeps only role-consistent
  structure (see ``cross_validated_spectrum``).
* ``whiten_a{alpha}``: ``unit((C + alpha * tr(C)/D * I)^-1 mean_g a)`` with
  ``C`` the pooled within-axis gradient covariance (label-free LDA).

Every set is read out either by projection ``v_g^T dh`` or by the dual basis
``[(V V^T)^-1 V dh]_g``. Endpoint labels are used only for scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sspace.core.prompts.templates import AXIS_ORDER, PROMPT_STYLE_ORDER
from sspace.experiments.validation.coco1800.selection.layers import (
    POSITIVE_ENDPOINT,
)
from sspace.run_records import atomic_json

STYLE_COUNT = len(PROMPT_STYLE_ORDER)
READOUTS = ("project", "dual")


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def _top_right_vectors(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-``k`` uncentered right singular vectors ``[D,k]``.

    Also returns the leading eigenvalues of ``M^T M / n`` divided by its trace.
    """
    rows, dim = matrix.shape
    if rows < dim:
        _, values, vt = np.linalg.svd(matrix, full_matrices=False)
        energy = values**2
        vectors = vt[:k].T
    else:
        energy, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
        energy = energy[::-1]
        vectors = eigenvectors[:, ::-1][:, :k]
    return vectors, energy[:k] / energy.sum()


def _project_means(basis: np.ndarray, means: np.ndarray) -> np.ndarray:
    return _unit(means @ basis @ basis.T)


def _principal_cosines(first: np.ndarray, second: np.ndarray) -> list[float]:
    """Cosines of principal angles between the row spans of two ``[k,D]``."""
    q1, _ = np.linalg.qr(first.T)
    q2, _ = np.linalg.qr(second.T)
    return np.linalg.svd(q1.T @ q2, compute_uv=False).tolist()


def build_axis_sets(
    states: np.ndarray, groups: np.ndarray, alphas: list[float]
) -> tuple[dict[str, np.ndarray], dict]:
    """Build every candidate ``[3,D]`` axis set from structured gradients.

    Args:
        states: Role gradients ``[S, 2 orientations, 5 styles, D]``.
        groups: Axis index per sample ``[S]``.
        alphas: Shrinkage multipliers for the whitened variant.

    Returns:
        Axis sets by name, and spectra reused by the diagnostics.
    """
    dim = states.shape[-1]
    rows = states.reshape(-1, dim)
    row_groups = np.repeat(groups, 2 * STYLE_COUNT)
    means = np.stack([rows[row_groups == g].mean(axis=0) for g in range(3)])
    sets = {"mean": _unit(means)}

    raw_basis, raw_energy = _top_right_vectors(rows, 20)
    sets["raw_svd"] = _project_means(raw_basis[:, :3], means)
    sample_means = states.mean(axis=2).reshape(-1, dim)
    sample_basis, sample_energy = _top_right_vectors(sample_means, 20)
    sets["sample_svd"] = _project_means(sample_basis[:, :3], means)
    per_sample = states.mean(axis=2)
    cross = per_sample[:, 0].T @ per_sample[:, 1]
    _, eigenvectors = np.linalg.eigh((cross + cross.T) / 2)
    sets["cross_svd"] = _project_means(eigenvectors[:, ::-1][:, :3], means)

    within = rows - means[row_groups]
    covariance = within.T @ within / (len(rows) - 3)
    scale = np.trace(covariance) / dim
    identity = np.eye(dim)
    for alpha in alphas:
        solved = np.linalg.solve(covariance + alpha * scale * identity, means.T)
        sets[f"whiten_a{alpha:g}"] = _unit(solved.T)
    spectra = {
        "means": means,
        "raw_basis": raw_basis,
        "raw_energy_fraction": raw_energy.tolist(),
        "sample_energy_fraction": sample_energy.tolist(),
    }
    return sets, spectra


def cross_validated_spectrum(
    states: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    splits: int,
    k: int,
) -> dict:
    """Role-consistent gradient energy per dimension (cross-validated PCA).

    Per-sample means are split by orientation. The spatial part of the role
    gradient keeps its sign when query and reference swap, while role-bound
    parts (e.g. object identity) flip, and prompt noise is independent. So the
    symmetrized cross-covariance ``(P_orig^T P_swap + P_swap^T P_orig) / 2``
    is positive only along role-consistent directions. Its eigenvectors are
    fitted on half of the samples and scored on the other half; scores are
    fractions of the held-out uncentered energy, so noise dimensions sit near
    zero and role-bound ones go negative. ``mean_span_captured`` scores the
    span of the three fitted axis means the same way.
    """
    per_sample = states.mean(axis=2)
    count = per_sample.shape[0]
    fractions, captured, totals = [], [], []
    for _ in range(splits):
        order = rng.permutation(count)
        halves = (order[: count // 2], order[count // 2 :])
        for fit, test in (halves, halves[::-1]):
            first, second = per_sample[fit, 0], per_sample[fit, 1]
            cross = first.T @ second
            _, eigenvectors = np.linalg.eigh((cross + cross.T) / 2)
            basis = eigenvectors[:, ::-1][:, :k]
            original, swapped = per_sample[test, 0], per_sample[test, 1]
            energy = (
                np.einsum("nd,nd->n", original, original)
                + np.einsum("nd,nd->n", swapped, swapped)
            ).mean() / 2
            score = ((original @ basis) * (swapped @ basis)).mean(axis=0)
            fractions.append(score / energy)
            totals.append(np.einsum("nd,nd->n", original, swapped).mean() / energy)
            fit_means = per_sample[fit].mean(axis=1)
            means = np.stack(
                [fit_means[groups[fit] == g].mean(axis=0) for g in range(3)]
            )
            span, _ = np.linalg.qr(means.T)
            held_out = ((original @ span) * (swapped @ span)).sum(axis=1).mean()
            captured.append(float(held_out / energy))
    values = np.asarray(fractions)
    return {
        "fraction_mean": values.mean(axis=0).tolist(),
        "fraction_std": values.std(axis=0).tolist(),
        "cumulative_3": float(values[:, :3].sum(axis=1).mean()),
        "tail_after_3": float(values[:, 3:].sum(axis=1).mean()),
        "total_cross_fraction": float(np.mean(totals)),
        "mean_span_captured": float(np.mean(captured)),
    }


def split_half_stability(
    states: np.ndarray, groups: np.ndarray, rng: np.random.Generator, splits: int
) -> dict:
    """Agreement of mean axes estimated from disjoint sample halves."""
    cosines, angles = [], []
    for _ in range(splits):
        halves = [[], []]
        for group in range(3):
            members = rng.permutation(np.flatnonzero(groups == group))
            halves[0].extend(members[: len(members) // 2])
            halves[1].extend(members[len(members) // 2 :])
        estimates = []
        for half in halves:
            half = np.asarray(half)
            rows = states[half].reshape(-1, states.shape[-1])
            row_groups = np.repeat(groups[half], 2 * STYLE_COUNT)
            estimates.append(
                _unit(np.stack([rows[row_groups == g].mean(0) for g in range(3)]))
            )
        cosines.append(np.einsum("gd,gd->g", *estimates))
        angles.append(_principal_cosines(*estimates))
    return {
        "axis_cosine_mean": np.mean(cosines, axis=0).tolist(),
        "principal_cosine_mean": np.mean(angles, axis=0).tolist(),
    }


def axis_diagnostics(states: np.ndarray, groups: np.ndarray, spectra: dict) -> dict:
    dim = states.shape[-1]
    rows = states.reshape(-1, dim)
    row_groups = np.repeat(groups, 2 * STYLE_COUNT)
    norms = np.linalg.norm(rows, axis=1)
    means = spectra["means"]
    coherence, null = [], []
    for group in range(3):
        selected = row_groups == group
        mean_norm = norms[selected].mean()
        coherence.append(float(np.linalg.norm(means[group]) / mean_norm))
        rms = np.sqrt((norms[selected] ** 2).mean())
        null.append(float(rms / (mean_norm * np.sqrt(selected.sum()))))
    counts = np.bincount(row_groups, minlength=3)
    mean_energy = float(
        (counts * np.linalg.norm(means, axis=1) ** 2).sum() / (norms**2).sum()
    )
    unit_means = _unit(means)
    return {
        "coherence": coherence,
        "coherence_no_shared_direction": null,
        "mean_energy_fraction": mean_energy,
        "raw_energy_fraction": spectra["raw_energy_fraction"],
        "sample_energy_fraction": spectra["sample_energy_fraction"],
        "raw_top3_vs_mean_span_cosines": _principal_cosines(
            spectra["raw_basis"][:, :3].T, unit_means
        ),
        "mean_axis_cosines": (unit_means @ unit_means.T).tolist(),
    }


def coordinates(differences: np.ndarray, axes: np.ndarray, readout: str) -> np.ndarray:
    """Return H/V/D coordinates ``[N,3]`` of state differences ``[N,D]``."""
    projected = differences @ axes.T
    if readout == "project":
        return projected
    return np.linalg.solve(axes @ axes.T, projected.T).T


class Validation:
    """COCO-1800 differences, scored with the frozen margin-sign protocol."""

    def __init__(self, directory: Path) -> None:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Validation dump is incomplete: {directory}")
        self.layers = [int(value) for value in manifest["layers"]]
        self.states = np.load(directory / "differences.npy", mmap_mode="r")
        jobs = [
            json.loads(line)
            for line in (directory / "jobs.jsonl").read_text("utf-8").splitlines()
        ]
        if len(jobs) != self.states.shape[0]:
            raise ValueError("Validation job metadata differs from the state array")
        sample_ids = np.asarray([job["global_sample_index"] for job in jobs])
        self.samples, self.job_sample = np.unique(sample_ids, return_inverse=True)
        if np.any(np.bincount(self.job_sample) != STYLE_COUNT):
            raise ValueError("Every validation sample needs all prompt styles")
        group_index = {name: index for index, name in enumerate(AXIS_ORDER)}
        self.job_group = np.asarray([group_index[job["group"]] for job in jobs])
        self.groups = np.zeros(len(self.samples), dtype=np.int64)
        self.positive = np.zeros(len(self.samples), dtype=bool)
        for job, sample in zip(jobs, self.job_sample, strict=True):
            self.groups[sample] = group_index[job["group"]]
            self.positive[sample] = job["label"] == POSITIVE_ENDPOINT[job["group"]]

    def layer_differences(self, layer: int, orientation: int = 0) -> np.ndarray:
        position = self.layers.index(layer)
        return np.asarray(self.states[:, orientation, position, :], dtype=np.float64)

    def correct(
        self, differences: np.ndarray, axes: np.ndarray, readout: str
    ) -> np.ndarray:
        """Per-sample correctness after averaging the five style margins."""
        values = coordinates(differences, axes, readout)
        margins = values[np.arange(len(values)), self.job_group]
        scores = np.bincount(self.job_sample, weights=margins) / STYLE_COUNT
        return (scores != 0.0) & ((scores > 0.0) == self.positive)

    def margins(self, differences: np.ndarray, axes: np.ndarray) -> np.ndarray:
        return np.einsum("jd,jd->j", differences, axes[self.job_group])


def select_layer(correct: np.ndarray, groups: np.ndarray, layers: list[int]) -> int:
    """Overall accuracy, then worst-axis accuracy, then the shallower layer."""
    overall = correct.mean(axis=0)
    worst = np.min([correct[groups == g].mean(axis=0) for g in range(3)], axis=0)
    return max(range(len(layers)), key=lambda i: (overall[i], worst[i], -layers[i]))


def cross_fitted(
    correct: np.ndarray,
    groups: np.ndarray,
    positive: np.ndarray,
    layers: list[int],
    seeds: int,
) -> np.ndarray:
    """Per-sample correctness with the layer chosen on the other fold.

    Two stratified folds; repeated over ``seeds`` and averaged, so the result
    is a fractional correctness per sample free of selection bias.
    """
    total = np.zeros(correct.shape[0])
    strata = groups * 2 + positive
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        fold = np.zeros(correct.shape[0], dtype=np.int64)
        for stratum in np.unique(strata):
            members = rng.permutation(np.flatnonzero(strata == stratum))
            fold[members[1::2]] = 1
        for held_out in (0, 1):
            fit = fold != held_out
            chosen = select_layer(correct[fit], groups[fit], layers)
            total[~fit] += correct[~fit, chosen]
    return total / seeds


def summarize_accuracy(correct: np.ndarray, groups: np.ndarray) -> dict:
    axis = [float(correct[groups == g].mean()) for g in range(3)]
    return {
        "overall": float(correct.mean()),
        "axis": axis,
        "worst_axis": min(axis),
    }


def paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, draws: int, seed: int
) -> dict:
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    resampled = rng.integers(0, len(difference), size=(draws, len(difference)))
    values = difference[resampled].mean(axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return {"mean": float(difference.mean()), "ci95": [float(low), float(high)]}


class TrainingGradients:
    """Per-sample role gradients from every sealed worker shard."""

    def __init__(self, run_dir: Path) -> None:
        directories = sorted((run_dir / "shards").glob("rank_*"))
        if not directories:
            raise FileNotFoundError(f"No worker shards under {run_dir}")
        self.arrays, records = [], []
        for directory in directories:
            manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError(f"Worker shard is incomplete: {directory}")
            array = np.load(directory / "role_gradients.npy", mmap_mode="r")
            lines = (directory / "records.jsonl").read_text("utf-8").splitlines()
            if len(lines) != array.shape[0] * 2:
                raise ValueError(f"Record count differs from gradients: {directory}")
            self.arrays.append(array)
            records.extend(json.loads(line) for line in lines)
        group_index = {name: index for index, name in enumerate(AXIS_ORDER)}
        style_index = {name: index for index, name in enumerate(PROMPT_STYLE_ORDER)}
        sample = np.asarray([record["global_sample_index"] for record in records])
        orientation = np.asarray([record["orientation_index"] for record in records])
        style = np.asarray([style_index[record["style"]] for record in records])
        self.order = np.lexsort((style, orientation, sample))
        sorted_samples = sample[self.order].reshape(-1, 2 * STYLE_COUNT)
        if np.any(sorted_samples != sorted_samples[:, :1]):
            raise ValueError("Every training sample needs 2 orientations x 5 styles")
        row_groups = np.asarray([group_index[record["group"]] for record in records])
        self.groups = row_groups[self.order][:: 2 * STYLE_COUNT]
        self.dim = self.arrays[0].shape[-1]

    def layer(self, layer: int) -> np.ndarray:
        """Return structured gradients ``[S,2,5,D]`` in float64."""
        rows = np.concatenate(
            [
                np.asarray(array[:, :, layer, :], dtype=np.float64).reshape(
                    -1, self.dim
                )
                for array in self.arrays
            ]
        )
        return rows[self.order].reshape(-1, 2, STYLE_COUNT, self.dim)


def _released_axes(directory: Path | None) -> tuple[dict[int, np.ndarray], int | None]:
    if directory is None:
        return {}, None
    from safetensors.numpy import load_file

    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    axes = load_file(directory / "tensors.safetensors")["axes"]
    layers = manifest["model"]["layer_ids"]
    selected = manifest["construction"]["validation"]["selected_layer"]
    return {int(layer): axes[i] for i, layer in enumerate(layers)}, int(selected)


def _variant_names(sets: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    return [(name, readout) for name in sets for readout in READOUTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-states", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--released-artifact", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alphas", default="0.01,0.1,1")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cv-dims", type=int, default=12)
    parser.add_argument("--crossfit-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--efficiency-sizes", default="10,25,50,100")
    parser.add_argument("--efficiency-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alphas = [float(value) for value in args.alphas.split(",")]
    rng = np.random.default_rng(args.seed)
    training = TrainingGradients(args.run_dir)
    validation = Validation(args.validation_states)
    with np.load(args.run_dir / "merged/axes.npz", allow_pickle=False) as archive:
        merged_axes = archive["axes"].astype(np.float64)
        merged_layers = archive["layer_ids"].tolist()
    layers = [layer for layer in merged_layers if layer in validation.layers]
    if args.layers != "all":
        requested = {int(value) for value in args.layers.split(",")}
        layers = [layer for layer in layers if layer in requested]
    if not layers:
        raise ValueError("No layer is shared by the run and the validation dump")
    released, released_selected = _released_axes(args.released_artifact)
    if released and next(iter(released.values())).shape[-1] != training.dim:
        raise ValueError("Released artifact hidden size differs from the run")
    stored_margins = None
    if (args.run_dir / "validation/margins.npy").is_file():
        stored_margins = np.load(args.run_dir / "validation/margins.npy")

    args.out.mkdir(parents=True, exist_ok=True)
    per_layer, correct, all_axes = {}, {}, {}
    for layer in layers:
        states = training.layer(layer)
        sets, spectra = build_axis_sets(states, training.groups, alphas)
        merged = _unit(merged_axes[merged_layers.index(layer)])
        record = {
            "diagnostics": axis_diagnostics(states, training.groups, spectra),
            "cv_spectrum": cross_validated_spectrum(
                states, training.groups, rng, args.splits, args.cv_dims
            ),
            "split_half": split_half_stability(
                states, training.groups, rng, args.splits
            ),
            "cosine_to_mean": {
                name: np.einsum("gd,gd->g", axes, sets["mean"]).tolist()
                for name, axes in sets.items()
            },
            "mean_vs_merged_cosine": np.einsum(
                "gd,gd->g", sets["mean"], merged
            ).tolist(),
        }
        if layer in released:
            record["mean_vs_released_cosine"] = np.einsum(
                "gd,gd->g", sets["mean"], _unit(released[layer].astype(np.float64))
            ).tolist()
        differences = validation.layer_differences(layer)
        if stored_margins is not None:
            ours = validation.margins(differences, merged.astype(np.float32))
            stored = stored_margins[:, 0, merged_layers.index(layer)]
            record["reproduction_max_abs_margin_diff"] = float(
                np.max(np.abs(ours - stored))
            )
            record["reproduction_margin_scale"] = float(np.max(np.abs(stored)))
        for name, axes in sets.items():
            all_axes.setdefault(name, []).append(axes.astype(np.float32))
            for readout in READOUTS:
                correct.setdefault((name, readout), []).append(
                    validation.correct(differences, axes, readout)
                )
        per_layer[layer] = record
        print(f"layer {layer}: done", flush=True)

    groups, positive = validation.groups, validation.positive
    variants = {key: np.stack(value, axis=1) for key, value in correct.items()}
    baseline_key = ("mean", "project")
    baseline_layer = select_layer(variants[baseline_key], groups, layers)
    honest = {
        key: cross_fitted(value, groups, positive, layers, args.crossfit_seeds)
        for key, value in variants.items()
    }
    comparison = {}
    for key, value in variants.items():
        chosen = select_layer(value, groups, layers)
        comparison["/".join(key)] = {
            "selected_layer": layers[chosen],
            "selected": summarize_accuracy(value[:, chosen], groups),
            "at_baseline_layer": summarize_accuracy(value[:, baseline_layer], groups),
            "cross_fitted": summarize_accuracy(honest[key], groups),
            "cross_fitted_minus_baseline": paired_bootstrap(
                honest[key], honest[baseline_key], args.bootstrap, args.seed
            ),
            "per_layer_overall": value.mean(axis=0).tolist(),
        }

    reference_layer = layers[baseline_layer]
    states = training.layer(reference_layer)
    differences = validation.layer_differences(reference_layer)
    efficiency = {}
    for size in (int(value) for value in args.efficiency_sizes.split(",")):
        if size > min(np.bincount(training.groups, minlength=3)):
            continue
        runs: dict[str, list[list[float]]] = {}
        for _ in range(args.efficiency_repeats):
            chosen = np.concatenate(
                [
                    rng.choice(np.flatnonzero(training.groups == g), size, False)
                    for g in range(3)
                ]
            )
            sets, _ = build_axis_sets(states[chosen], training.groups[chosen], alphas)
            for name, readout in _variant_names(sets):
                value = validation.correct(differences, sets[name], readout)
                summary = summarize_accuracy(value, groups)
                runs.setdefault(f"{name}/{readout}", []).append(
                    [summary["overall"], summary["worst_axis"]]
                )
        efficiency[str(size)] = {
            key: {
                "overall_mean": float(np.mean(value, axis=0)[0]),
                "overall_std": float(np.std(value, axis=0)[0]),
                "worst_axis_mean": float(np.mean(value, axis=0)[1]),
            }
            for key, value in runs.items()
        }
        print(f"efficiency n={size}: done", flush=True)

    for name, value in all_axes.items():
        np.savez(
            args.out / f"axes_{name}.npz",
            axes=np.stack(value),
            layer_ids=np.asarray(layers, dtype=np.int64),
        )
    report = {
        "run_dir": str(args.run_dir),
        "validation_states": str(args.validation_states),
        "training_samples_per_axis": np.bincount(training.groups).tolist(),
        "validation_samples": len(groups),
        "layers": layers,
        "baseline_selected_layer": reference_layer,
        "released_selected_layer": released_selected,
        "comparison": comparison,
        "crossfit_seeds": args.crossfit_seeds,
        "efficiency_layer": reference_layer,
        "efficiency": efficiency,
        "per_layer": {str(layer): value for layer, value in per_layer.items()},
    }
    atomic_json(args.out / "report.json", report)
    (args.out / "summary.md").write_text(render_summary(report), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")


def render_summary(report: dict) -> str:
    """Render the decision-relevant numbers as Markdown tables."""
    layer = report["baseline_selected_layer"]
    counts = report["training_samples_per_axis"]
    released_layer = report["released_selected_layer"]
    seeds = report["crossfit_seeds"]
    lines = [
        "# Spectral subspace Phase 0",
        "",
        f"- 训练样本/轴: {counts}，验证样本: {report['validation_samples']}",
        f"- 基线选层: L{layer}（发布 artifact 选层: L{released_layer}）",
        "",
        "## 读出准确率",
        "",
        (
            f"交叉拟合 = 在一半验证集上选层、在另一半上测，{seeds} 次随机划分"
            "取平均；Δ 为相对 mean/project 的配对 bootstrap 95% CI。"
        ),
        "",
        (
            "| 轴 / 读出 | 选层 | 选层准确率 | 基线层准确率 "
            "| 交叉拟合 | 最差轴 | Δ (95% CI) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for key, value in report["comparison"].items():
        delta = value["cross_fitted_minus_baseline"]
        low, high = delta["ci95"]
        lines.append(
            f"| {key} | L{value['selected_layer']} "
            f"| {value['selected']['overall']:.4f} "
            f"| {value['at_baseline_layer']['overall']:.4f} "
            f"| {value['cross_fitted']['overall']:.4f} "
            f"| {value['cross_fitted']['worst_axis']:.4f} "
            f"| {delta['mean']:+.4f} [{low:+.4f}, {high:+.4f}] |"
        )
    record = report["per_layer"][str(layer)]
    diagnostics = record["diagnostics"]
    cv = record["cv_spectrum"]
    pairs = diagnostics["mean_axis_cosines"]
    reproduction = record.get("reproduction_max_abs_margin_diff")
    scale = record.get("reproduction_margin_scale")
    coherence = _fmt(diagnostics["coherence"])
    coherence_null = _fmt(diagnostics["coherence_no_shared_direction"])
    raw_angles = _fmt(diagnostics["raw_top3_vs_mean_span_cosines"])
    pair_cosines = _fmt([pairs[0][1], pairs[0][2], pairs[1][2]])
    lines += [
        "",
        f"## 诊断（L{layer}）",
        "",
        f"- 与 merged 轴余弦: {_fmt(record['mean_vs_merged_cosine'])}",
        f"- 与发布轴余弦: {_fmt(record.get('mean_vs_released_cosine', []))}",
        f"- 复现误差 max|Δmargin|: {reproduction}（margin 量级 {scale}）",
        f"- 方向一致性 H/V/D: {coherence}（无共享方向时约 {coherence_null}）",
        f"- 均值能量占比: {diagnostics['mean_energy_fraction']:.4f}",
        f"- 原始 SVD 前 6 能量: {_fmt(diagnostics['raw_energy_fraction'][:6])}",
        f"- 原始 SVD top-3 与均值子空间主角余弦: {raw_angles}",
        f"- 均值轴两两余弦 HV/HD/VD: {pair_cosines}",
        f"- 角色一致能量（orig×swap，前 8 维）: {_fmt(cv['fraction_mean'][:8])}",
        (
            f"- 前 3 维累计 {cv['cumulative_3']:.4f}，"
            f"其后 {cv['tail_after_3']:.4f}；"
            f"总角色一致能量 {cv['total_cross_fraction']:.4f}；"
            f"均值轴子空间捕获 {cv['mean_span_captured']:.4f}"
        ),
        f"- 折半轴余弦 H/V/D: {_fmt(record['split_half']['axis_cosine_mean'])}",
        "",
        "## 与 mean 轴的余弦（H/V/D）",
        "",
    ]
    for name, value in record["cosine_to_mean"].items():
        lines.append(f"- {name}: {_fmt(value)}")
    lines += [
        "",
        f"## 样本效率（L{report['efficiency_layer']}，每轴 n 个样本）",
        "",
    ]
    efficiency = report["efficiency"]
    if efficiency:
        keys = list(next(iter(efficiency.values())))
        lines.append("| n | " + " | ".join(keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        for size, values in efficiency.items():
            cells = [
                f"{values[key]['overall_mean']:.3f}±{values[key]['overall_std']:.3f}"
                for key in keys
            ]
            lines.append(f"| {size} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


if __name__ == "__main__":
    main()
