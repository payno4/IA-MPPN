from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def _build_permutations(P: int, perm_limit: Optional[int], seed: int) -> List[Tuple[int, ...]]:
    perms = list(itertools.permutations(range(P)))
    if perm_limit is not None and perm_limit < len(perms):
        rng = np.random.default_rng(seed)
        choose = rng.choice(len(perms), size=perm_limit, replace=False)
        perms = [perms[i] for i in choose]
    return perms


def _compute_auc(curve: np.ndarray) -> float:
    x_ax = np.arange(len(curve), dtype=np.float32)
    return float(np.trapz(curve, x_ax) / (len(curve) - 1))  # type: ignore


def deletion_insertion_metric(
    model,
    x_batch: torch.Tensor,              # [B,P,H,W]
    explanation_batch: torch.Tensor,    # [B,P,H,W]
    mode: str,                          # "del" or "ins"
    step: int,
    substrate_fn: Callable[[torch.Tensor], torch.Tensor],
    B_limit: Optional[int] = 1,         # e.g. 100
    perm_limit: Optional[int] = None,   # None = all P! perms, or int to subsample -> Monte Carlo Approximation
    seed: int = 0,                      # used only if perm_limit is not None
    use_gpu_sort: bool = True,          # keep sorting/indexing on GPU
):
    """Permutation-averaged per-perspective deletion/insertion.

    Returns:
        all_scores[p]: list of curves (np.ndarray) per original perspective p
        all_scores_norm[p]: same curves normalized
        summary: per-perspective mean/std AUC (+ overall)
    """
    if mode not in {"del", "ins"}:
        raise ValueError("mode must be 'del' or 'ins'")
    if step <= 0:
        raise ValueError("step must be a positive integer")

    B, P, H, W = x_batch.shape
    HW = H * W
    if B_limit is not None:
        B = min(B, B_limit)

    perms = _build_permutations(P, perm_limit, seed)
    n_perms = len(perms)
    n_steps = (HW + step - 1) // step

    all_scores: List[List[np.ndarray]] = [[] for _ in range(P)]
    all_scores_norm: List[List[np.ndarray]] = [[] for _ in range(P)]
    aucs: List[List[float]] = [[] for _ in range(P)]
    aucs_norm: List[List[float]] = [[] for _ in range(P)]

    total_iterations = B * P * n_perms
    device = x_batch.device
    use_gpu_sort = use_gpu_sort and (device.type == "cuda")

    print(f"Running {mode} metric: B={B}, P={P}, n_perms={n_perms}")

    with tqdm(total=total_iterations, desc=f"Computing {mode} curves (perm avg)") as pbar:
        for b in range(B):
            x = x_batch[b]          # [P,H,W] on device
            x_base = substrate_fn(x)  # [P,H,W] on device

            if mode == "del":
                start0 = x
                finish = x_base
            else:
                start0 = x_base
                finish = x

            with torch.no_grad():
                ref = model(x.unsqueeze(0), evaluation=False)[0].view(-1)[0]
                base = model(x_base.unsqueeze(0), evaluation=False)[0].view(-1)[0]
            denom = (ref - base).abs() + 1e-8

            for perm in perms:
                for p in range(P):
                    current_idx = perm[p]  # original perspective index

                    exp = explanation_batch[b, current_idx].detach()  # [H,W]
                    if use_gpu_sort:
                        order = torch.argsort(exp.reshape(-1), descending=True)
                    else:
                        order = np.argsort(exp.cpu().numpy().reshape(-1))[::-1].copy()

                    start = start0.clone()  # [P,H,W]
                    scores = torch.empty(n_steps + 1, device=device, dtype=torch.float32)
                    scores_norm = torch.empty(n_steps + 1, device=device, dtype=torch.float32)

                    for i in range(n_steps + 1):
                        with torch.no_grad():
                            out = model(start.unsqueeze(0), evaluation=False)[0].view(-1)[0]

                        d = (out - ref).abs()
                        scores[i] = d
                        scores_norm[i] = d / denom

                        if i < n_steps:
                            idx = order[i * step:(i + 1) * step]
                            s = start[current_idx].view(-1)
                            f = finish[current_idx].view(-1)
                            s[idx] = f[idx]

                    scores_np = scores.detach().cpu().numpy()
                    scores_norm_np = scores_norm.detach().cpu().numpy()

                    all_scores[current_idx].append(scores_np)
                    all_scores_norm[current_idx].append(scores_norm_np)
                    aucs[current_idx].append(_compute_auc(scores_np))
                    aucs_norm[current_idx].append(_compute_auc(scores_norm_np))

                    pbar.update(1)

    summary = {"per_perspective": [], "overall": {}, "settings": {}}
    per_p_mean_auc = []
    per_p_mean_auc_norm = []

    for p in range(P):
        m_auc = float(np.mean(aucs[p])) if len(aucs[p]) else float("nan")
        s_auc = float(np.std(aucs[p])) if len(aucs[p]) else float("nan")
        m_auc_n = float(np.mean(aucs_norm[p])) if len(aucs_norm[p]) else float("nan")
        s_auc_n = float(np.std(aucs_norm[p])) if len(aucs_norm[p]) else float("nan")

        summary["per_perspective"].append(
            {"p": p, "auc_mean": m_auc, "auc_std": s_auc, "auc_norm_mean": m_auc_n, "auc_norm_std": s_auc_n}
        )
        per_p_mean_auc.append(m_auc)
        per_p_mean_auc_norm.append(m_auc_n)

    summary["overall"]["auc_mean_over_p"] = float(np.mean(per_p_mean_auc)) if P else float("nan")
    summary["overall"]["auc_norm_mean_over_p"] = float(np.mean(per_p_mean_auc_norm)) if P else float("nan")
    summary["settings"] = {
        "mode": mode,
        "step": step,
        "B_used": B,
        "n_perms": n_perms,
        "perm_limit": perm_limit,
        "use_gpu_sort": bool(use_gpu_sort),
    }

    return all_scores, all_scores_norm, summary


def _plot_mean_curves(
    curves_by_p: List[List[np.ndarray]],
    auc_key: str,
    summary: dict,
    mode: str,
    label: str,
    ylabel: str,
    title_suffix: str,
    save_path: str,
) -> None:
    P = len(curves_by_p)
    plt.figure()
    for p in range(P):
        curves = np.stack(curves_by_p[p], axis=0)
        mean_curve = curves.mean(axis=0)

        x_percent = np.linspace(0, 100, len(mean_curve))
        auc_m = summary["per_perspective"][p][auc_key]
        plt.plot(x_percent, mean_curve, label=f"P{p} {auc_key.upper()}={auc_m:.4f}")

    plt.xlabel(f"% pixels {'removed' if mode=='del' else 'inserted'}")
    plt.ylabel(ylabel)
    plt.title(
        f"{label} {mode.upper()} {title_suffix} | "
        f"Overall AUC={summary['overall']['auc_mean_over_p']:.4f} | "
        f"Overall nAUC={summary['overall']['auc_norm_mean_over_p']:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def plot_deletion_insertion_curves(
    all_scores: List[List[np.ndarray]],
    all_scores_norm: List[List[np.ndarray]],
    summary: dict,
    mode: str,
    label: str = "",
) -> None:
    """Plot mean deletion/insertion curves and their normalized variants."""
    _plot_mean_curves(
        curves_by_p=all_scores,
        auc_key="auc_mean",
        summary=summary,
        mode=mode,
        label=label,
        ylabel="|f(x) - f(x_ref)|",
        title_suffix="Curves",
        save_path=f"{label}_{mode}_curve.png",
    )

    _plot_mean_curves(
        curves_by_p=all_scores_norm,
        auc_key="auc_norm_mean",
        summary=summary,
        mode=mode,
        label=label,
        ylabel="|f(x)-f(x_ref)| / |f(x)-f(x_base)|",
        title_suffix="Normalized Curves",
        save_path=f"{label}_{mode}_norm_curve.png",
    )


def plot_del_ins_diff(
    curves_del_by_method,
    curves_ins_by_method,
    labels=None,
    title="Diff between Insertion and Deletion Scores",
    save_path=None,
):
    """Plot the difference between insertion and deletion mean curves."""

    def _mean_curve_from_list(curves_list):
        min_len = min(len(c) for c in curves_list)
        trimmed = [np.asarray(c, dtype=np.float32)[:min_len] for c in curves_list]
        return np.stack(trimmed, axis=0).mean(axis=0)

    def _to_mean_curve(x):
        if isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 1:
                raise ValueError(f"Expected 1D mean curve, got shape {x.shape}")
            return x

        if not isinstance(x, (list, tuple)) or len(x) == 0:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 1:
                raise ValueError(f"Expected 1D mean curve, got shape {x.shape}")
            return x

        if isinstance(x[0], np.ndarray):
            return _mean_curve_from_list(x)

        if isinstance(x[0], (list, tuple)):
            flat = [curve for per_p in x for curve in per_p]
            if len(flat) == 0:
                raise ValueError("No curves found after flattening per-perspective container.")
            return _mean_curve_from_list(flat)

        raise ValueError("Unsupported curves container structure.")

    m = len(curves_del_by_method)
    if labels is None:
        labels = [f"method_{i}" for i in range(m)]

    plt.figure()
    for i in range(m):
        del_mean = _to_mean_curve(curves_del_by_method[i])  # 1D
        ins_mean = _to_mean_curve(curves_ins_by_method[i])  # 1D

        T = min(len(del_mean), len(ins_mean))
        del_mean = del_mean[:T]
        ins_mean = ins_mean[:T]

        x_percent = np.linspace(0, 100, T, dtype=np.float32)
        diff = ins_mean - del_mean  # 1D

        auc = float(np.trapz(diff, x_percent) / 100.0) # type: ignore
        plt.plot(x_percent, diff, label=f"{labels[i]} AUC={auc:.4f}")

    plt.xlabel("% inserted / deleted")
    plt.ylabel("Insertion - Deletion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        save_path = "del_ins_diff.png"
    plt.savefig(save_path, dpi=150)

