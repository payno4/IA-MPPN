from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F

from config.configs import Configs
from evaluation.qualitative_metrics import (
    deletion_insertion_metric,
    plot_del_ins_diff,
    plot_deletion_insertion_curves,
)
from evaluation.utils import load_lightning_model


class Metrics:
    """Compute qualitative metrics for IA-MPPN."""

    def __init__(self, configs: Configs) -> None:
        """Initialize model, trainer, and config."""
        self.configs = configs
        self.model_type: str = configs.eval_optim_config.evaluation.model_type
        model_path = configs.eval_optim_config.evaluation.model_path_1

        self.model, self.lightning_module = load_lightning_model(
            model_path=model_path,
            model_type=self.model_type,
            config=configs,
            lightning_datamodule=None
        )
        print(f"Loaded model for Quantus metrics: {self.model_type} from {model_path}")
        self.trainer = pl.Trainer(
            accelerator=configs.experiment_config.device,
            devices="auto",
            precision="16-mixed",
            logger=False,
            enable_progress_bar=False,
            limit_predict_batches=1.0 #permitLog 0.4 InterDec 1.0 DomesticDec 1.0
        )

        self.lightning_module.eval() # type: ignore

    def calculate_metrics(self, dataset) -> None:
        """Run metrics for the configured model type."""
        if self.model_type == "IA-MPPN":
            self._calculate_ia_mppn_metrics(dataset)
            return
        raise ValueError(f"Unsupported model type for Quantus metrics: {self.model_type}")

    def _process_outs(self, outs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert model outputs into attention, inputs, and targets."""
        #outs: list of (inputs, targets, logits, inter_logits, ca, features)
        # Check if relation attention exists by looking at first batch
        has_r_attn = (outs[0][4].relation_attention is not None)

        inputs_all = torch.cat([o[0] for o in outs], dim=0)   # [B,P,H,W]
        targets_all = torch.cat([o[1] for o in outs], dim=0)  # [B]
        
        # Collect perspective attentions from all batches
        p_attn_list = []
        for o in outs:
            ca = o[4]  # CollectedAttentions
            # Each p.new_atts has shape [B_batch, N, N]
            # Stack perspectives: [P, B_batch, N, N]
            batch_p_atts = torch.stack([p.new_atts for p in ca.perspective_attentions], dim=0)
            # Transpose to [B_batch, P, N, N]
            batch_p_atts = batch_p_atts.transpose(0, 1)
            p_attn_list.append(batch_p_atts)
        # Concatenate along batch dimension: [Total_B, P, N, N]
        p_attn_all = torch.cat(p_attn_list, dim=0)
        
        if has_r_attn:
            r_attn = torch.cat([o[4].relation_attention.new_atts for o in outs], dim=0)  # [B,N*P,N*P]

        _, P, H, W = inputs_all.shape

        ### process relation attention
        if has_r_attn:
            _, PN, _ = r_attn.shape
            N = PN // P                   # number of patches per perspective
            r_attn_per_p = torch.stack(
                [r_attn[:, c * N:(c + 1) * N, c * N:(c + 1) * N] for c in range(P)],
                dim=1,
            )

            ### fusion of relation and perspective attentions
            attn = p_attn_all * r_attn_per_p  # [B,P,N,N]
        else:
            attn = p_attn_all  # [B,P,N,N]
        
        attn_tok = attn.mean(dim=2)  # [B,P,N]
        n = attn_tok.shape[-1]
        g = int(math.isqrt(n))
        attn_grid = attn_tok.view(attn_tok.shape[0], attn_tok.shape[1], g, g)
        attn_up = F.interpolate(attn_grid, size=(H, W), mode="bilinear", align_corners=False)

        a_batch = attn_up
        x_batch = inputs_all   # [B,P,H,W]
        y_batch = targets_all  # [B]

        return a_batch, x_batch, y_batch
    
    def _calculate_ia_mppn_metrics(self, dataset) -> None:
        """Compute deletion/insertion metrics and save reports."""
        out_dir = self._prepare_output_dir()

        # 1) prediction -> (a_batch, x_batch)
        outs = self.trainer.predict(self.lightning_module, datamodule=dataset)
        a_batch, x_batch, _ = self._process_outs(outs)  # [B,P,H,W]

        model_device = self._get_model_device(x_batch)

        x_batch = x_batch.to(model_device)
        a_batch = a_batch.to(model_device).float()

        a_random = self._random_explanations(a_batch, model_device)

        self.model.eval()

        # config
        B_limit = 10
        step = 9  # 90x90=8100 pixels -> ~50 steps for smooth curves
        perm_limit = 18
        seed = self.configs.experiment_config.seed

        def substrate_fn(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        methods = {"original": a_batch, "random": a_random}

        run_summary: Dict = {
            "model_type": self.model_type,
            "B_limit": B_limit,
            "step": step,
            "perm_limit": perm_limit,
            "seed": seed,
            "out_dir": out_dir,
            "results": {},
        }

        del_by_method: List = []
        ins_by_method: List = []
        labels: List[str] = []

        for name, explanation_batch in methods.items():
            run_summary["results"][name] = {}

            # Deletion
            all_scores_del, all_scores_norm_del, summary_del = deletion_insertion_metric(
                model=self.model,
                x_batch=x_batch,
                explanation_batch=explanation_batch,
                mode="del",
                step=step,
                substrate_fn=substrate_fn,
                B_limit=B_limit,
                perm_limit=perm_limit,
                seed=seed,
                use_gpu_sort=True,
            )
            label = os.path.join(out_dir, f"{name}_{self.model_type}_AB{B_limit}")
            plot_deletion_insertion_curves(all_scores_del, all_scores_norm_del, summary_del, mode="del", label=label)

            # Insertion
            all_scores_ins, all_scores_norm_ins, summary_ins = deletion_insertion_metric(
                model=self.model,
                x_batch=x_batch,
                explanation_batch=explanation_batch,
                mode="ins",
                step=step,
                substrate_fn=substrate_fn,
                B_limit=B_limit,
                perm_limit=perm_limit,
                seed=seed,
                use_gpu_sort=True,
            )
            plot_deletion_insertion_curves(all_scores_ins, all_scores_norm_ins, summary_ins, mode="ins", label=label)

            # store summaries
            run_summary["results"][name]["del"] = summary_del
            run_summary["results"][name]["ins"] = summary_ins

            # For diff plot (treat each explanation source as one method)
            del_by_method.append(all_scores_norm_del)
            ins_by_method.append(all_scores_norm_ins)
            labels.append(name)

        # Diff plot into same folder
        diff_path = os.path.join(out_dir, f"{self.model_type}_ins_minus_del_norm.png")
        plot_del_ins_diff(
            curves_del_by_method=del_by_method,
            curves_ins_by_method=ins_by_method,
            labels=labels,
            title=f"{self.model_type}: Insertion - Deletion (normalized)",
            save_path=diff_path,
        )

        txt_path = self._write_text_report(run_summary, out_dir)
        json_path = self._write_json_report(run_summary, out_dir)

        print(f"[In/Del] Saved outputs to: {out_dir}")
        print(f"[In/Del] Text:  {txt_path}")
        print(f"[In/Del] JSON:  {json_path}")
        print(f"[In/Del] Diff:  {diff_path}")

    def _prepare_output_dir(self) -> str:
        """Create and return a timestamped output directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("results", "in_del", f"{self.model_type}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _get_model_device(self, fallback: torch.Tensor) -> torch.device:
        """Get model device or fall back to a tensor device."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return fallback.device

    def _random_explanations(self, a_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Generate random explanations with a fixed seed."""
        g = torch.Generator(device=device)
        g.manual_seed(self.configs.experiment_config.seed)
        return torch.rand(a_batch.shape, generator=g, device=device, dtype=a_batch.dtype)

    def _write_text_report(self, run_summary: Dict, out_dir: str) -> str:
        """Write a human-readable results summary."""
        txt_path = os.path.join(out_dir, "results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.model_type}\n")
            f.write(
                f"B_limit={run_summary['B_limit']}, step={run_summary['step']}, "
                f"perm_limit={run_summary['perm_limit']}, seed={run_summary['seed']}\n\n"
            )

            def _write_section(method_name: str, mode_name: str, summ: Dict) -> None:
                f.write(f"[{method_name} | {mode_name}]\n")
                f.write(
                    f"Overall auc_mean_over_p: {summ['overall'].get('auc_mean_over_p', float('nan')):.6f}\n"
                )
                f.write(
                    f"Overall auc_norm_mean_over_p: {summ['overall'].get('auc_norm_mean_over_p', float('nan')):.6f}\n"
                )
                f.write("Per perspective:\n")
                for row in summ.get("per_perspective", []):
                    p = row.get("p")
                    n = row.get("n", row.get("n_samples", ""))
                    auc_m = row.get("auc_mean", float("nan"))
                    auc_s = row.get("auc_std", float("nan"))
                    auc_se = row.get("auc_se", float("nan"))
                    aucn_m = row.get("auc_norm_mean", float("nan"))
                    aucn_s = row.get("auc_norm_std", float("nan"))
                    aucn_se = row.get("auc_norm_se", float("nan"))

                    f.write(
                        f"  P{p}: n={n} "
                        f"AUC={auc_m:.6f} (std={auc_s:.6f}, se={auc_se:.6f}) "
                        f"nAUC={aucn_m:.6f} (std={aucn_s:.6f}, se={aucn_se:.6f})\n"
                    )
                f.write("\n")

            for method_name in ["original", "random"]:
                _write_section(method_name, "del", run_summary["results"][method_name]["del"])
                _write_section(method_name, "ins", run_summary["results"][method_name]["ins"])

        return txt_path

    def _write_json_report(self, run_summary: Dict, out_dir: str) -> str:
        """Write results to JSON for later analysis."""
        json_path = os.path.join(out_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        return json_path



def shuffle_explanations(a: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle explanation values per sample (and channel)."""
    rng = np.random.default_rng(seed)
    a_shuf = a.copy()
    if a.ndim == 4:
        B, C, H, W = a.shape
        for b in range(B):
            for c in range(C):
                flat = a_shuf[b, c].reshape(-1)
                rng.shuffle(flat)
                a_shuf[b, c] = flat.reshape(H, W)
    elif a.ndim == 2:
        B, N = a.shape
        for b in range(B):
            perm = rng.permutation(N)
            a_shuf[b] = a_shuf[b, perm]
    return a_shuf
