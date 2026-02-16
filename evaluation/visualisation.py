from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Trainer

from config.configs import Configs
from evaluation.utils import load_lightning_model
from models.IA_MPPN.ia_mppn import CollectedAttentions


class Visualisation:
    def __init__(self, model_path: str, configs: Configs) -> None:
        self.model_path = model_path
        self.configs = configs
        self.eval_config = configs.eval_optim_config.evaluation
        self.model_type: str = self.eval_config.model_type

        self.model, self.lightning_module = load_lightning_model(
            model_path=model_path,
            model_type=self.model_type,
            config=configs,
            lightning_datamodule=None
        )

        self.trainer = Trainer(
            accelerator=configs.experiment_config.device,
            devices="auto",
            precision="16-mixed",
            logger=False,
            enable_progress_bar=True,
            limit_predict_batches=1
        )
    
    def visualize(self, dataset) -> None:
        if self.model_type == "IA-MPPN":
            self._visualize_ia_mppn(dataset)
            return
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _visualize_ia_mppn(self, dataset, num_plots: int = 1) -> None:
        outs = self.trainer.predict(self.lightning_module, datamodule=dataset, return_predictions=True)
        first = outs[0] # type: ignore

        # expected: inputs, targets, logits, inter_logits, ca
        inputs, _, _, _, ca= first
        ca: CollectedAttentions

        # Collect interpreter attentions (patch-to-patch): [B, P, N, N]
        p_atts = torch.stack([p.new_atts for p in ca.perspective_attentions], dim=1) # type: ignore

        # Collect relation attention if available: [B, P*N, P*N]
        r_attn = None
        if ca.relation_attention is not None:
            r_attn = ca.relation_attention.new_atts

        raw_atts = self._collect_raw_attentions(ca)

        B, P, H, W = inputs.shape
        for idx in range(min(B, num_plots)):
            x_i = inputs[idx]
            p_i = p_atts[idx]
            r_attn_sample = r_attn[idx] if r_attn is not None else None
            N = p_i.shape[-1]

            for perspective in range(P):
                feature = x_i[perspective].detach().cpu().numpy()

                teacher_patch_vec = self._get_teacher_patch_vector(raw_atts, perspective, idx)

                # Visualize interpreter patch-to-patch attention
                student_patch_vec = p_i[perspective].detach().cpu().mean(dim=1)
                student_heat = self._vec_to_heatmap(student_patch_vec, H, W)
                self._plot_attention_overlay(
                    feature,
                    student_heat.detach().cpu().numpy(),
                    label_1=f"Perspective {perspective} Feature",
                    label_2="Interpreter Patch-to-Patch Attention",
                    file_name=f"sample{idx}_perspective{perspective}_interpreter_attention.png",
                )

                # Visualize teacher patch-to-patch attention (from last layer, as used in training)
                if teacher_patch_vec is not None:
                    teacher_heat = self._vec_to_heatmap(teacher_patch_vec, H, W)
                    self._plot_attention_overlay(
                        feature,
                        teacher_heat.numpy(),
                        label_1=f"Perspective {perspective} Feature",
                        label_2="Teacher Patch-to-Patch Attention (Last Layer)",
                        file_name=f"sample{idx}_perspective{perspective}_teacher_patch_attention.png"
                    )

                    # Compare teacher vs interpreter (difference map)
                    diff_map = torch.abs(teacher_heat - student_heat).numpy()
                    self._plot_attention_comparison(
                        feature,
                        teacher_heat.numpy(),
                        student_heat.numpy(),
                        diff_map,
                        perspective_idx=perspective,
                        sample_idx=idx
                    )

                # Visualize teacher CLS rollout for comparison
                if raw_atts is not None:
                    raw_full = raw_atts[perspective, :, idx, :, :].detach().cpu()
                    R = self._attention_rollout(raw_full, add_residual=True)
                    cls_vec = R[0, 1:]
                    cls_heat = self._vec_to_heatmap(cls_vec, H, W)

                    self._plot_attention_overlay(
                        feature, cls_heat.numpy(),
                        label_1=f"Perspective {perspective} Feature",
                        label_2="Teacher CLS Rollout (All Layers)",
                        file_name=f"sample{idx}_perspective{perspective}_teacher_cls_rollout.png"
                    )
            
            # Visualize perspective importance from relation attention
            if r_attn_sample is not None:
                feat_imp = []
                for p in range(P):
                    block = r_attn_sample[p*N:(p+1)*N, p*N:(p+1)*N]  # [N,N]
                    feat_imp.append(block.sum().item())

                feat_imp = np.array(feat_imp, dtype=np.float64)
                perc = 100.0 * feat_imp / (feat_imp.sum() + 1e-12)

                labels = [f"Perspective {i}" for i in range(P)]
                plt.figure(figsize=(10, 6))
                bars = plt.bar(labels, perc)
                plt.ylabel("Importance (%)")
                plt.title("Perspective Importance from Relation Attention")
                
                # Add percentage values on top of bars
                for bar, percentage in zip(bars, perc):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{percentage:.1f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"sample{idx}_perspective_importance.png")
                plt.close()

    def _visualize_layer_attentions(self, feature, raw_atts, perspective_idx, sample_idx):
        """Visualize attention patterns for each layer separately."""
        H, W = feature.shape
        num_layers = len(raw_atts)
        
        fig, axes = plt.subplots(1, num_layers + 1, figsize=((num_layers + 1) * 3, 4))
        
        # Plot original feature
        im_feat = axes[0].imshow(feature, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Perspective {perspective_idx}\nFeature')
        axes[0].axis('off')
        plt.colorbar(im_feat, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot attention for each layer
        for i, layer in enumerate(raw_atts):
            cls_attn = layer[0, 1:]
            n_patches = cls_attn.shape[0]
            grid_size = int(np.sqrt(n_patches))
            
            attn_grid = cls_attn.view(grid_size, grid_size).unsqueeze(0).unsqueeze(0)
            attn_upsampled = F.interpolate(attn_grid, size=(H, W), mode="bilinear", align_corners=False)
            attn_upsampled = attn_upsampled.squeeze().cpu().numpy()
            attn_normalized = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
            
            axes[i + 1].imshow(feature, cmap='gray', aspect='auto')
            im_attn = axes[i + 1].imshow(attn_normalized, cmap='hot', alpha=0.6, aspect='auto')
            axes[i + 1].set_title(f'Layer {i}')
            axes[i + 1].axis('off')
            plt.colorbar(im_attn, ax=axes[i + 1], fraction=0.046, pad=0.04)
        
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        plt.savefig(f'sample{sample_idx}_perspective{perspective_idx}_layer_attentions.png', dpi=150, bbox_inches='tight')
        plt.close()


    def _plot_attention_overlay(self, feature, attention_map, label_1, label_2, file_name):
        """Plot feature and attention overlay side by side."""
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(feature.shape[1] / 5, feature.shape[0] / 10)
        )

        im1 = ax1.imshow(feature, cmap='viridis', aspect='auto')
        ax1.set_title(label_1)
        plt.colorbar(im1, ax=ax1)

        ax2.imshow(feature, cmap='gray', aspect='auto')
        im2 = ax2.imshow(attention_map, cmap='hot', alpha=0.5, aspect='auto')
        ax2.set_title(label_2)
        plt.colorbar(im2, ax=ax2)

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    def _plot_attention_comparison(self, feature, teacher_attn, student_attn, diff_map, perspective_idx, sample_idx):
        """Plot teacher vs interpreter attention with difference map."""
        fig, axes = plt.subplots(1, 4, figsize=(feature.shape[1] / 2.5, feature.shape[0] / 10))
        
        # Original feature
        im0 = axes[0].imshow(feature, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Perspective {perspective_idx}\nFeature')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Teacher attention
        axes[1].imshow(feature, cmap='gray', aspect='auto')
        im1 = axes[1].imshow(teacher_attn, cmap='hot', alpha=0.6, aspect='auto')
        axes[1].set_title('Teacher\nPatch-to-Patch')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Interpreter attention
        axes[2].imshow(feature, cmap='gray', aspect='auto')
        im2 = axes[2].imshow(student_attn, cmap='hot', alpha=0.6, aspect='auto')
        axes[2].set_title('Interpreter\nPatch-to-Patch')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Difference map
        im3 = axes[3].imshow(diff_map, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
        axes[3].set_title('Absolute\nDifference')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        plt.savefig(f'sample{sample_idx}_perspective{perspective_idx}_teacher_vs_interpreter.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _attention_rollout(self, attn_layers: torch.Tensor, add_residual: bool = True) -> torch.Tensor:
        """
        Compute attention rollout across layers.
        
        Args:
            attn_layers: Attention matrices [L, N, N] including CLS token
                        where L = num_layers, N = 1 + num_patches
            add_residual: Whether to add identity matrix (residual connection)
        
        Returns:
            Rollout matrix [N, N]
        """
        A = attn_layers.float().clone()
        L, N, _ = A.shape

        if add_residual:
            I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
            A = A + I

        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        R = A[0]
        for l in range(1, L):
            R = A[l] @ R
        return R

    def _collect_raw_attentions(self, ca: CollectedAttentions) -> Optional[torch.Tensor]:
        if ca.perspective_attentions is None:
            return None
        raw_list = []
        for p in ca.perspective_attentions:
            at = p.attn_weights
            if isinstance(at, list):
                at = torch.stack(at, dim=0)  # [L, B, H, N+1, N+1]
            at = at.mean(dim=2)  # [L, B, N+1, N+1]
            raw_list.append(at)
        return torch.stack(raw_list, dim=0)  # [P, L, B, N+1, N+1]

    def _get_teacher_patch_vector(
        self,
        raw_atts: Optional[torch.Tensor],
        perspective: int,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        if raw_atts is None:
            return None
        teacher_last_layer = raw_atts[perspective, -1, batch_idx, 1:, 1:].detach().cpu()
        return teacher_last_layer.mean(dim=1)

    def _vec_to_heatmap(self, vec_1d: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert 1D attention vector to upsampled 2D heatmap."""
        vec_1d = vec_1d.float()
        grid_size = int(np.sqrt(vec_1d.numel()))
        grid = vec_1d.view(grid_size, grid_size).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
