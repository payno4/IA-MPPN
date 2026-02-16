from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as pl
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from config.configs import Configs
from evaluation.utils import load_lightning_model


class PerformanceEvaluator:

    def __init__(self, model_path: str, model_type: str, configs: Configs) -> None:
        """Initialize evaluator for a single model."""
        self.model_type = model_type
        self.configs = configs

        self.model, self.lightning_module = self._load_model(model_path, model_type, configs)

        self.trainer = pl.Trainer(
            accelerator=configs.experiment_config.device,
            devices="auto",
            precision="16-mixed",
            logger=False,
            enable_progress_bar=True,
            limit_predict_batches=1.0,  # 1.0 for InterDec, 0.4 for PermitLog DomDec
        )

        self.lightning_module.eval() # type: ignore
    
    def evaluate_performance(self, dataloader, output_dir: str = "results/performance") -> Dict:
        """Evaluate model performance (single run)."""
        print(f"Evaluating {self.model_type} performance...")
        mae, mse, r2 = self._eval_performance(dataloader)
        
        results = {
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "single_run",
            "metrics": {
                "MAE": float(mae),
                "MSE": float(mse),
                "R2": float(r2)
            }
        }

        # Print results
        self._print_single_run(results)

        # Save results
        self._save_results(results, output_dir)
        
        return results
    
    def evaluate_with_bootstrap(
        self,
        dataloader,
        num_bootstraps: int = 2500,
        confidence_level: float = 0.95,
        output_dir: str = "results/performance",
    ) -> Dict:
        """Evaluate model performance with bootstrap confidence intervals."""
        print(f"Evaluating {self.model_type} with bootstrap (n={num_bootstraps})...")
        y_hat, targets = self._collect_predictions(dataloader)
        n_samples = len(y_hat)
        
        print(f"Total samples: {n_samples}")
        
        # Bootstrap resampling
        bootstrap_maes: List[float] = []
        bootstrap_mses: List[float] = []
        bootstrap_r2s: List[float] = []
        
        for i in range(num_bootstraps):
            # Random sample with replacement
            indices = torch.randint(0, n_samples, (n_samples,))
            y_hat_boot = y_hat[indices]
            targets_boot = targets[indices]
            
            mae, mse, r2 = _compute_metrics(y_hat_boot, targets_boot)
            
            # Only add valid R2 values (skip if no variance in targets)
            if not (torch.isnan(torch.tensor(r2)) or torch.isinf(torch.tensor(r2))):
                bootstrap_maes.append(mae)
                bootstrap_mses.append(mse)
                bootstrap_r2s.append(r2)
            
            if (i + 1) % 100 == 0:
                print(f"Bootstrap progress: {i + 1}/{num_bootstraps} (valid R2s: {len(bootstrap_r2s)})")
        
        # Compute original metrics
        original_mae, original_mse, original_r2 = _compute_metrics(y_hat, targets)
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "bootstrap",
            "num_bootstraps": num_bootstraps,
            "confidence_level": confidence_level,
            "n_samples": n_samples,
            "metrics": {
                "MAE": {
                    "value": float(original_mae),
                    "mean": float(torch.tensor(bootstrap_maes).mean()),
                    "std": float(torch.tensor(bootstrap_maes).std()),
                    "ci_lower": float(torch.tensor(bootstrap_maes).quantile(lower_percentile / 100)),
                    "ci_upper": float(torch.tensor(bootstrap_maes).quantile(upper_percentile / 100))
                },
                "MSE": {
                    "value": float(original_mse),
                    "mean": float(torch.tensor(bootstrap_mses).mean()),
                    "std": float(torch.tensor(bootstrap_mses).std()),
                    "ci_lower": float(torch.tensor(bootstrap_mses).quantile(lower_percentile / 100)),
                    "ci_upper": float(torch.tensor(bootstrap_mses).quantile(upper_percentile / 100))
                },
                "R2": {
                    "value": float(original_r2),
                    "mean": float(torch.tensor(bootstrap_r2s).mean()),
                    "std": float(torch.tensor(bootstrap_r2s).std()),
                    "ci_lower": float(torch.tensor(bootstrap_r2s).quantile(lower_percentile / 100)),
                    "ci_upper": float(torch.tensor(bootstrap_r2s).quantile(upper_percentile / 100))
                }
            }
        }

        # Print results
        self._print_bootstrap(results, confidence_level)

        # Save results
        self._save_results(results, output_dir, suffix="bootstrap")
        
        return results

    def _eval_performance(self, dataloader) -> Tuple[float, float, float]:
        """Evaluate model performance on a single run."""
        y_hat, targets = self._collect_predictions(dataloader)
        return _compute_metrics(y_hat, targets)
    
    def _save_results(self, results: dict, output_dir: str, suffix: str = "") -> None:
        """Save evaluation results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"{self.model_type}_performance{suffix_str}_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")

    def _load_model(self, model_path: str, model_type: str, configs: Configs):
        if model_type == "IA-MPPN":
            from models.IA_MPPN.lightning_module_ia_mppn import IA_MPPNLightningModule
            return load_lightning_model(
                model_path=model_path,
                model_type=model_type,
                config=configs,
                lightning_datamodule=None,
            )
        if model_type == "MPPN":
            from models.MPPN.lightning_module_mppn import MPPNLightningModule
            return load_lightning_model(
                model_path=model_path,
                model_type=model_type,
                config=configs,
                lightning_datamodule=None,
            )
        raise ValueError(f"Unknown model type: {model_type}")

    def _collect_predictions(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.trainer.predict(self.lightning_module, datamodule=dataloader)
        y_hat: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        if self.model_type == "MPPN":
            for batch in predictions: # type: ignore
                _, pred, target = batch
                target = target.view(pred.shape)
                mask = (~torch.isnan(target)) & (~torch.isnan(pred))
                if mask.any():
                    y_hat.append(pred[mask])
                    targets.append(target[mask])
        elif self.model_type == "IA-MPPN":
            for batch in predictions: # type: ignore
                _, target, logits, *_ = batch
                target = target.view(logits.shape)
                mask = (~torch.isnan(target)) & (~torch.isnan(logits))
                if mask.any():
                    y_hat.append(logits[mask])
                    targets.append(target[mask])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return torch.cat(y_hat).cpu(), torch.cat(targets).cpu()

    def _print_single_run(self, results: Dict) -> None:
        print(f"\nModel: {self.model_type}")
        print(f"MAE: {results['metrics']['MAE']:.4f}")
        print(f"MSE: {results['metrics']['MSE']:.4f}")
        print(f"R2: {results['metrics']['R2']:.4f}")

    def _print_bootstrap(self, results: Dict, confidence_level: float) -> None:
        print(f"\n{'='*60}")
        print(f"Model: {self.model_type}")
        print(f"Bootstrap Results (n={results['num_bootstraps']}, CI={confidence_level*100:.0f}%)")
        print(f"{'='*60}")
        for metric in ["MAE", "MSE", "R2"]:
            m = results["metrics"][metric]
            print(f"{metric}:")
            print(f"  Original: {m['value']:.4f}")
            print(f"  Bootstrap Mean: {m['mean']:.4f} ± {m['std']:.4f}")
            print(f"  {confidence_level*100:.0f}% CI: [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")


def _compute_metrics(y_hat: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    """Compute MAE, MSE and R² metrics.
    
    Note: R² is computed on normalized data for numerical stability.
    With large denormalized ranges, floating-point precision errors can cause -inf.
    R² is scale-invariant, so normalized and denormalized values are mathematically equivalent.
    """
    mae = MeanAbsoluteError()(y_hat, targets).item()
    mse = MeanSquaredError()(y_hat, targets).item()
    
    # Compute R² on normalized data for numerical stability
    r2 = R2Score()(y_hat, targets).item()
    
    return mae, mse, r2
    

        

    