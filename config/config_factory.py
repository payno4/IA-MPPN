from pathlib import Path

from dataset_loader.event_log_dataset import EventLogDataset
from config.dataloader_config import DataloaderConfig
from torchvision import transforms


class ConfigFactory:

    def __init__(self, config: DataloaderConfig, h5_directory: str, verbose: bool = True):
        self._config = config
        self.h5_directory = Path(h5_directory)
        self.verbose = verbose
    
    def _get_transformation(self, name:str):
        if name == "none":
            return None
        elif name == "random_affine":
            augm_transform = transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), translate=(0.02, 0.02))
            return augm_transform
        elif name == "strong_augmentation":
            # Neu: Stärkere Augmentation
            augm_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=15,           # ← Mehr Rotation
                    scale=(0.85, 1.15),   # ← Mehr Zoom
                    translate=(0.1, 0.1)  # ← Mehr Translation
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
            return augm_transform
        else:
            raise ValueError(f"Transformation '{name}' not recognized.")
        
    def build_dataset(self):
        h5_path = self.h5_directory / f"{self._config.h5_file_name}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Dataset file {h5_path} not found.")
        
        transform = self._get_transformation(self._config.transform)
        train_dataset = EventLogDataset(h5_path, mode="train", config=self._config, verbose=self.verbose, transform=transform) # type: ignore
        val_dataset = EventLogDataset(h5_path, mode="val", config=self._config, verbose=self.verbose, transform=None) # type: ignore
        test_dataset = EventLogDataset(h5_path, mode="test",config=self._config, verbose=self.verbose, transform=None) # type: ignore
        return train_dataset, val_dataset, test_dataset

