import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config.logger import setup_logger
from src.config.settings import settings

logger = setup_logger(__name__)

class EmotionDataLoader:
    """Handles data loading and augmentations for the Emotion dataset."""
    
    def __init__(self, data_dir: str = str(settings.DATA_DIR), batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self._setup_transforms()

    def _setup_transforms(self):
        # Advanced Data Augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Standard transformations for testing
        self.test_transform = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Creates and returns the training and testing dataloaders."""
        if not os.path.exists(self.train_dir) or not os.path.exists(self.test_dir):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. "
                "Please ensure you have placed the 'train' and 'test' folders inside it."
            )

        train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        test_data = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, test_loader

def get_dataloaders(data_dir=str(settings.DATA_DIR), batch_size=64):
    """Legacy helper function to maintain backwards compatibility if needed."""
    loader = EmotionDataLoader(data_dir=data_dir, batch_size=batch_size)
    return loader.get_dataloaders()
