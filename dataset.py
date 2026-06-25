import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', batch_size=64):
    """
    Creates and returns the training and testing dataloaders for the Emotion dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Advanced Data Augmentation for training
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5), # Handle lighting changes
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Unique: Handle tilted faces
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Standard transformations for testing
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            "Please ensure you have downloaded FER-2013 and placed the 'train' and 'test' folders inside 'data/'."
        )

    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

if __name__ == "__main__":
    # Quick test to see if dataset is loaded correctly
    try:
        train_loader, test_loader = get_dataloaders()
        print(f"Successfully loaded {len(train_loader.dataset)} training images.")
        print(f"Successfully loaded {len(test_loader.dataset)} testing images.")
    except Exception as e:
        print(f"Error: {e}")
