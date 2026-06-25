import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from model import EmotionCNN_Attention
import argparse
import logging
from logger_config import setup_logger

logger = setup_logger(__name__)

def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA for training.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")

    # Load data
    logger.info("Loading data...")
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    num_classes = len(train_loader.dataset.classes)
    logger.info(f"Detected {num_classes} classes: {train_loader.dataset.classes}")

    # Initialize model, criterion, and optimizer
    model = EmotionCNN_Attention(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % 50 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{epochs}] Training Accuracy: {train_acc:.2f}% | Average Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        logger.info(f"Epoch [{epoch+1}/{epochs}] Validation Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')
            logger.info(f"--> Saved new best model with accuracy: {best_acc:.2f}%")

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Analytics Model")
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
