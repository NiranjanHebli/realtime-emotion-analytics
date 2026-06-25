#!/bin/bash

# Exit on any error
set -e

echo "Starting Real-Time Emotion Analytics setup..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Creating virtual environment using uv..."
uv venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies using uv..."
uv pip install -r requirements.txt

echo "Setting up the dataset..."
python scripts/setup_dataset.py

echo ""
echo "========================================================="
echo "Setup complete!"
echo "To start working, activate your virtual environment with:"
echo "source venv/bin/activate"
echo "========================================================="
