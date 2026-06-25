# Real-Time Emotion Analytics

**Real-Time Emotion Analytics** is a next-generation emotion analytics system that leverages MediaPipe 3D Face Mesh and a Spatial Attention-based Convolutional Neural Network (CNN) to provide real-time facial expression recognition and continuous emotion journey tracking.

## Unique Features

- **Spatial Attention CNN**: Focuses deep learning specifically on highly expressive regions (eyes, mouth) to reduce background noise.
- **MediaPipe Face Tracking**: Highly accurate 3D face mesh for robust facial landmark detection.
- **Emotion Journey Timeline**: Real-time analytics dashboard mapping emotional fluctuations throughout a session.
- **Active Learning Loop**: Interactive UI that allows the user to correct low-confidence predictions to continuously fine-tune the model.

## Installation and Setup

We use `uv` for fast dependency management. The provided setup script will automatically create a virtual environment, install all dependencies, and download the dataset.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd realtime-emotion-analytics
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

## Data Preparation

The `setup.sh` script automatically downloads and organizes the required Kaggle dataset (4 emotion classes: angry, happy, sad, neutral) using `kagglehub`. The images are placed into `data/train` and `data/test` folders so they are ready for PyTorch.
