import os
from pathlib import Path
import torch

class Settings:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    
    # Model configuration
    MODEL_PATH = BASE_DIR / 'model.pth'
    NUM_CLASSES = 4
    CLASSES = ['angry', 'happy', 'neutral', 'sad']
    
    # MediaPipe Face Detection
    MEDIAPIPE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
    MEDIAPIPE_MODEL_PATH = BASE_DIR / 'blaze_face_short_range.tflite'
    
    # Inference parameters
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    MIN_DETECTION_CONFIDENCE = 0.7
    EMOTION_CONFIDENCE_THRESHOLD = 0.6
    
    # Image processing
    IMAGE_SIZE = (48, 48)

settings = Settings()
