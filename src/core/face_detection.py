import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from src.config.settings import settings
from src.config.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetectorService:
    """Handles face detection using MediaPipe Tasks API."""
    def __init__(self, model_path: str = str(settings.MEDIAPIPE_MODEL_PATH)):
        self.model_path = model_path
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options, 
            min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        if not os.path.exists(self.model_path):
            logger.info("Downloading MediaPipe Face Detection model...")
            urllib.request.urlretrieve(settings.MEDIAPIPE_MODEL_URL, self.model_path)

    def detect(self, rgb_image: np.ndarray):
        """
        Detect faces in an RGB image array.
        Returns MediaPipe DetectionResult.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.detector.detect(mp_image)
