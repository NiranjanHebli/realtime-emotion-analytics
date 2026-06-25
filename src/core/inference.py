import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.emotion_cnn import EmotionCNN_Attention
from src.core.face_detection import FaceDetectorService
from src.config.settings import settings
from src.config.logger import setup_logger
import os

logger = setup_logger(__name__)

class EmotionInferenceEngine:
    """Core logic for orchestrating face detection and emotion classification."""
    def __init__(self, model_weights_path: str = str(settings.MODEL_PATH)):
        self.device = torch.device(settings.DEVICE)
        self.model = EmotionCNN_Attention(num_classes=settings.NUM_CLASSES).to(self.device)
        
        if os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model weights from {model_weights_path} onto {self.device}")
        else:
            logger.warning(f"{model_weights_path} not found. Using untrained weights.")
            
        self.face_detector = FaceDetectorService()
        self.classes = settings.CLASSES
        
        self.transform = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, float, np.ndarray | None]:
        """
        Processes a BGR frame, detects face, and predicts emotion.
        Returns: modified frame, emotion label, confidence, face_img (if detected)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = self.face_detector.detect(rgb_frame)
        
        predicted_emotion = "None"
        confidence = 0.0
        face_img = None
        
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                
                # Add padding to the bounding box
                padding = int(h * 0.1)
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(iw - x_pad, w + 2 * padding)
                h_pad = min(ih - y_pad, h + 2 * padding)
                
                if w_pad > 0 and h_pad > 0:
                    face_crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    if face_crop.size == 0:
                        continue
                        
                    face_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_img)
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        conf, predicted_idx = torch.max(probabilities, 1)
                        predicted_emotion = self.classes[predicted_idx.item()]
                        confidence = conf.item()
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if confidence > settings.EMOTION_CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), color, 2)
                    text = f"{predicted_emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x_pad, y_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                break # Process only the first detected face

        return frame, predicted_emotion, confidence, face_img
