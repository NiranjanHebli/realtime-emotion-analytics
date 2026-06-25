import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import EmotionCNN_Attention
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from logger_config import setup_logger

logger = setup_logger(__name__)

class EmotionRecognizer:
    def __init__(self, model_path='model.pth', num_classes=4):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionCNN_Attention(num_classes=num_classes).to(self.device)
        
        # Load weights if available
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model weights from {model_path} onto {self.device}")
        else:
            logger.warning(f"{model_path} not found. Using untrained weights.")
            
        # The 4 classes based on our setup_dataset.py logic
        self.classes = ['angry', 'happy', 'neutral', 'sad']
        
        # Transformation for input faces
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Download MediaPipe Face Detection model if not present
        self.mp_model_path = 'blaze_face_short_range.tflite'
        if not os.path.exists(self.mp_model_path):
            url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
            logger.info("Downloading MediaPipe Face Detection model...")
            urllib.request.urlretrieve(url, self.mp_model_path)

        # Initialize MediaPipe Face Detection using the new Tasks API
        base_options = python.BaseOptions(model_asset_path=self.mp_model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
        self.face_detector = vision.FaceDetector.create_from_options(options)

    def process_frame(self, frame):
        """
        Processes a BGR frame, detects face, and predicts emotion.
        Returns: modified frame, emotion label, confidence, face_img (if detected)
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run face detection
        detection_result = self.face_detector.detect(mp_image)
        
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
                
                # Crop face
                if w_pad > 0 and h_pad > 0:
                    face_crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    if face_crop.size == 0:
                        continue
                        
                    # Save a copy of the raw face for active learning
                    face_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image for PyTorch transforms
                    pil_img = Image.fromarray(face_img)
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    
                    # Predict emotion
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        conf, predicted_idx = torch.max(probabilities, 1)
                        predicted_emotion = self.classes[predicted_idx.item()]
                        confidence = conf.item()
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255) # Orange if low confidence
                    cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), color, 2)
                    text = f"{predicted_emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x_pad, y_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Just take the first detected face for simplicity in this demo
                break 

        return frame, predicted_emotion, confidence, face_img

if __name__ == "__main__":
    logger.info("Testing Inference module...")
    recognizer = EmotionRecognizer()
    # Create a dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    processed_img, emotion, conf, _ = recognizer.process_frame(dummy_img)
    logger.info(f"Processed empty image. Output emotion: {emotion}")
