import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import numpy as np

model_path = 'blaze_face_short_range.tflite'
if not os.path.exists(model_path):
    url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_img)
detection_result = detector.detect(mp_image)
print(detection_result)
