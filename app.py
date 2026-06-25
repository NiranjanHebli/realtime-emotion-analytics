import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
import os
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from inference import EmotionRecognizer

# Set up page config
st.set_page_config(page_title="Real-Time Emotion Analytics", layout="wide", page_icon="🎭")

# Initialize the model (cache it to avoid reloading)
@st.cache_resource
def load_model():
    return EmotionRecognizer()

recognizer = load_model()

# Setup Custom Dataset dir for Active Learning
CUSTOM_DATASET_DIR = "custom_dataset"
os.makedirs(CUSTOM_DATASET_DIR, exist_ok=True)
for c in recognizer.classes:
    os.makedirs(os.path.join(CUSTOM_DATASET_DIR, c), exist_ok=True)

# CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E2E;
        color: #FFFFFF;
    }
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #89B4FA;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #313244;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Real-Time Emotion Analytics</div>", unsafe_allow_html=True)



# Define the WebRTC Video Transformer
class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        processed_img, emotion, confidence, face_img = recognizer.process_frame(img)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

st.subheader("Live Feed & Analysis")
ctx = webrtc_streamer(
    key="emotion-recognition",
    video_processor_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

st.sidebar.markdown("### System Status")
st.sidebar.success("Model Loaded")
st.sidebar.info("Device: " + str(recognizer.device))
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Allows camera access.
2. The CNN uses Spatial Attention to analyze facial expressions.
""")
