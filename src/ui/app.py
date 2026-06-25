import streamlit as st
import av
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from src.core.inference import EmotionInferenceEngine
from src.config.logger import setup_logger
from src.config.settings import settings
from src.ui.components import apply_custom_css, render_header, render_sidebar

logger = setup_logger(__name__)

st.set_page_config(page_title="Real-Time Emotion Analytics", layout="wide", page_icon="🎭")
apply_custom_css()

# Cache the model engine to avoid reloading
@st.cache_resource
def load_engine():
    return EmotionInferenceEngine()

engine = load_engine()

render_header()
render_sidebar(device_info=str(engine.device))

class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, emotion, confidence, face_img = engine.process_frame(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

st.subheader("Live Feed & Analysis")
ctx = webrtc_streamer(
    key="emotion-recognition",
    video_processor_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
