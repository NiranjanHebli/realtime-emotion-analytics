import sys
from pathlib import Path

# Add project root to sys.path using absolute resolution to guarantee it works on Streamlit Cloud
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import av
import time
import pandas as pd
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from src.core.inference import EmotionInferenceEngine
from src.config.logger import setup_logger
from src.config.settings import settings
from src.ui.components import apply_custom_css, render_header, render_sidebar

logger = setup_logger(__name__)

st.set_page_config(page_title="Real-Time Emotion Analytics", layout="wide")
apply_custom_css()

# Cache the model engine to avoid reloading
@st.cache_resource
def load_engine():
    return EmotionInferenceEngine()

engine = load_engine()

render_header()
render_sidebar(device_info=str(engine.device))

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.emotion_counts = Counter()
        self.total_frames = 0
        self.total_confidence = 0.0
        self.is_paused = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, emotion, confidence, face_img = engine.process_frame(img)

        # Only accumulate stats when analytics is not paused
        if not self.is_paused and emotion and emotion != "None":
            self.emotion_counts[emotion] += 1
            self.total_frames += 1
            self.total_confidence += confidence
            
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

col_video, col_analytics = st.columns([1, 1])

with col_video:
    st.subheader("Live Feed")
    ctx = webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"controls": False, "autoPlay": True, "muted": True},
    )

with col_analytics:
    st.subheader("Live Session Statistics")
    pause_toggle = st.toggle("Pause Analytics", value=False)
    dashboard_placeholder = st.empty()

while ctx.state.playing:
    if ctx.video_processor:
        # Sync the pause state from the UI toggle to the background thread
        ctx.video_processor.is_paused = pause_toggle

        counts = ctx.video_processor.emotion_counts
        total = ctx.video_processor.total_frames
        total_conf = ctx.video_processor.total_confidence
        
        with dashboard_placeholder.container():
            if pause_toggle:
                st.warning("Analytics paused. Toggle above to resume.")

            if sum(counts.values()) > 0:
                # Render Metrics horizontally
                m1, m2, m3 = st.columns(3)
                m1.metric("Frames Analyzed", total)
                
                dominant = counts.most_common(1)[0][0]
                m2.metric("Dominant Emotion", dominant.capitalize())
                
                avg_conf = (total_conf / total) * 100 if total > 0 else 0
                m3.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                st.write("")
                
                # Render Bar Chart with controlled height
                df = pd.DataFrame(list(counts.items()), columns=["Emotion", "Count"])
                st.bar_chart(df.set_index("Emotion"), height=280)
            else:
                st.info("Waiting for face detection to begin analytics...")
                    
    time.sleep(0.5)
