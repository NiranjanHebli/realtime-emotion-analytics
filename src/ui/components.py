import streamlit as st

def apply_custom_css():
    """Injects custom CSS for the Streamlit UI."""
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

def render_header():
    st.markdown("<div class='main-header'>Real-Time Emotion Analytics</div>", unsafe_allow_html=True)

def render_sidebar(device_info: str):
    st.sidebar.markdown("### System Status")
    st.sidebar.success("Model Loaded")
    st.sidebar.info(f"Device: {device_info}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **How it works:**
    1. Allows camera access.
    2. The CNN uses Spatial Attention to analyze facial expressions.
    """)
