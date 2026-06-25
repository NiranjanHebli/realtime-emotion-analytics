# Real-Time Emotion Analytics

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00B2FF.svg?style=flat&logo=MediaPipe&logoColor=white)

**Real-Time Emotion Analytics** is a next-generation emotion analytics system that leverages MediaPipe 3D Face Mesh and a Spatial Attention-based Convolutional Neural Network (CNN) to provide real-time facial expression recognition and continuous emotion journey tracking.

## Unique Features

- **Spatial Attention CNN**: Focuses deep learning specifically on highly expressive regions (eyes, mouth) to reduce background noise.
- **MediaPipe Face Tracking**: Highly accurate 3D face mesh for robust facial landmark detection.
- **Emotion Journey Timeline**: Real-time analytics dashboard mapping emotional fluctuations throughout a session.
- **Active Learning Loop**: Interactive UI that allows the user to correct low-confidence predictions to continuously fine-tune the model.

## Real-Life Use Cases

- **Customer Feedback & Retail Analytics**: Monitor customer reactions in real-time at retail kiosks, checkout lines, or while testing new products to gauge satisfaction and emotional engagement.
- **Telehealth & Remote Therapy**: Assist therapists during virtual sessions by providing objective, continuous tracking of a patient's emotional state over time.
- **E-Learning & Student Engagement**: Analyze student focus and frustration levels during online classes to help educators adjust their pacing or intervene when students are struggling.
- **Driver Monitoring Systems**: Detect signs of road rage, fatigue, or stress in drivers to trigger safety alerts and prevent accidents.

## Installation and Setup

We use `uv` for fast dependency management. The provided setup script will automatically create a virtual environment, install all dependencies, and download the dataset.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NiranjanHebli/realtime-emotion-analytics.git
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

### Manual Setup (Alternative)

If you prefer not to use the automated `setup.sh` script or `uv`, you can manually set up the project using standard Python tools:

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset manually:**
   ```bash
   python scripts/setup_dataset.py
   ```
## Data Preparation

The `setup.sh` script automatically downloads and organizes the required Kaggle dataset (4 emotion classes: angry, happy, sad, neutral) using `kagglehub`. The images are placed into `data/train` and `data/test` folders so they are ready for PyTorch.

## Running the Application

To launch the real-time emotion analytics dashboard, run the following command from the project root:

```bash
python -m streamlit run src/ui/app.py
```

Ensure your virtual environment is active before running this command.

## Future Scope

- **Temporal Sequence Modeling**: Integrate recurrent networks (LSTMs) or Transformers to capture facial expression dynamics over time.
- **Multi-Face Analytics**: Extend the system to detect and log concurrent session analytics for multiple people.
- **Session Exporting**: Support exporting the live session metrics and distribution charts to PDF or CSV reports.
- **Active Learning Integration**: Enable saving low-confidence classification frames to trigger model retraining.

