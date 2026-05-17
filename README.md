# AI Fitness Trainer

AI-powered exercise posture correction system using OpenCV, MediaPipe, and Streamlit.

## Features
- Real-time posture correction
- Rep counting
- Form scoring
- Voice feedback
- Session analytics dashboard

## Supported Exercises
- Bicep Curls
- Squats
- Lunges
- Planks

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Streamlit
- Pandas
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Run Exercise Detection

```bash
python run_detection.py --exercise bicep
```

Other exercises:

```bash
python run_detection.py --exercise squat
python run_detection.py --exercise plank
python run_detection.py --exercise lunge
```

## Run Dashboard

```bash
streamlit run dashboard.py
```
