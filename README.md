# Cement Bag Counter — Video Upload MVP

A local web application built with Streamlit, OpenCV, Ultralytics YOLO, and ByteTrack to count cement bags on a roller/conveyor belt from uploaded video files.

---

## 🌟 Key Features

1. **Video File Upload & Validation**: Accepts `.mp4`, `.avi`, `.mov`, `.mkv` files with instant format validation.
2. **YOLO Object Detection**: Configurable model path (default `models/best.pt` with fallback to `yolov8n.pt` for pipeline testing).
3. **Multi-Object Tracking (ByteTrack)**: Assigns persistent tracking IDs to individual cement bags across frames.
4. **Virtual Line Crossing & Directional Counting**: Counts bags crossing a user-defined virtual line in `LEFT_TO_RIGHT`, `RIGHT_TO_LEFT`, `TOP_TO_BOTTOM`, or `BOTTOM_TO_TOP` directions.
5. **Guaranteed Double-Counting Protection**: Tracks counted IDs in `counted_track_ids` set to ensure every bag is counted **exactly once**.
6. **Optional Region of Interest (ROI)**: Restricts detection/counting to specific polygon region.
7. **Annotated Output Video**: Renders bounding boxes, tracking IDs, confidence scores, counting line visual indicator, and live count HUD onto the video.
8. **Video Download**: One-click download of processed annotated MP4 video.
9. **Synthetic Test Suite**: Standalone unit tests verifying line-crossing mathematical logic independently of YOLO models.

---

## 🏗️ Architecture

```text
                USER
                  |
                  v
          Upload Video (.mp4)
                  |
                  v
          Streamlit App (app.py)
                  |
        +---------+---------+
        |                   |
        v                   v
   YOLO Detection      Video Processing
        |                   |
        +---------+---------+
                  |
                  v
       ByteTrack Tracker
                  |
                  v
          Line Crossing Logic
                  |
                  v
             Count = N
                  |
        +---------+---------+
        |                   |
        v                   v
   Total Count        Annotated Video
                            |
                            v
                       Download (.mp4)
```

---

## 🛠️ Installation & Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 🧪 Running Unit Tests

Run the isolated synthetic counting algorithm test suite:
```bash
python -m pytest tests/test_counter.py
```

Expected output:
```text
tests/test_counter.py ...... [100%]
6 passed in 0.62s
```

---

## 🚀 Running the Streamlit Application

Start the web application:
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎯 Model Configuration & Custom Model

By default, the application checks for a custom trained model at:
`models/best.pt`

If no custom model is found at that path, it falls back to `yolov8n.pt` for pipeline verification and alerts:
> ⚠️ *Detection model is not suitable for cement bags. A custom cement-bag detection model is required at models/best.pt.*

To use your custom cement bag model:
1. Place your trained `.pt` weights file at `models/best.pt`
2. OR upload your `.pt` file directly via the **Sidebar Configuration** panel in the app.

---

## 📂 Project Structure

```text
cement-bag-counter/
│
├── app.py                  # Streamlit UI Application
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
│
├── models/
│   └── best.pt             # Custom YOLO model location
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Application settings & parameters
│   ├── detector.py         # Ultralytics YOLO wrapper
│   ├── tracker.py          # ByteTrack multi-object tracker
│   ├── counter.py          # Line crossing & counting algorithm
│   └── video_processor.py  # Video processing & frame annotation engine
│
├── uploads/                # Directory for uploaded input videos
├── outputs/                # Directory for generated annotated videos
└── tests/
    └── test_counter.py     # Standalone unit test suite
```
