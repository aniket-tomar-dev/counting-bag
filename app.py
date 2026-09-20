"""
Cement Bag Counter — Streamlit Web Application MVP
Provides video upload, interactive counting zone setup (Rectangle Box or Line),
spatial conveyor belt detection zone filtering, real-time tracking visualization,
statistics summary, compact video player layout, and annotated video download.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CementBagCounter.App")

# Ensure project modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.detector import CementBagDetector
from src.counter import BagCounter
from src.video_processor import VideoProcessor

# Streamlit Page Config
st.set_page_config(
    page_title="Cement Bag Counter MVP",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for compact viewport & dark glassmorphism look
st.markdown("""
<style>
    /* Dark glassmorphism & compact viewport layout */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        margin-top: 2px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.0rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }

    /* Compact video player wrapper */
    .compact-video-container {
        max-width: 520px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    video, [data-testid="stVideo"] video {
        max-height: 300px !important;
        width: 100% !important;
        object-fit: contain !important;
        border-radius: 10px;
    }

    [data-testid="stImage"] img {
        max-height: 300px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain !important;
        border-radius: 10px;
        margin: 0 auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏗️ Cement Bag Counter — Video Upload MVP")
    st.markdown(
        "Upload a plant conveyor video, draw a **Counting Rectangle Box**, Polygon, or Line, and accurately count cement bags."
    )
    st.divider()

    # Sidebar: Configurations
    st.sidebar.header("⚙️ Configuration")

    # 1. Model Configuration
    st.sidebar.subheader("1. Detection Model")
    custom_model_uploaded = st.sidebar.file_uploader(
        "Upload Custom Model (.pt)", type=["pt"], help="Upload your custom cement_bag trained YOLO model"
    )
    
    model_path = config.MODEL_PATH
    if custom_model_uploaded:
        temp_model_path = os.path.join(config.MODELS_DIR, custom_model_uploaded.name)
        with open(temp_model_path, "wb") as f:
            f.write(custom_model_uploaded.getvalue())
        model_path = temp_model_path
        st.sidebar.success(f"Custom model loaded: {custom_model_uploaded.name}")
    else:
        model_path_input = st.sidebar.text_input("Model Path", value=config.MODEL_PATH)
        if model_path_input:
            model_path = model_path_input

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.20,
        step=0.05,
        help="Lower confidence (e.g. 0.15-0.25) helps detect bags on conveyor belts."
    )

    class_mode = st.sidebar.radio(
        "Target Object Filter",
        options=["All Objects (-1) [Recommended for Testing]", "Class 0 (Cement Bag / Person)", "Custom Class ID"],
        index=0,
        help="Select 'All Objects' when testing with pretrained YOLO models so any item on conveyor is detected."
    )

    if class_mode == "All Objects (-1) [Recommended for Testing]":
        target_class_id = -1
    elif class_mode == "Class 0 (Cement Bag / Person)":
        target_class_id = 0
    else:
        target_class_id = int(st.sidebar.number_input("Custom Class ID", min_value=0, max_value=80, value=0))

    # 2. Conveyor Belt Detection Zone (Crop Top Machinery)
    st.sidebar.subheader("2. Conveyor Belt Detection Zone")
    enable_zone = st.sidebar.checkbox(
        "Enable Conveyor Area Filter (Ignore Top Machinery)",
        value=True,
        help="Ignores objects detected at the top of the video (machinery, lights, chutes) so green box ONLY appears on conveyor belt!"
    )

    # 3. Optional ROI Configuration
    st.sidebar.subheader("3. Advanced / Debug")
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    # Initialize Detector
    try:
        detector = CementBagDetector(
            model_path=model_path,
            fallback_model=config.FALLBACK_MODEL_PATH,
            conf_threshold=conf_threshold,
            target_class_id=target_class_id
        )
    except Exception as e:
        st.error(f"Failed to initialize YOLO detector: {e}")
        return

    # Model status notice
    if not detector.is_custom_model:
        st.warning(
            "⚠️ **Running with standard YOLO fallback model.** "
            "Use **Conveyor Belt Zone Filter** in sidebar to ignore top machinery fixtures!"
        )
    else:
        st.success("✅ Loaded custom cement bag detection model: `models/best.pt`")

    # Step 1: Video Upload Section
    st.subheader("📹 1. Upload Plant Conveyor Video")
    uploaded_file = st.file_uploader(
        "Choose a video file (.mp4, .avi, .mov, .mkv)",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is None:
        st.info("Please upload a video file above to start processing.")
        return

    # Save uploaded video to uploads directory
    upload_filename = f"input_{int(time.time())}_{uploaded_file.name}"
    upload_path = os.path.join(config.UPLOAD_DIR, upload_filename)
    
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Validate uploaded video using OpenCV
    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        st.error("Unable to open video. Please upload a valid video file.")
        os.remove(upload_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    # Read frame 1 for preview
    ret, sample_frame = cap.read()
    cap.release()

    if total_frames <= 0 or not ret:
        st.error("Uploaded video contains no readable frames.")
        return

    video_signature = (uploaded_file.name, width, height, total_frames)
    if st.session_state.get("counting_box_video_signature") != video_signature:
        st.session_state["counting_box_video_signature"] = video_signature
        st.session_state["counting_box"] = (
            int(width * 0.15),
            int(height * 0.55),
            int(width * 0.85),
            int(height * 0.75),
        )
        st.session_state["counting_polygon"] = []

    # Dynamic Zone Sliders based on video height
    if enable_zone:
        y_min_val = st.sidebar.slider(
            "Min Y (Cutoff Top Machinery)",
            min_value=0,
            max_value=height,
            value=int(height * 0.40),
            step=10,
            help="Higher value cuts out top machinery (chute/pipes)."
        )
        y_max_val = st.sidebar.slider(
            "Max Y (Conveyor Bottom)",
            min_value=0,
            max_value=height,
            value=height,
            step=10
        )
        y_min = float(y_min_val)
        y_max = float(y_max_val)
    else:
        y_min = None
        y_max = None

    # Display Video Metadata Cards in compact grid
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Resolution</div><div class="metric-value" style="font-size:1.8rem">{width}x{height}</div></div>',
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">FPS</div><div class="metric-value" style="font-size:1.8rem">{fps:.1f}</div></div>',
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Frames</div><div class="metric-value" style="font-size:1.8rem">{total_frames}</div></div>',
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Duration</div><div class="metric-value" style="font-size:1.8rem">{duration:.1f}s</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 2: Interactive Counting Zone (Rectangle Box vs Virtual Line)
    st.subheader("📦 2. Draw Counting Zone (Rectangle Box or Line)")
    
    counting_mode_sel = st.radio(
        "Select Counting Method:",
        options=["Rectangle Box Zone [Recommended]", "Polygon Zone", "Virtual Line Crossing"],
        index=0,
        horizontal=True,
        help="Use a polygon zone when the conveyor area is not rectangular."
    )

    col_line_left, col_line_right = st.columns([1.1, 1.0])

    if "Rectangle" in counting_mode_sel:
        counter_mode = "RECTANGLE_BOX"
        line_start = (0, 0)
        line_end = (0, 0)
        count_direction = "ANY"
        roi_polygon = []

        with col_line_left:
            st.markdown("### 🔲 Draw Yellow Counting Rectangle Box")
            st.markdown(
                "👈 **Mouse Drag Tool**: Video frame ke upar mouse click & drag karke box banayein ya edges ko pakad kar resize karein!"
            )

            box_tool_mode = st.radio(
                "Select Box Creation Tool:",
                options=["🖱️ Interactive Mouse Canvas (Draw/Drag on Video)", "🎛️ Manual Sliders (Fine-tune Coordinates)"],
                index=0,
                horizontal=True
            )

            default_bx1 = int(width * 0.15)
            default_bx2 = int(width * 0.85)
            default_by1 = int(height * 0.55)
            default_by2 = int(height * 0.75)

            if "Interactive" in box_tool_mode:
                sample_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
                disp_w = 580
                disp_h = max(250, int(disp_w * (height / width))) if width > 0 else 350
                scale_x = width / disp_w
                scale_y = height / disp_h

                saved_bx1, saved_by1, saved_bx2, saved_by2 = st.session_state["counting_box"]
                disp_left = float(saved_bx1 / scale_x)
                disp_top = float(saved_by1 / scale_y)
                disp_width = float((saved_bx2 - saved_bx1) / scale_x)
                disp_height = float((saved_by2 - saved_by1) / scale_y)

                initial_drawing = {
                    "version": "4.4.0",
                    "objects": [
                        {
                            "type": "rect",
                            "version": "4.4.0",
                            "originX": "left",
                            "originY": "top",
                            "left": disp_left,
                            "top": disp_top,
                            "width": disp_width,
                            "height": disp_height,
                            "fill": "rgba(0, 255, 255, 0.25)",
                            "stroke": "#00FFFF",
                            "strokeWidth": 3,
                            "strokeDashArray": None,
                            "strokeLineCap": "butt",
                            "strokeDashOffset": 0,
                            "strokeLineJoin": "miter",
                            "strokeUniform": True,
                            "scaleX": 1,
                            "scaleY": 1,
                            "angle": 0,
                            "flipX": False,
                            "flipY": False,
                            "opacity": 1,
                            "visible": True,
                        }
                    ]
                }

                st.markdown(
                    "👉 **Instructions**: Frame ke upar ek **Fixed Rectangle Box pre-loaded** hai. "
                    "Aap uske **edges/corners ko pakad kar top-bottom, left-right drag & resize** kar sakte hain ya pure box ko kahi bhi move kar sakte hain!"
                )

                drawing_mode = "rect"

                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 255, 0.25)",
                    stroke_width=3,
                    stroke_color="#00FFFF",
                    background_image=sample_pil,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode=drawing_mode,
                    initial_drawing=initial_drawing,
                    key="interactive_rect_canvas_v2"
                )

                drawn_coords = None
                if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                    rect_objs = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "rect"]
                    if rect_objs:
                        last_rect = rect_objs[-1]
                        l = last_rect.get("left", 0)
                        t = last_rect.get("top", 0)
                        w = last_rect.get("width", 0) * last_rect.get("scaleX", 1.0)
                        h = last_rect.get("height", 0) * last_rect.get("scaleY", 1.0)

                        x1 = int(round(l * scale_x))
                        y1 = int(round(t * scale_y))
                        x2 = int(round((l + w) * scale_x))
                        y2 = int(round((t + h) * scale_y))

                        bx1, bx2 = min(x1, x2), max(x1, x2)
                        by1, by2 = min(y1, y2), max(y1, y2)

                        bx1 = max(0, min(width, bx1))
                        bx2 = max(0, min(width, bx2))
                        by1 = max(0, min(height, by1))
                        by2 = max(0, min(height, by2))

                        if (bx2 - bx1) >= 10 and (by2 - by1) >= 10:
                            drawn_coords = (bx1, by1, bx2, by2)

                if drawn_coords:
                    st.session_state["counting_box"] = drawn_coords
                    counting_box = st.session_state["counting_box"]
                    st.success(f"✅ **Adjusted Rectangle Box Active**: `(X1={counting_box[0]}, Y1={counting_box[1]}, X2={counting_box[2]}, Y2={counting_box[3]})`")
                else:
                    counting_box = st.session_state["counting_box"]
                    st.info(f"💡 Default Box Active: `(X1={counting_box[0]}, Y1={counting_box[1]}, X2={counting_box[2]}, Y2={counting_box[3]})`. Drag edges above to adjust.")
            else:
                saved_bx1, saved_by1, saved_bx2, saved_by2 = st.session_state["counting_box"]
                box_left_x = st.slider("Box Left X", min_value=0, max_value=width, value=saved_bx1, step=10)
                box_right_x = st.slider("Box Right X", min_value=0, max_value=width, value=saved_bx2, step=10)
                box_top_y = st.slider("Box Top Y", min_value=0, max_value=height, value=saved_by1, step=10)
                box_bottom_y = st.slider("Box Bottom Y", min_value=0, max_value=height, value=saved_by2, step=10)

                bx1 = min(box_left_x, box_right_x)
                bx2 = max(box_left_x, box_right_x)
                by1 = min(box_top_y, box_bottom_y)
                by2 = max(box_top_y, box_bottom_y)

                counting_box = (bx1, by1, bx2, by2)
                st.session_state["counting_box"] = counting_box

            test_det_button = st.button("🔍 Test Detections inside Counting Box")
    elif "Polygon" in counting_mode_sel:
        counter_mode = "POLYGON"
        counting_box = None
        line_start = (0, 0)
        line_end = (0, 0)
        count_direction = "ANY"

        with col_line_left:
            st.markdown("### 🔷 Draw Polygon Counting Zone")
            st.markdown(
                "👉 Corners par click karein. **Polygon save/activate karne ke liye first white vertex par dobara click karke shape close karein.** "
                "Jab tak white vertex handles dikh rahe hain, polygon pending hai."
            )

            sample_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
            disp_w = 580
            disp_h = max(250, int(disp_w * (height / width))) if width > 0 else 350
            scale_x = width / disp_w
            scale_y = height / disp_h
            saved_polygon = st.session_state.get("counting_polygon", [])
            initial_objects = []
            if len(saved_polygon) >= 3:
                initial_objects = [{
                    "type": "polygon",
                    "version": "4.4.0",
                    "originX": "left",
                    "originY": "top",
                    "left": 0,
                    "top": 0,
                    "pathOffset": {"x": 0, "y": 0},
                    "points": [
                        {"x": float(x / scale_x), "y": float(y / scale_y)}
                        for x, y in saved_polygon
                    ],
                    "fill": "rgba(0, 255, 255, 0.20)",
                    "stroke": "#00FFFF",
                    "strokeWidth": 3,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "opacity": 1,
                }]

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 255, 0.20)",
                stroke_width=3,
                stroke_color="#00FFFF",
                background_image=sample_pil,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="polygon",
                initial_drawing={"version": "4.4.0", "objects": initial_objects},
                key="interactive_polygon_canvas_v1"
            )

            polygon_coords = None
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                polygon_objs = [
                    obj for obj in canvas_result.json_data["objects"]
                    if str(obj.get("type", "")).lower() == "polygon"
                ]
                if polygon_objs:
                    polygon = polygon_objs[-1]
                    left = float(polygon.get("left", 0))
                    top = float(polygon.get("top", 0))
                    object_width = float(polygon.get("width", 0))
                    object_height = float(polygon.get("height", 0))
                    scale_polygon_x = float(polygon.get("scaleX", 1.0))
                    scale_polygon_y = float(polygon.get("scaleY", 1.0))
                    path_offset = polygon.get("pathOffset", {})
                    offset_x = float(path_offset.get("x", 0))
                    offset_y = float(path_offset.get("y", 0))
                    points = polygon.get("points", [])
                    raw_points = []
                    for point in points:
                        if isinstance(point, dict):
                            raw_x = float(point.get("x", 0))
                            raw_y = float(point.get("y", 0))
                        else:
                            raw_x = float(point[0])
                            raw_y = float(point[1])
                        raw_points.append((raw_x, raw_y))

                    def to_video_points(origin_x, origin_y, point_scale_x=1.0, point_scale_y=1.0):
                        return [
                            (
                                int(round((origin_x + raw_x * point_scale_x) * scale_x)),
                                int(round((origin_y + raw_y * point_scale_y) * scale_y)),
                            )
                            for raw_x, raw_y in raw_points
                        ]

                    # Fabric polygon coordinates can be serialized either
                    # relative to pathOffset or relative to the object origin.
                    candidates = [
                        to_video_points(
                            left + object_width / 2 - offset_x * scale_polygon_x,
                            top + object_height / 2 - offset_y * scale_polygon_y,
                            scale_polygon_x,
                            scale_polygon_y,
                        ),
                        to_video_points(left, top, scale_polygon_x, scale_polygon_y),
                        to_video_points(0, 0),
                    ]
                    candidates.sort(
                        key=lambda candidate: sum(
                            0 <= x <= width and 0 <= y <= height
                            for x, y in candidate
                        ),
                        reverse=True,
                    )
                    if candidates and len(candidates[0]) >= 3:
                        in_bounds = [
                            (x, y) for x, y in candidates[0]
                            if 0 <= x <= width and 0 <= y <= height
                        ]
                        if len(in_bounds) >= 3:
                            polygon_coords = [
                                (max(0, min(width, x)), max(0, min(height, y)))
                                for x, y in candidates[0]
                            ]

            if polygon_coords:
                st.session_state["counting_polygon"] = polygon_coords
            roi_polygon = st.session_state.get("counting_polygon", [])
            if len(roi_polygon) >= 3:
                st.success(f"✅ Polygon counting zone active ({len(roi_polygon)} points)")
            else:
                st.warning("⚠️ Polygon abhi pending hai. First white vertex par dobara click karke close karein; uske baad preview me yellow polygon dikhega.")

            test_det_button = st.button("🔍 Test Detections inside Polygon")
    else:
        counter_mode = "LINE"
        counting_box = None
        roi_polygon = []
        with col_line_left:
            st.markdown("**📏 Align Yellow Counting Line**")
            count_direction = st.selectbox(
                "Counting Direction",
                options=["ANY (Count All Crossings)", "LEFT_TO_RIGHT", "RIGHT_TO_LEFT", "TOP_TO_BOTTOM", "BOTTOM_TO_TOP"],
                index=0
            )
            if "ANY" in count_direction:
                count_direction = "ANY"

            line_start_x = st.slider("Start X", min_value=0, max_value=width, value=int(width * 0.10), step=10)
            line_start_y = st.slider("Start Y", min_value=0, max_value=height, value=int(height * 0.70), step=10)
            line_end_x = st.slider("End X", min_value=0, max_value=width, value=int(width * 0.90), step=10)
            line_end_y = st.slider("End Y", min_value=0, max_value=height, value=int(height * 0.70), step=10)

            line_start = (int(line_start_x), int(line_start_y))
            line_end = (int(line_end_x), int(line_end_y))

            test_det_button = st.button("🔍 Test Detections on Sample Frame")

    # Render Preview Frame with Overlays
    preview_img = sample_frame.copy()
    
    # 1. Draw Conveyor Zone Boundary if enabled (Dashed Blue Cutoff)
    if enable_zone and y_min is not None:
        cv2.line(preview_img, (0, int(y_min)), (width, int(y_min)), (255, 100, 0), 2)
        cv2.putText(
            preview_img,
            f"CONVEYOR ZONE BOUNDARY (Y >= {int(y_min)})",
            (15, max(25, int(y_min) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 0),
            1,
            cv2.LINE_AA
        )

    # 2. Draw Counting Element
    if counter_mode == "RECTANGLE_BOX" and counting_box:
        bx1, by1, bx2, by2 = counting_box
        cv2.rectangle(preview_img, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
        cv2.putText(
            preview_img,
            "COUNTING BOX ZONE",
            (bx1 + 10, max(by1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
    elif counter_mode == "POLYGON" and len(roi_polygon) >= 3:
        polygon_points = np.array(roi_polygon, dtype=np.int32)
        cv2.polylines(preview_img, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=3)
        label_x, label_y = polygon_points[0]
        cv2.putText(
            preview_img,
            "POLYGON COUNTING ZONE",
            (int(label_x), max(int(label_y) - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
    elif counter_mode == "LINE":
        cv2.line(preview_img, line_start, line_end, (0, 255, 255), 4)
        mid_x = int((line_start[0] + line_end[0]) / 2)
        mid_y = int((line_start[1] + line_end[1]) / 2)
        cv2.putText(
            preview_img,
            f"COUNTING LINE ({count_direction})",
            (max(10, mid_x - 120), max(25, mid_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Test detection if button clicked
    if test_det_button:
        detections = detector.detect(
            sample_frame,
            conf_threshold=conf_threshold,
            target_class_id=target_class_id,
            y_min=y_min,
            y_max=y_max
        )
        st.write(f"**Found {len(detections)} detection(s) inside Conveyor Zone**")
        for bbox, conf_val, c_id in detections:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                preview_img,
                f"Bag ({conf_val:.2f})",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    with col_line_right:
        st.markdown("**Sample Frame Preview**")
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, use_container_width=True)

    st.divider()

    # Step 3: Start Processing Section
    st.subheader("⚡ 3. Start Processing & Counting")
    
    if st.button("🚀 Start Full Video Processing"):
        output_filename = f"processed_{int(time.time())}.mp4"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)

        counter = BagCounter(
            mode=counter_mode,
            counting_box=counting_box,
            line_start=line_start,
            line_end=line_end,
            direction=count_direction,
            roi_polygon=roi_polygon
        )

        processor = VideoProcessor(
            detector=detector,
            counter=counter,
            tracker_type=config.TRACKER_TYPE,
            debug_mode=debug_mode
        )

        # Progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Compact live preview container
        col_prev_l, col_prev_c, col_prev_r = st.columns([1.5, 3, 1.5])
        with col_prev_c:
            preview_container = st.empty()

        def progress_callback(
            curr_frame: int,
            tot_frames: int,
            proc_fps: float,
            current_count: int,
            frame_img: np.ndarray
        ) -> bool:
            pct = min(int((curr_frame / tot_frames) * 100), 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f"**Processing Video:** {pct}% | "
                f"Frame: **{curr_frame} / {tot_frames}** | "
                f"Processing FPS: **{proc_fps:.1f}** | "
                f"Current Count: **{current_count}**"
            )
            # Render frame preview periodically (every 5 frames)
            if curr_frame % 5 == 0 or curr_frame == tot_frames:
                frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                preview_container.image(frame_rgb, channels="RGB", use_container_width=True)
            return True

        with st.spinner("Processing video frame-by-frame..."):
            try:
                metrics = processor.process_video(
                    input_path=upload_path,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    target_class_id=target_class_id,
                    y_min=y_min,
                    y_max=y_max,
                    progress_callback=progress_callback
                )
                st.session_state["last_metrics"] = metrics
                st.success("🎉 Video Processing Completed!")
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
                logger.error(f"Processing error: {e}", exc_info=True)
                return

    # Display Results in Compact Viewport Layout
    if "last_metrics" in st.session_state:
        metrics = st.session_state["last_metrics"]
        out_path = metrics["output_path"]

        st.divider()
        st.subheader("📊 4. Processing Results & Compact Video View")

        res_col_stats, res_col_video = st.columns([1.1, 1.9])
        
        with res_col_stats:
            st.markdown(
                f'''
                <div class="metric-card" style="border: 2px solid #10b981;">
                    <div class="metric-label">TOTAL BAGS COUNTED</div>
                    <div class="metric-value" style="font-size: 3.5rem;">{metrics["total_bags"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"**Video FPS:** `{metrics['video_fps']:.1f}`")
            st.markdown(f"**Processing FPS:** `{metrics['processing_fps']:.1f}`")
            st.markdown(f"**Total Frames Processed:** `{metrics['processed_frames']}`")
            st.markdown(f"**Video Duration:** `{metrics['video_duration_sec']:.1f}s`")
            st.markdown(f"**Processing Duration:** `{metrics['processing_duration_sec']:.1f}s`")

            st.markdown("<br>", unsafe_allow_html=True)

            # Download Processed Video Button
            if os.path.exists(out_path):
                with open(out_path, "rb") as vf:
                    video_bytes = vf.read()
                st.download_button(
                    label="💾 Download Processed Video (.mp4)",
                    data=video_bytes,
                    file_name=os.path.basename(out_path),
                    mime="video/mp4"
                )

        with res_col_video:
            st.markdown("**Compact Output Video Player**")
            if os.path.exists(out_path):
                # Wrapped in compact container styling
                st.markdown('<div class="compact-video-container">', unsafe_allow_html=True)
                st.video(out_path)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
