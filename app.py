
import os
import subprocess
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Object Detection System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection System")
st.write("Upload an image or video to detect objects using YOLO.")


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

st.sidebar.header("Detection Settings")

weights = st.sidebar.text_input(
    "YOLO Model",
    value="yolov8n.pt"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 0.95, 0.25, 0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    0.10, 0.90, 0.45, 0.05
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# --------------------------------------------------
# Image detection
# --------------------------------------------------

def detect_image(uploaded_file, model):

    image = Image.open(uploaded_file).convert("RGB")

    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    result = model.predict(
        frame,
        conf=confidence,
        iou=iou_threshold,
        verbose=False
    )[0]

    # Draw YOLO boxes, labels and confidence scores
    detected_frame = result.plot()

    detected_frame = cv2.cvtColor(
        detected_frame,
        cv2.COLOR_BGR2RGB
    )

    return detected_frame, len(result.boxes)


# --------------------------------------------------
# Video detection
# --------------------------------------------------

def detect_video(uploaded_file, model):

    input_path = None
    avi_path = None
    output_path = None
    cap = None
    writer = None

    try:
        # Save uploaded video temporarily
        suffix = os.path.splitext(
            uploaded_file.name
        )[1] or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_input:
            temp_input.write(
                uploaded_file.getbuffer()
            )
            input_path = temp_input.name

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError(
                "Unable to open the uploaded video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Unable to read video dimensions."
            )

        # Temporary AVI for reliable frame writing
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi"
        ) as temp_avi:
            avi_path = temp_avi.name

        writer = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Unable to initialize video writer."
            )

        progress = st.progress(0)
        status = st.empty()

        frame_count = 0
        detection_count = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            result = model.predict(
                frame,
                conf=confidence,
                iou=iou_threshold,
                verbose=False
            )[0]

            # Render object boxes, labels and scores
            detected_frame = result.plot()

            detection_count += len(result.boxes)

            cv2.putText(
                detected_frame,
                f"Frame: {frame_count + 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            writer.write(detected_frame)
            frame_count += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_count / total_frames, 1.0)
                )

            status.write(
                f"Processing frame {frame_count}"
                + (
                    f" / {total_frames}"
                    if total_frames > 0
                    else ""
                )
            )

        cap.release()
        cap = None

        writer.release()
        writer = None

        if frame_count == 0:
            raise RuntimeError(
                "No readable frames found in the video."
            )

        progress.progress(1.0)
        status.success("Video detection completed.")

        # Convert AVI to browser-compatible H.264 MP4
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_mp4:
            output_path = temp_mp4.name

        command = [
            ffmpeg_exe,
            "-y",
            "-i", avi_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True
        )

        return output_path, frame_count, detection_count

    except Exception:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        for path in (input_path, avi_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# --------------------------------------------------
# Select input type
# --------------------------------------------------

input_type = st.radio(
    "Select Input Type",
    ["Image", "Video"],
    horizontal=True
)


# --------------------------------------------------
# Image upload UI
# --------------------------------------------------

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )

    if uploaded_file is not None:

        try:
            model = load_model(weights)

            with st.spinner("Detecting objects..."):
                detected_image, object_count = detect_image(
                    uploaded_file,
                    model
                )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(
                    uploaded_file,
                    use_column_width=True
                )

            with col2:
                st.subheader("Detected Image")
                st.image(
                    detected_image,
                    use_column_width=True
                )

            st.success(
                f"Detection completed. "
                f"Objects detected: {object_count}"
            )

            # Download detected image
            from io import BytesIO

            output_buffer = BytesIO()

            Image.fromarray(detected_image).save(
                output_buffer,
                format="JPEG"
            )

            st.download_button(
                "Download Detected Image",
                data=output_buffer.getvalue(),
                file_name="detected_image.jpg",
                mime="image/jpeg"
            )

        except Exception as error:
            st.error(f"Image detection failed: {error}")


# --------------------------------------------------
# Video upload UI
# --------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_upload"
    )

    if uploaded_file is not None:

        st.subheader("Original Video")
        st.video(uploaded_file)

        if st.button(
            "Start Video Detection",
            type="primary"
        ):

            output_path = None

            try:
                model = load_model(weights)

                with st.spinner("Processing video..."):
                    (
                        output_path,
                        frame_count,
                        detection_count
                    ) = detect_video(
                        uploaded_file,
                        model
                    )

                # Read the complete MP4 before cleanup
                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()

                st.subheader("Detected Video")
                st.video(video_bytes)

                col1, col2 = st.columns(2)

                col1.metric(
                    "Frames Processed",
                    frame_count
                )

                col2.metric(
                    "Detection Instances",
                    detection_count
                )

                st.download_button(
                    "Download Detected Video",
                    data=video_bytes,
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )

            except Exception as error:
                st.error(f"Video detection failed: {error}")

            finally:
                if (
                    output_path
                    and os.path.exists(output_path)
                ):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass