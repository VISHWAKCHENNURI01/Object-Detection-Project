"""
YOLO Object Detection on Video

Examples:

Webcam:
    python src/detect_video.py --source 0

Video:
    python src/detect_video.py --source samples/test.mp4

Save:
    python src/detect_video.py --source samples/test.mp4 --save
"""

import argparse
import os
import subprocess
import tempfile
import time

import cv2
from ultralytics import YOLO


def convert_to_mp4(
    input_video,
    output_video
):
    """
    Convert video to browser-compatible
    H264 MP4 format.
    """

    try:

        import imageio_ffmpeg

        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        command = [
            ffmpeg_exe,

            "-y",

            "-i",
            input_video,

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-movflags",
            "+faststart",

            "-an",

            output_video
        ]

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        return True

    except Exception as error:

        print(
            "FFmpeg conversion failed:"
        )

        print(error)

        return False


def detect_video(
    source,
    weights="yolov8n.pt",
    conf=0.30,
    iou=0.45,
    save=False,
    output="outputs/video_out.mp4"
):

    # -------------------------------------------------
    # Source
    # -------------------------------------------------

    if isinstance(
        source,
        str
    ) and source.isdigit():

        source = int(source)

    cap = cv2.VideoCapture(
        source
    )

    if not cap.isOpened():

        raise FileNotFoundError(
            f"Could not open video: {source}"
        )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = YOLO(
        weights
    )

    # -------------------------------------------------
    # Video properties
    # -------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # -------------------------------------------------
    # Temporary AVI
    # -------------------------------------------------

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    temp_file.close()

    temp_path = temp_file.name

    writer = cv2.VideoWriter(
        temp_path,

        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        os.remove(
            temp_path
        )

        raise RuntimeError(
            "Could not create video writer."
        )

    frame_count = 0
    detection_count = 0

    previous_time = time.time()

    # -------------------------------------------------
    # Process video
    # -------------------------------------------------

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            # -----------------------------------------
            # YOLO prediction
            # -----------------------------------------

            results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                verbose=False
            )

            result = results[0]

            # -----------------------------------------
            # Draw detections
            # -----------------------------------------

            frame = result.plot()

            detection_count += (
                len(result.boxes)
            )

            # -----------------------------------------
            # FPS
            # -----------------------------------------

            current_time = time.time()

            fps_display = (
                1 /
                max(
                    current_time -
                    previous_time,
                    0.001
                )
            )

            previous_time = (
                current_time
            )

            cv2.putText(
                frame,

                f"FPS: "
                f"{fps_display:.1f}",

                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            # -----------------------------------------
            # Save processed frame
            # -----------------------------------------

            writer.write(
                frame
            )

            frame_count += 1

            # -----------------------------------------
            # Display
            # -----------------------------------------

            cv2.imshow(
                "YOLO Object Detection",
                frame
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"Detection instances: "
        f"{detection_count}"
    )

    # -------------------------------------------------
    # Save final MP4
    # -------------------------------------------------

    if save:

        output_dir = os.path.dirname(
            output
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        success = convert_to_mp4(
            temp_path,
            output
        )

        os.remove(
            temp_path
        )

        if not success:

            raise RuntimeError(
                "Could not create browser-compatible "
                "MP4 video."
            )

        print(
            f"Detected video saved to: "
            f"{output}"
        )

    else:

        os.remove(
            temp_path
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO Video Object Detection"
        )
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Video path or webcam index"
        )
    )

    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLO model weights"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detected video"
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/"
            "video_out.mp4"
        ),
        help="Output video path"
    )

    args = parser.parse_args()

    detect_video(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        output=args.out
    )


if __name__ == "__main__":
    main()