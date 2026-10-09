# Object-Detection-Project

🎯 YOLO-Based Image and Video Object Detection

## 📌 Project Overview

This project is a real-time object detection application built using **YOLO, OpenCV, and Streamlit**. It detects and identifies objects in uploaded images and videos by drawing bounding boxes, displaying class labels, and showing confidence scores.

The application provides a simple, interactive web interface that allows users to upload images or videos, visualize detection results, and download the processed output.

## ✨ Features

* **Image Detection:** Upload images in JPG, JPEG, PNG, and WEBP formats.

* **Video Detection:** Upload videos in MP4, AVI, MOV, MKV, and WEBM formats.

* **YOLO Object Detection:** Identify multiple objects in images and video frames.

* **Bounding Boxes:** Display detected objects with class labels and confidence scores.

* **Frame-by-Frame Processing:** Analyze video frames to detect objects throughout the video.

* **Video Playback:** Preview the processed video in the application.

* **Download Results:** Download detected images and processed videos.

* **Adjustable Parameters:** Configure confidence and IoU thresholds.

* **Interactive Interface:** Use the application through a Streamlit web interface.

## 🛠️ Tech Stack

* **Programming Language:** Python

* **Object Detection:** YOLO (Ultralytics)

* **Computer Vision:** OpenCV

* **Web Framework:** Streamlit

* **Image Processing:** Pillow, NumPy

* **Video Conversion:** FFmpeg via `imageio-ffmpeg`

## 📂 Project Structure

```
object_detection_project/
│
├── app.py
├── requirements.txt
│
└── src/
    ├── detect_image.py
    ├── detect_video.py
    └── utils.py
```

## ⚙️ Installation

**1. Clone the repository**

```
git clone <your-repository-url>
cd object_detection_project
```

**2. Create a virtual environment (optional)**

```
python -m venv venv
```

Activate it on Windows:

```
venv\Scripts\activate
```

**3. Install dependencies**

```
pip install -r requirements.txt
```

Ensure the requirements include:

```
streamlit
ultralytics
opencv-python
numpy
pillow
imageio-ffmpeg
```

## ▶️ Run the Application

Start the Streamlit application:

```
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚀 How to Use

1. Launch the application.

2. Select **Image** or **Video** as the input type.

3. Upload your image or video.

4. Adjust the confidence and IoU thresholds if needed.

5. Start detection for video inputs.

6. View the detected objects and their bounding boxes.

7. Download the processed image or video.

## 🧠 How It Works

1. The user uploads an image or video through Streamlit.

2. YOLO analyzes the image or processes each video frame.

3. The model predicts object classes, bounding boxes, and confidence scores.

4. Detection results are rendered onto the images or video frames.

5. Processed video is converted into a browser-compatible MP4 format.

6. The application displays the results and provides download options.

## 🎯 Applications

* Traffic and vehicle monitoring

* Pedestrian detection

* Surveillance and security

* Object recognition in everyday scenes

* Computer vision demonstrations

* Educational and research projects

## 🔮 Future Enhancements

* Live webcam object detection through the web interface

* Object tracking across video frames

* Detection statistics and visual analytics

* Support for custom-trained YOLO models

* Deployment to a cloud platform
