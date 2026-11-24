# ACT — Computer Vision-Powered Assistant

ACT is a computer vision-powered personal assistant that interprets a user's physical activities and uses those observations to provide context-aware assistance. Instead of relying entirely on manually entered prompts, ACT uses visual information to identify objects, estimate human poses, recognize actions, and predict higher-level activities.

The system combines object detection, pose estimation, action recognition, machine learning, and a language model to interpret the user's environment and generate relevant responses. It is designed to explore how an assistant can understand what a person is doing and provide assistance based on observed context.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Activity Recognition](#activity-recognition)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Local Installation](#local-installation)
- [Model Setup](#model-setup)
- [Running ACT Locally](#running-act-locally)
- [Camera and Video Input](#camera-and-video-input)
- [Language Model Setup](#language-model-setup)
- [Smart Home Integration](#smart-home-integration)
- [Monitoring](#monitoring)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Project Overview

Traditional conversational assistants generally depend on users explicitly describing what they need. ACT explores an alternative approach: using computer vision to recognize a user's current activities and provide assistance based on the surrounding context.

For example, ACT can process visual information to identify a person interacting with a laptop, sitting at a desk, or handling objects such as a backpack. These lower-level observations are used to predict a broader activity, such as studying or preparing to leave.

The project focuses on three primary activity classes:

- **Studying**
- **Going to school**
- **Going to bed**

The system can also recognize lower-level interactions, including sitting, reaching for objects, picking up objects, holding objects, and interacting with a laptop.

### Core objectives

1. Use computer vision to understand a user's physical environment.
2. Identify relevant objects and estimate human body landmarks.
3. Recognize actions using visual and geometric features.
4. Predict higher-level activities using a machine learning classifier.
5. Use a language model to generate context-aware responses.
6. Explore voice-based assistance and smart-home integrations.

## System Architecture

ACT uses a multi-stage processing pipeline.

```text
                Webcam / Video / RTSP
                         |
                         v
                  OpenCV Input
                         |
              +----------+----------+
              |                     |
              v                     v
         YOLOv8 Detection      MediaPipe Pose
         Object Detection     Landmark Estimation
              |                     |
              +----------+----------+
                         |
                         v
                  Action Recognition
              Geometric Feature Extraction
                         |
                         v
                Random Forest Classifier
                  Activity Prediction
                         |
                         |
                         v
                Smart-Home Actions
                (Kasa Integration)
```

### 1. Video acquisition

OpenCV captures frames from a local webcam, video file, or supported RTSP stream.

### 2. Object detection

YOLOv8 identifies relevant objects in the camera feed, including:

- Backpacks
- Handbags
- Suitcases
- Bottles and cups
- Laptops
- Cell phones
- Books
- Chairs and couches

Object detections provide contextual information about the user's surroundings and interactions.

### 3. Pose estimation

MediaPipe Pose Landmarker estimates human body landmarks, including wrist, shoulder, hip, knee, and ankle positions.

ACT uses these landmarks to derive geometric relationships and identify interactions between the person and detected objects.

### 4. Action recognition

The action-recognition component combines object detections and pose landmarks to identify lower-level actions, such as:

- Sitting
- Reaching for an object
- Picking up an object
- Holding an object
- Putting an object down
- Touching an object
- Drinking from a cup or bottle
- Using a laptop

### 5. Activity prediction

A trained Random Forest classifier processes extracted activity features and predicts a higher-level activity.

The current activity categories include studying, going to school, and going to bed.

### 6. Language model and assistance (No longer in plan)

ACT uses Qwen through Ollama to interpret activity context and generate natural-language responses. The language model is intended to run locally rather than requiring a hosted LLM API.

### 7. Voice and smart-home integration

The system includes voice-output functionality and an integration with Kasa smart-home devices. These components allow ACT to explore context-aware assistance beyond displaying computer vision predictions.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Video processing | OpenCV |
| Object detection | Ultralytics YOLOv8 |
| Pose estimation | MediaPipe |
| Activity classification | Scikit-learn Random Forest |
| Smart-home integration | Kasa |
| Metrics | Prometheus |
| Dashboard | Grafana |
| Containerization | Docker |
| Model hosting | Hugging Face |

## Activity Recognition

ACT separates low-level visual observations from higher-level activity predictions.

For example, detecting a laptop alone does not necessarily mean that someone is studying. ACT combines multiple signals, including object detections, pose information, and interaction patterns, to make a broader prediction.

The Random Forest classifier was trained using a structured activity dataset containing multiple activity categories.

The project also evaluates different model configurations to understand the tradeoff between computational cost and recognition quality.

### Selected computer vision configuration

The current configuration selected for further development is:

- **Object detector:** YOLOv8n
- **Pose estimator:** MediaPipe Heavy

YOLOv8n achieved the strongest corrected per-second multi-label F1 score in the evaluated comparison:

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| YOLOv8n | 0.69 | 0.41 | 0.51 |
| YOLOv8s | 0.56 | 0.41 | 0.47 |
| YOLOv8l | 0.56 | 0.41 | 0.47 |

MediaPipe Heavy was visually observed to provide more accurate wrist localization during movement and partial occlusion than MediaPipe Full, at the cost of slower processing.

These results come from separate detector and pose-model evaluations. They should not be interpreted as a jointly benchmarked end-to-end accuracy result.

---

## Project Structure

The following is the intended organization of the project. Some supporting files may differ depending on the current repository version.

```text
ACT/
│
├── src/
│   ├── ACT.py
│   ├── event_predictor.py
│   └── ...
│
├── artifacts/
│   ├── event_model.joblib
│   └── event_model_meta.json
│
├── YOLO/
│   └── yolov8s.pt
│
├── POSES/
│   └── pose_landmarker_heavy.task
│
├── resources/
│   └── dataset.csv
│
├── output/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── prometheus.yml
├── .dockerignore
├── .gitignore
└── README.md
```

**Important:** The selected detector configuration is YOLOv8n. If the repository currently uses `yolov8s.pt` as its default weight file, update the configuration and model asset before describing YOLOv8n as the active runtime detector.

Large model artifacts may be downloaded separately rather than committed to GitHub.

---

## Requirements

### Hardware

Recommended:

- Windows 10 or Windows 11
- A webcam or accessible video stream
- At least 8 GB RAM
- A modern multi-core CPU
- NVIDIA GPU for accelerated inference, if available

ACT can use CPU inference, although processing speed may be lower.

### Software

Install the following:

- Python 3.10 or 3.11
- Git
- FFmpeg (for RTSP streaming)
- Ollama (for local language model inference)

Docker Desktop is optional.

---

## Local Installation

The following instructions use Windows with Git Bash.

### 1. Clone the repository

```bash
git clone https://github.com/danielkoo-ufl/ACT.git

cd ACT
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/Scripts/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

If the repository uses a separate Docker dependency file, use `requirements-docker.txt` only for the container environment unless it is also intended for native Windows execution.

If a dependency fails to install, check the Python version and the package's platform compatibility before changing versions.

---

## Model Setup

ACT requires trained activity-classification weights and computer vision model assets.

### 1. Download the trained activity model

The trained Random Forest model is hosted separately on Hugging Face.

Repository:

[DanielK00/act-models](https://huggingface.co/DanielK00/act-models)

Install the Hugging Face client:

```bash
pip install huggingface_hub joblib
```

The activity model is stored under the `artifacts/` directory in the Hugging Face repository.

Download it using:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path

ROOT = Path(__file__).resolve().parent

model_path = hf_hub_download(
    repo_id="DanielK00/act-models",
    filename="artifacts/event_model.joblib",
    repo_type="model",
    local_dir=ROOT
)

print("Model downloaded:", model_path)
```

If the metadata file is available in the model repository, download it as well:

```python
metadata_path = hf_hub_download(
    repo_id="DanielK00/act-models",
    filename="artifacts/event_model_meta.json",
    repo_type="model",
    local_dir=ROOT
)
```

Run this code from a script located in the project root, or adjust `ROOT` to point to the actual project directory.

Only load model files from sources you trust because Joblib files use Python's pickle-based serialization.

### 2. Download YOLO weights

Ultralytics can download model weights automatically when a model name is provided:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

For the selected configuration, place the downloaded weights in the `YOLO/` directory and ensure the ACT configuration references the correct file.

### 3. Download MediaPipe Pose Landmarker

Create the pose-model directory:

```bash
mkdir -p POSES
```

Download the Heavy model from the official MediaPipe model storage:

```bash
curl -L \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" \
  -o POSES/pose_landmarker_heavy.task
```

Make sure the model path used by ACT matches the downloaded filename.

---

## Running ACT Locally

After installing dependencies and downloading the required model files, run the main application from the repository root.

Activate the virtual environment:

```bash
source .venv/Scripts/activate
```

Start ACT:

```bash
python src/ACT.py
```

The application initializes its video input, computer vision models, and activity prediction components.

Depending on the selected input mode, ACT can process a video file, a local webcam, or an RTSP stream.

**Input configuration:** The project supports video-file and webcam modes. Set the input mode and camera or video source using the configuration already defined in `src/ACT.py`. For a local webcam, use camera index `0` unless your computer exposes the camera under another index.

For example, the conceptual input settings are:

```python
input_options = 1  # Webcam mode
camera_index = 0
```

Use the actual variable names and configuration structure in the current source file if they differ.

---

## Camera and Video Input

### Webcam

For a directly connected webcam, select webcam mode and configure the appropriate camera index.

```python
camera_index = 0
```

If the camera is unavailable, try another index or close other applications that may be using it.

### Video file

For prerecorded footage, select video mode and provide the video path.

Example:

```text
resources/example_video.mp4
```

Ensure the video file exists and that the path is accessible from the project directory.

### RTSP stream

ACT can also receive video from an RTSP stream.

One way to provide an RTSP stream from a Windows webcam is to use MediaMTX and FFmpeg.

Start MediaMTX:

```bash
docker run --rm -it \
  --name mediamtx \
  -p 8554:8554 \
  -e MTX_RTSPTRANSPORTS=tcp \
  bluenviron/mediamtx:latest
```

In another terminal, identify the webcam device:

```bash
ffmpeg -list_devices true -f dshow -i dummy
```

Start streaming the webcam:

```bash
ffmpeg \
  -f dshow \
  -i video="Integrated Camera" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -pix_fmt yuv420p \
  -f rtsp \
  -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/act
```

Replace `Integrated Camera` with the device name reported by FFmpeg.

When ACT runs directly on Windows, use:

```text
rtsp://127.0.0.1:8554/act
```

When ACT runs inside Docker Desktop, use:

```text
rtsp://host.docker.internal:8554/act
```

The RTSP stream must be running before ACT attempts to connect.

---

## Language Model Setup

ACT uses Qwen through Ollama for local language-model inference.

### 1. Install Ollama

Download and install Ollama:

https://ollama.com/

### 2. Download the Qwen model

Open a terminal and run:

```bash
ollama pull qwen2.5:7b
```

Use the model identifier expected by the current ACT source code. If the project specifies another Qwen model or tag, pull that exact model instead.

### 3. Start Ollama

Ollama normally runs as a background service after installation.

Verify that it is available:

```bash
ollama list
```

You can also test inference directly:

```bash
ollama run qwen2.5:7b
```

ACT should invoke the local Ollama service when its language-model component is triggered.

---

## Smart Home Integration

ACT includes Kasa smart-home integration for exploring context-aware device control.

The application connects to supported Kasa devices on the local network.

For this integration:

- Ensure the computer and smart-home device are connected to the same network.
- Verify that the device is supported by the installed Kasa library.
- Configure the device information using the settings expected by the application.
- Confirm that the device is reachable before testing commands.

Smart-home actions should be treated as consequential operations. ACT should not execute an action solely because a visual activity prediction was made. User confirmation and the ability to interrupt or disable the system are important safeguards.

---

## Monitoring

ACT includes a Prometheus metrics component and a Grafana dashboard configuration for observing application behavior.

The intended metrics include:

| Metric | Description |
|---|---|
| `act_predictions_total` | Total activity predictions, grouped by predicted class |
| `act_processing_seconds` | Time spent processing the measured prediction operation |
| `act_prediction_confidence` | Most recent activity prediction confidence |

The processing-time metric must be interpreted according to where its timer is placed in the source code. If it surrounds only the Random Forest prediction, it does not represent the total YOLO, MediaPipe, and video-processing pipeline latency.

The intended local service addresses are:

| Service | Address |
|---|---|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| ACT metrics | http://localhost:8000/metrics |

Monitoring is optional for basic local execution. The metrics endpoint must be running and accessible before Prometheus can collect data.

---

## Docker Deployment

Docker can be used to package ACT and its supporting monitoring services.

Build and start the services:

```bash
docker compose up --build -d
```

Check running containers:

```bash
docker compose ps
```

View ACT logs:

```bash
docker compose logs -f act
```

Stop the services:

```bash
docker compose down
```

The application container should expose its metrics endpoint on port `8000`. Prometheus should scrape the ACT service using its Compose service name and internal port:

```text
act:8000
```

Inside a Docker Compose network, `localhost` refers to the current container, not another service.

**Camera access:** Running ACT in a Linux container on Windows does not automatically provide access to the Windows webcam through OpenCV's camera index. An RTSP stream or another supported video input method is needed to provide the camera feed to the container.

**Model files:** Ensure the trained activity model, YOLO weights, and MediaPipe task model are available inside the container, either through the image, mounted volumes, or a model-download step.

---

## Troubleshooting

### ModuleNotFoundError

Confirm that the virtual environment is active and dependencies are installed:

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

### YOLO weights cannot be found

Check that the weights exist and that the path in the source code matches the filename.

For example:

```text
YOLO/yolov8n.pt
```

### MediaPipe model cannot be found

Confirm that the task file exists:

```text
POSES/pose_landmarker_heavy.task
```

The configured path and downloaded filename must match.

### Webcam cannot be opened

- Verify that the camera is connected.
- Close other programs using the webcam.
- Try another camera index.
- Confirm that the correct input mode is selected.

### RTSP connection fails

- Confirm that MediaMTX is running.
- Confirm that FFmpeg is publishing to the correct stream path.
- Check that ACT is using the correct address for its execution environment.
- For Docker Desktop, use `host.docker.internal` rather than the container's own `localhost`.

### Ollama cannot connect

Confirm that Ollama is running:

```bash
ollama list
```

Verify that the model expected by the application has been downloaded.

### Slow inference

Processing time depends on the detector size, pose-estimation model, hardware, frame resolution, and input source.

A smaller detector or MediaPipe Full may improve throughput, while MediaPipe Heavy may provide better landmark localization under challenging visual conditions.

---

## Limitations and Future Improvements

ACT is an experimental computer vision assistant, not a fully generalized activity-understanding system.

Current limitations include:

- Activity recognition depends on the available training data and predefined activity classes.
- Unfamiliar routines may reduce prediction accuracy.
- Occlusion, camera placement, lighting, and viewing angle can affect object detection and pose estimation.
- Some actions may be missed or incorrectly classified.
- Higher-level activities require multiple observations and can be ambiguous when considered from a single frame.
- Larger vision models may increase processing time without guaranteeing better activity classification.
- Language-model responses depend on the quality and relevance of the context provided to the model.

Potential future improvements include:

1. Expanding the activity dataset with additional users, environments, and routines.
2. Improving temporal modeling to capture sequences of actions rather than relying primarily on individual observations.
3. Adding confidence-based event confirmation and better handling of uncertain predictions.
4. Improving real-time inference performance.
5. Evaluating event prediction across longer recordings and more diverse environments.
6. Expanding smart-home integrations and user-controlled automation.
7. Improving monitoring to measure end-to-end processing latency and prediction reliability.

---

## Project Status

ACT is an ongoing AI Systems project exploring the integration of computer vision, machine learning, and local language models into a context-aware assistant.

The current implementation includes object detection, pose estimation, action recognition, activity classification, local language-model integration, and smart-home integration components.

The project is intended as a foundation for further experimentation with assistants that can interpret physical context and provide useful assistance without requiring every interaction to begin with a manually written prompt.

## License

Add the project's license here before redistributing or permitting reuse of the source code.
