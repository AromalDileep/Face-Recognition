---
title: Face Recognition
emoji: 👁
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.57.0"
app_file: app.py
pinned: false
---

# Live Face Recognition Web App

A real-time facial recognition application built with Streamlit and OpenCV. This application uses FaceNet for generating high-quality face embeddings and allows users to enroll faces, manage profiles, and perform live recognition directly through their web browser using WebRTC.

## Features

* **Face Enrollment**: Capture a photo from your webcam and register a name to save the face embedding to the local database.
* **Live Recognition**: Real-time video processing using WebRTC to detect and recognize faces instantly.
* **Manage Profiles**: View and delete existing enrolled profiles from the database.

## How to Run Locally

Because this repository contains large Machine Learning models (such as `facenet.onnx` which is ~94MB), **do not use the "Download ZIP" button on GitHub**. Downloading the ZIP will only download a text pointer instead of the actual model, causing the app to crash. 

Please follow these steps to set it up correctly:

### 1. Clone the Repository
You must have Git and Git LFS installed on your system. Run these commands in your terminal:
```bash
git lfs install
git clone https://github.com/AromalDileep/Face-Recognition.git
cd Face-Recognition
```

### 2. Install Dependencies
Ensure you have Python 3.11 or higher installed. You can install the required packages using standard `pip` or `uv`:

**Using standard pip:**
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Using uv (Recommended):**
```bash
uv pip sync requirements.txt
```

### 3. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## Important Note on Cloud Deployment
If you are deploying this to the public internet (like Hugging Face Spaces or Streamlit Cloud), the live camera feed (WebRTC) relies on STUN/TURN servers to penetrate firewalls. By default, it uses a free Google STUN server which works on most local networks, but for a 100% reliable internet deployment, you should configure a TURN server (like Twilio) in `app.py`.
