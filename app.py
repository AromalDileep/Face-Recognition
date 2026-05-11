import streamlit as st
import cv2
import numpy as np
import av
import logging

from core.face_detector import FaceDetector
from core.embedder import FaceEmbedder
from core.enrollment import Enroller
from core.recognition import FaceRecognizer
from utils.storage import load_embeddings
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Face Recognition App", layout="wide")

# Initialize models and state
@st.cache_resource
def load_models():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    return detector, embedder

try:
    detector, embedder = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Load embeddings into session state so it's globally available
if "embeddings_db" not in st.session_state:
    st.session_state.embeddings_db = load_embeddings()

enroller = Enroller(st.session_state.embeddings_db)
recognizer = FaceRecognizer(st.session_state.embeddings_db)

import os
from twilio.rest import Client

@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server if credentials are set, fallback to free STUN."""
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if account_sid and auth_token:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
    except Exception as e:
        logging.warning(f"Failed to fetch Twilio TURN server: {e}")
        
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": get_ice_servers()}
)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Detect faces
    boxes = detector.detect(img)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Extract face
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
            
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_expanded = np.expand_dims(face_resized, axis=0)
        
        # Get embedding
        emb = embedder.get_embedding(face_expanded)
        
        # Recognize
        name, score = recognizer.recognize(emb)
        
        # Draw bounding box and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} ({score:.2f})", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Enrollment", "Live Recognition", "Manage Profiles"])

if page == "Enrollment":
    st.title("Face Enrollment")
    st.write("Enter your name and take a picture to enroll your face in the system.")
    
    name = st.text_input("Name")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None and name:
        # Convert the file buffer to an opencv image
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detect face
        boxes = detector.detect(cv2_img)
        
        if len(boxes) == 0:
            st.error("No face detected. Please try again.")
        elif len(boxes) > 1:
            st.error("Multiple faces detected. Please ensure only one person is in the frame.")
        else:
            box = boxes[0]
            x1, y1, x2, y2 = box
            
            face = cv2_img[y1:y2, x1:x2]
            if face.size > 0:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (160, 160))
                face_expanded = np.expand_dims(face_resized, axis=0)
                
                # Get embedding
                emb = embedder.get_embedding(face_expanded)
                
                # Enroll
                enroller.enroll(name, emb)
                
                # Update recognizer to use new embeddings immediately
                recognizer.update_db(enroller.db)
                
                # Draw box on image to show user
                cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Enrolled Face")
                
                st.success(f"Successfully enrolled {name}!")

elif page == "Live Recognition":
    st.title("Live Face Recognition")
    st.write("Start the video stream to detect and recognize faces.")
    
    webrtc_streamer(
        key="recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif page == "Manage Profiles":
    st.title("Manage Profiles")
    st.write("View and remove existing enrolled profiles.")
    
    profiles = list(enroller.db.keys())
    
    if not profiles:
        st.info("No profiles enrolled yet.")
    else:
        st.write("### Enrolled Profiles")
        for profile_name in profiles:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{profile_name}**")
            with col2:
                if st.button("Remove", key=f"remove_{profile_name}"):
                    if enroller.remove(profile_name):
                        recognizer.update_db(enroller.db)
                        st.success(f"Removed profile: {profile_name}")
                        st.rerun()
