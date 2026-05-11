import os
import sys
import shutil

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_path()

PROTOTXT = os.path.join(
    BASE_DIR, "models", "face_detector", "deploy.prototxt"
)

MODEL = os.path.join(
    BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"
)

APP_DATA_DIR = BASE_DIR
DATA_DIR = os.path.join(APP_DATA_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")