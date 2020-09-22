import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_CASCADE_FILE_PATH = ROOT_DIR + "/data/haarcascade_frontalface_default.xml"
SHAPE_PREDICTOR_FILE_PATH = ROOT_DIR + "/data/shape_predictor_68_face_landmarks.dat"
