import os
DEBUG = True
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_CASCADE_FILE_PATH = ROOT_DIR + "/data/haarcascade_frontalface_default.xml"
SHAPE_PREDICTOR_FILE_PATH = ROOT_DIR + "/data/shape_predictor_68_face_landmarks.dat"
TEST_IMAGE = ROOT_DIR + "/samples/me.jpg"

MULTIPLIER = 100
FINAL_PHOTO_WIDTH = int(3 * MULTIPLIER)
FINAL_PHOTO_HEIGHT = int(4 * MULTIPLIER)
TOP_HEAD_LINE = int(.4 * MULTIPLIER)
BOTTOM_HEAD_LINE = TOP_HEAD_LINE + int(2.5 * MULTIPLIER)
