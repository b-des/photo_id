import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_CASCADE_FILE_PATH = ROOT_DIR + "/data/haarcascade_frontalface_default.xml"
SHAPE_PREDICTOR_FILE_PATH = ROOT_DIR + "/data/shape_predictor_68_face_landmarks.dat"
TMP_IMAGE_PATH = '/tmp/photo-id'
HANDLER_URL = os.environ['HANDLER_URL'] or 'handler.php'
REMOVE_BG_API_KEY = os.environ['REMOVE_BG_API_KEY'] or None


MULTIPLIER = 10
FINAL_PHOTO_WIDTH = int(30 * MULTIPLIER)
FINAL_PHOTO_HEIGHT = int(40 * MULTIPLIER)
TOP_HEAD_LINE = int(2.5 * MULTIPLIER)
BOTTOM_HEAD_LINE = TOP_HEAD_LINE + int(31 * MULTIPLIER)
