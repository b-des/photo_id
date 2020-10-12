import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_CASCADE_FILE_PATH = ROOT_DIR + "/data/haarcascade_frontalface_default.xml"
SHAPE_PREDICTOR_FILE_PATH = ROOT_DIR + "/data/shape_predictor_68_face_landmarks.dat"
TMP_IMAGE_PATH = '/tmp/photo-id/{}'
LOGGING_FILE = '/tmp/photo-id/logs/logs.log'
HANDLER_URL = os.environ['HANDLER_URL'] if 'HANDLER_URL' in os.environ else 'handler.php'
REMOVE_BG_API_KEY = os.environ['REMOVE_BG_API_KEY'] if 'REMOVE_BG_API_KEY' in os.environ else None
IS_PROD = True if 'environment' in os.environ and os.environ['environment'] == 'prod' else False
LOGGER_NAME = "root"
WATERMARK_TEXT_SIZE = 15

ORIGINAL_PHOTO_NAME = 'original'
NO_BG_PHOTO_NAME = 'original-without-bg'
NO_BG_WATERMARK_PHOTO_NAME = 'original-without-bg-watermark'
NO_BG_BIG_SIZE_PHOTO_NAME = 'original-without-bg-full-size'
RESULT_PHOTO_NAME = 'final-result'
DEFAULT_PHOTO_EXT = 'jpg'

NO_FACE = 'no_face'
MORE_ONE_FACES = 'more_one_faces'

