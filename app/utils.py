import logging
import os
import shutil

import imutils
import numpy
import numpy as np
import cv2
import requests
from PIL import Image
import cv2
from app import config

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
logger = logging.getLogger(config.LOGGER_NAME)


def count_number_of_faces(url):
    logger.info("Counting faces....")
    face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE_PATH)
    img = imutils.url_to_image(url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = 0
    for i in numpy.arange(1.1, 1.9, 0.1):
        logger.info("Try with scale factor: %s", i)
        faces = face_cascade.detectMultiScale(gray, i, 5)
        faces = len(faces)
        logger.info("Number of faces: %s", faces)
        if faces == 1:
            return 1
    logger.info("No face detected")
    return faces


def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)


def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)


def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


def save_tmp_file(uid, image: Image, file_name='blank.jpg'):
    tmp_dir = config.TMP_IMAGE_PATH.format(uid)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    photo_name = '{}/{}'.format(tmp_dir, file_name)
    image.save(photo_name, quality=100, dpi=(600, 600))
    return photo_name


def remove_tmp_dir(uid=None):
    if uid is not None:
        tmp_dir_path = config.TMP_IMAGE_PATH.format(uid)
        try:
            shutil.rmtree(os.path.dirname(tmp_dir_path))
        except Exception:
            pass


def send_file_over_http(host, file_path, uid, photo_name=None, remove_tmp_path=True):
    file = open(file_path, 'rb')
    head, tail = os.path.split(file_path)
    data = {
        'uid': uid,
        'photo_name': photo_name or tail
    }
    files = {
        'file': file,
    }
    if host.find('localhost') != -1:
        host = 'http://localhost/{}'.format(config.HANDLER_URL)
    else:
        host = '{}/{}'.format(host, config.HANDLER_URL)

    try:
        result = requests.post(host, files=files, data=data)
        result.raise_for_status()
    except Exception as e:
        logger.error("Failed to send file: %s to host: %s, uid: %s, reason: %s", file_path, host, uid, str(e))
        return {}
    finally:
        if remove_tmp_path is True:
            shutil.rmtree(os.path.dirname(file_path))
    return result.json()
