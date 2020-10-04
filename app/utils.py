import os
import shutil

import numpy as np
import cv2
import requests
from PIL import Image

from app import config

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]


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


def save_tmp_file(uid, image: Image, file_name='blank'):
    tmp_dir = '{}/{}'.format(config.TMP_IMAGE_PATH, uid)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    photo_name = '{}/{}'.format(tmp_dir, file_name)
    image.save(photo_name, quality=100, dpi=(600, 600))
    return photo_name


def send_file_over_http(host, file_path, uid, photo_name="blank.jpg"):
    no_bg_photo = open(file_path, 'rb')
    data = {
        'uid': uid,
        'photo_name': photo_name
    }
    files = {
        'file': no_bg_photo,
    }
    host = '{}/{}'.format(host, config.HANDLER_URL)
    try:
        result = requests.post(host, files=files, data=data)
        result.raise_for_status()
    except Exception:
        return {}
    finally:
        shutil.rmtree(os.path.dirname(file_path))
    return result.json()
