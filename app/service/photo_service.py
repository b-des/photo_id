import base64
import io
import os
import shutil
import textwrap
import time
from io import BytesIO
import logging
import cv2
import dlib
import imutils
import requests

from app import config
from ..utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, save_tmp_file, \
    send_file_over_http, remove_tmp_dir
import numpy as np
from PIL import Image as PillowImage, ImageDraw, ImageFont, ImageOps, ImageEnhance
from sklearn.cluster import KMeans
from removebg import RemoveBg
import matplotlib.pyplot as plt
import uuid
logger = logging.getLogger(config.LOGGER_NAME)


class PhotoService:
    image = []
    document_dimensions = {}
    original_head_height = 0
    center_of_face = 0
    host = ''

    def __init__(self, image_url, dimensions, debug=False):
        self.debug = debug
        self.document_dimensions = dimensions
        self.face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE_PATH)
        self.shape_predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_FILE_PATH)
        self.frontal_face_detector = dlib.get_frontal_face_detector()

        # read image
        self.image = imutils.url_to_image(image_url)

        # remove alpha channel on png image
        if image_url.endswith('png'):
            # replace transparent background with white color
            # make mask of where the transparent bits are
            trans_mask = self.image[:, :, 2] == 0

            # replace areas of transparency with white and not transparent
            self.image[trans_mask] = [255, 255, 255]

            # new image without alpha channel...
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

        # convert to RGB scheme
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def auto_align_face(self):
        gray_copy = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dets = self.frontal_face_detector(gray_copy, 1)

        for i, det in enumerate(dets):
            shape = self.shape_predictor(self.image, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)

            matrix = get_rotation_matrix(left_eye, right_eye)
            self.image = cv2.warpAffine(self.image, matrix, (self.image.shape[1], self.image.shape[0]),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def detect_face(self, use_morphology=False):
        # convert the image to grayscale
        gray_copy = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray_copy, (3, 3), 0)
        # get binary image from grayscale image
        _, binary = cv2.threshold(gauss, 235, 255, cv2.THRESH_BINARY_INV)

        if use_morphology is True:
            # Morph close and invert image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = 225 - cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # get contours of the objects
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return []
        contours = contours[0]

        # draw debug contour
        image = self.image
        if self.debug is True:
            image = cv2.drawContours(self.image, contours, -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(contours)

        # get gray copy
        gray_copy = gray_copy[y:y + h, x:x + w]

        if 'develop' in os.environ['environment']:
            plt.imshow(image)
            plt.colorbar()
            plt.show()

        # apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
        gauss = cv2.GaussianBlur(gray_copy, (3, 3), 0)

        # detect faces in the image
        faces = self.face_cascade.detectMultiScale(
            gauss,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print("found {0} faces!".format(len(faces)))

        if len(faces) == 1:
            # crop image by contours(cut object from the image)
            self.image = image[y:y + h, x:x + w]
        else:
            return []
        return faces

    def detect_landmarks(self, face):
        x, y, w, h = face
        top_head = ((0, 0), (self.image.shape[1], 0))

        # converting the opencv rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # detecting landmarks
        detected_landmarks = self.shape_predictor(self.image, dlib_rect).parts()
        # converting to np matrix
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        # landmarks array contains indices of landmarks.
        # making another copy  for showing final result
        result = self.image.copy()

        # making temporary copy
        temp = self.image.copy()
        # getting area of interest from image i.e., forehead (25% of face)
        forehead = temp[y:y + int(0.25 * h), x:x + w]
        rows, cols, bands = forehead.shape
        X = forehead.reshape(rows * cols, bands)
        """
            Applying kmeans clustering algorithm for forehead with 2 clusters 
            this clustering differentiates between hair and skin (thats why 2 clusters)
            """
        # kmeans
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)
        for i in range(0, rows):
            for j in range(0, cols):
                if y_kmeans[i * cols + j] == True:
                    forehead[i][j] = [255, 255, 255]
                if y_kmeans[i * cols + j] == False:
                    forehead[i][j] = [0, 0, 0]

        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], top_head[0][1])

        # draw reference lines in debug mode
        if self.debug is True:
            # drawing line1 on forehead with circles
            cv2.line(result, top_head[0], top_head[1], color=(0, 255, 0), thickness=2)

            # drawing line 2 with circles
            linepointleft = (landmarks[1, 0], landmarks[1, 1])
            linepointright = (landmarks[15, 0], landmarks[15, 1])
            cv2.line(result, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
            cv2.circle(result, linepointleft, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(result, linepointright, 5, color=(255, 0, 0), thickness=-1)

            # drawing line 3 with circles
            linepointleft = (landmarks[3, 0], landmarks[3, 1])
            linepointright = (landmarks[13, 0], landmarks[13, 1])
            cv2.line(result, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
            cv2.circle(result, linepointleft, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(result, linepointright, 5, color=(255, 0, 0), thickness=-1)

            # drawing line 4 with circles
            cv2.line(result, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
            cv2.circle(result, linepointtop, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(result, linepointbottom, 5, color=(255, 0, 0), thickness=-1)

            # draw a rectangle around the face
            cv2.rectangle(result, (x, top_head[0][1]), (x + w, linepointbottom[1]), (0, 255, 255), 2)

        self.original_head_height = linepointbottom[1] - top_head[0][1]
        self.center_of_face = landmarks[8, 0]

        self.image = result
        if 'develop' in os.environ['environment']:
            plt.imshow(result)
            plt.colorbar()
            plt.show()

    def generate_photo_with_size(self, width, height, top_head_line, bottom_head_line):
        # create blank image
        background = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

        # draw debug lines
        if self.debug is True:
            # draw top head line
            cv2.line(background, (0, top_head_line), (width, top_head_line),
                     color=(0, 255, 0),
                     thickness=2)

            # draw bottom head line
            cv2.line(background, (0, bottom_head_line), (width, bottom_head_line),
                     color=(0, 255, 255),
                     thickness=2)

            # draw vertical head line - center of head
            cv2.line(background, (int(width / 2), 0),
                     (int(width / 2), height),
                     color=(0, 255, 255),
                     thickness=2)

        # required height of head
        needed_head_height = bottom_head_line - top_head_line

        # coefficient of difference between original head size and required
        k = needed_head_height / self.original_head_height

        # offset along the x-axis to place the face at the center of the canvas
        result = imutils.translate(self.image, int(self.image.shape[1] / 2 - self.center_of_face), 0)

        # resize image by the highest side
        result = imutils.resize(result, height=int(result.shape[0] * k))

        # calculate offset to put result image at the center of frame
        offset_x = int((result.shape[1] - width) / 2)

        # create PIL image from the processed photo
        result = PillowImage.fromarray(result)
        # create PIL image from the frame
        background = PillowImage.fromarray(background)
        # put processed photo onto frame using offset by X and Y axis
        background.paste(result, (-offset_x, top_head_line))

        # draw debug line at the bottom of face
        if self.debug is True:
            d = ImageDraw.Draw(background)
            d.line([(0, bottom_head_line), (width, bottom_head_line)], fill='red', width=2)

        # save image as PIL object
        self.image = background

        if 'develop' in os.environ['environment']:
            plt.imshow(self.image)
            plt.colorbar()
            plt.show()

    def get_result(self, size=None):
        '''
        Return base64 encoded image
        Parameters
        ----------
        size

        Returns
        -------
        str - base64 string

        '''
        if size is not None and size[0] > 0:
            self.image = self.add_watermark(image=self.image)
            self.image.thumbnail(size, PillowImage.ANTIALIAS)
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG", quality=100)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    def save_generated_photo(self, uid, hue='', corner='none', ext=config.DEFAULT_PHOTO_EXT):
        file_name = '{}.{}'.format(config.RESULT_PHOTO_NAME, ext)
        # convert to grayscale if needed
        if hue == 'gray':
            self.image = ImageOps.grayscale(self.image)
        # draw triangular corner
        self.image = self.__draw_corner_triangle__(image=self.image, corner_position=corner)
        tmp_file = save_tmp_file(uid=uid, image=self.image, file_name=file_name)
        result = send_file_over_http(host=self.host, file_path=tmp_file, uid=uid, photo_name=file_name)
        return result

    @classmethod
    def add_watermark(cls, image, text='demo'):

        width, height = image.size

        watermark = PillowImage.new('RGBA', (width, height), (0, 0, 0, 255))
        font = ImageFont.truetype("fonts/Harabara-Mais-Demo.otf", 24)
        mask = PillowImage.new('L', (width, height), color=60)
        draw = ImageDraw.Draw(mask)

        text = textwrap.fill(text)
        text_size = draw.textsize(text, font)

        for x in range(width)[10::text_size[0] + 50]:
            for y in range(height)[::text_size[1] * 2]:
                draw.text((x, y), text, font=font)

        watermark.putalpha(mask)

        image.paste(watermark, (0, 0), watermark)
        return image

    @classmethod
    def save_base64_to_image(cls, base64_string, host, uid, hue='', corner='none', ext=config.DEFAULT_PHOTO_EXT, size=None):

        try:
            img_data = base64.b64decode(base64_string.replace("data:image/png;base64,", ""))
        except Exception as e:
            logger.exception("Can't read base64 image")
            return None
        image = PillowImage.open(io.BytesIO(img_data))
        image = image.convert('RGB')
        # resize image if size is present in request
        if size is not None and size[0] is not None and size[1] is not None:
            image = image.resize(size, PillowImage.ANTIALIAS)
        # convert to grayscale if needed
        if hue == 'gray':
            image = ImageOps.grayscale(image)
        # draw triangular corner
        image = cls.__draw_corner_triangle__(image=image, corner_position=corner)
        file_name = '{}.{}'.format(config.RESULT_PHOTO_NAME, ext)
        tmp_file = save_tmp_file(uid=uid, image=image, file_name=file_name)
        return send_file_over_http(host=host, file_path=tmp_file, uid=uid, photo_name=file_name)

    @classmethod
    def remove_photo_bg(cls, image_url, is_full_size=False, t_uid=None, remove_bg=False):
        uid = uuid.uuid4() if t_uid is None else t_uid

        tmp_dir = config.TMP_IMAGE_PATH.format(uid)

        no_bg_photo_name = '{}.{}'.format(config.NO_BG_PHOTO_NAME, config.DEFAULT_PHOTO_EXT)
        if is_full_size:
            no_bg_photo_name = '{}.{}'.format(config.NO_BG_BIG_SIZE_PHOTO_NAME, config.DEFAULT_PHOTO_EXT)

        original_photo_name = '{}.{}'.format(config.ORIGINAL_PHOTO_NAME, config.DEFAULT_PHOTO_EXT)

        no_bg_photo_path = '{}/{}'.format(tmp_dir, no_bg_photo_name)
        original_photo_path = '{}/{}'.format(tmp_dir, original_photo_name)

        # save original image in target directory
        img = imutils.url_to_image(image_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PillowImage.fromarray(img)
        save_tmp_file(uid, img, original_photo_name)

        # if use not regular size
        # don't save original image in target directory
        if not is_full_size:
            send_file_over_http(host=cls.host, file_path=original_photo_path, uid=uid,
                                photo_name=original_photo_name, remove_tmp_path=False)

        # remove background if key is present and received parameter to remove BG
        if config.REMOVE_BG_API_KEY is not None and config.IS_PROD is True and remove_bg is True:
            remove_bg = RemoveBg(config.REMOVE_BG_API_KEY, "")
            logger.info("Going to remove background, uid: %s, image url: %s", uid, image_url)
            try:
                size = 'regular'
                if is_full_size:
                    size = 'full'
                remove_bg.remove_background_from_img_url(image_url, new_file_name=no_bg_photo_path,
                                                         bg_color='white', size=size)
            except Exception as e:
                logger.error("Failed to remove background, uid: %s, image url: %s, reason: %s", uid, image_url, str(e))
                # save original instead
                save_tmp_file(uid, img, no_bg_photo_name)
        else:
            logger.info("Don't remove BG due to absence API KEY or none prod mode, uid: %s, image url: %s",
                         uid, image_url)
            # if no key - save original image instead
            save_tmp_file(uid, img, no_bg_photo_name)

        # save image without background in target directory
        result = send_file_over_http(host=cls.host, file_path=no_bg_photo_path, uid=uid, photo_name=no_bg_photo_name)

        if not is_full_size:
            # create image with watermark
            image_with_watermark = PillowImage.open(no_bg_photo_path)
            image_with_watermark = cls.add_watermark(image_with_watermark)
            # save ia with watermark
            watermark_photo_name = '{}.{}'.format(config.NO_BG_WATERMARK_PHOTO_NAME, config.DEFAULT_PHOTO_EXT)
            watermark_photo_path = '{}/{}'.format(tmp_dir, watermark_photo_name)
            save_tmp_file(uid, image_with_watermark, watermark_photo_name)
            watermark_result = send_file_over_http(host=cls.host, file_path=watermark_photo_path, uid=uid,
                                                   remove_tmp_path=False)
            result['watermark_url'] = watermark_result['url']

        return result

    @staticmethod
    def __draw_corner_triangle__(image, corner_position):
        if corner_position != "none":
            corner_size = int(image.size[0] / 2)
            logger.info("Draw triangle. Image width: %s, corner size: %s", image.size[0], corner_size)
            rotation_angles = {
                "TL": {
                    'angle': 180,
                    'position': (0, 0),
                },

                "TR": {
                    'angle': 90,
                    'position': (image.size[0] - corner_size, 0)
                },
                "BR": {
                    'angle': 0,
                    'position': (image.size[0] - corner_size, image.size[1] - corner_size)
                },
                "BL": {
                    'angle': -90,
                    'position': (0, image.size[1] - corner_size)
                }
            }
            corner_param = rotation_angles[corner_position]
            triangle = PillowImage.open('static/triangle.png')
            triangle = triangle.convert('RGBA')
            triangle = triangle.rotate(corner_param['angle'])
            triangle.thumbnail((corner_size, corner_size), PillowImage.ANTIALIAS)
            image.paste(triangle, corner_param['position'], triangle)
        return image

    def __del__(self):
        print("Class {} was deleted".format(self.__module__.__str__()))
