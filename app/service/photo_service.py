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
    send_file_over_http
import numpy as np
from PIL import Image as PillowImage, ImageDraw, ImageFont, ImageOps, ImageEnhance
from sklearn.cluster import KMeans
from removebg import RemoveBg
import matplotlib.pyplot as plt
import uuid


class PhotoService:
    image = []
    document_dimensions = {}
    original_head_height = 0
    center_of_face = 0
    host = ''

    def __init__(self, image_url, dimensions, debug=False):
        print("Gonna to do hard task")
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
            scaleFactor=1.05,
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
        if size is not None and size[0] > 0:
            self.image.thumbnail(size, PillowImage.ANTIALIAS)
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG", quality=100)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    def save_generated_photo(self, uid, hue='', corner=0, scale=1):
        file_name = 'result.jpg'
        if hue == 'gray':
            self.image = ImageOps.grayscale(self.image)
        self.image = self.__draw_corner_triangle__(image=self.image, corner_position=corner, scale=scale)
        tmp_file = save_tmp_file(uid=uid, image=self.image, file_name=file_name)
        return send_file_over_http(host=self.host, file_path=tmp_file, uid=uid, photo_name=file_name)

    @classmethod
    def save_base64_to_image(cls, base64_string, host, uid, hue='', corner=0):
        img_data = base64.b64decode(base64_string.replace("data:image/png;base64,", ""))
        image = PillowImage.open(io.BytesIO(img_data))
        image = image.convert('RGB')
        if hue == 'gray':
            image = ImageOps.grayscale(image)
        image = cls.__draw_corner_triangle__(image=image, corner_position=corner)
        file_name = 'result.jpg'
        tmp_file = save_tmp_file(uid=uid, image=image, file_name=file_name)
        return send_file_over_http(host=host, file_path=tmp_file, uid=uid, photo_name=file_name)

    @classmethod
    def remove_photo_bg(cls, image_url):
        uid = uuid.uuid4()

        tmp_dir = '{}/{}'.format(config.TMP_IMAGE_PATH, uid)

        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        photo_name = '{}/no-bg.jpg'.format(tmp_dir)

        img = imutils.url_to_image(image_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PillowImage.fromarray(img)
        img.save(photo_name, quality=100)

        result = send_file_over_http(host=cls.host, file_path=photo_name, uid=uid, photo_name='no-bg.jpg')
        return result

    @classmethod
    def add_watermark(cls, uid, text='Demo'):
        start_time = time.time()
        tmp_dir = '{}/{}'.format(config.TMP_IMAGE_PATH, uid)
        photo_name = '{}/no-bg.jpg'.format(tmp_dir)
        new_photo_name = '{}/new-no-bg.jpg'.format(tmp_dir)
        img = PillowImage.open(photo_name)
        width, height = img.size

        watermark = PillowImage.new('RGBA', (width, height), (0, 0, 0, 255))
        font = ImageFont.truetype("fonts/Harabara-Mais-Demo.otf", 24)
        mask = PillowImage.new('L', (width, height), color=40)
        draw = ImageDraw.Draw(mask)

        text = textwrap.fill(text)
        text_size = draw.textsize(text, font)

        for x in range(width)[10::text_size[0] + 50]:
            for y in range(height)[::text_size[1] * 2]:
                draw.text((x, y), text, font=font)

        watermark.putalpha(mask)

        img.paste(watermark, (0, 0), watermark)
        img.save(new_photo_name, quality=100, dpi=(600, 600))
        data = {
            'uid': uid
        }

        watermark_photo = open(new_photo_name, 'rb')
        files = {
            'watermark': watermark_photo,
            'original': watermark_photo
        }
        result = requests.post('http://localhost/handler.php', files=files, data=data)
        print("--- %s seconds ---" % (time.time() - start_time))
        return result.json()

    @staticmethod
    def __draw_corner_triangle__(image, corner_position, scale=1):
        if int(corner_position) > 0:
            corner_size = int(50 * scale)
            rotation_angles = {
                1: {
                    'angle': 180,
                    'position': (0, 0),
                },
                2: {
                    'angle': 90,
                    'position': (image.size[0] - corner_size, 0)
                },
                3: {
                    'angle': 0,
                    'position': (image.size[0] - corner_size, image.size[1] - corner_size)
                },
                4: {
                    'angle': -90,
                    'position': (0, image.size[1] - corner_size)
                }
            }
            corner_param = rotation_angles[int(corner_position)]
            triangle = PillowImage.open('static/triangle.png')
            triangle = triangle.convert('RGBA')
            triangle = triangle.rotate(corner_param['angle'])
            triangle.thumbnail((corner_size, corner_size), PillowImage.ANTIALIAS)
            image.paste(triangle, corner_param['position'], triangle)
        return image

    def __del__(self):
        print("Class {} was deleted".format(self.__module__.__str__()))