import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
import math
from math import degrees
import matplotlib.pyplot as plt
import base64


class Handler:
    multiplier = 10
    original_image = None

    def __init__(self) -> None:
        self.frameWidth = int(3 * self.multiplier)
        self.frameHeight = int(4 * self.multiplier)
        super().__init__()

    def read_image(self, image_path=None):
        self.original_image = cv2.imread(image_path)

    def find_contours(self):
        original = self.original_image.copy()

        image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        plt.imshow(binary, cmap="gray")
        # plt.show()

        # get image contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]

        # draw contours
        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # get rectangle from contours
        x, y, w, h = cv2.boundingRect(contours)

        # draw rectangle which fit in contours
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(image)
        plt.show()
        return gray

    def find_face(self):
        pass

    def create_dummy_background(self):
        pass

    def fill_image_to_frame(self):
        pass
