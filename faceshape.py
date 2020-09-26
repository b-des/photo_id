import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PillowImage, ImageDraw
from sklearn.cluster import KMeans
import app as ptp
from app.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix
from app import config
import time

start_time = time.time()
# load the image
imagepath = "./samples/me.jpg"


# create the haar cascade for detecting face and smile
faceCascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE_PATH)

# create the landmark predictor
predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_FILE_PATH)

detector = dlib.get_frontal_face_detector()

# read the image
original_image = cv2.imread(imagepath)

#original_image = imutils.resize(original_image, height=ptp.FINAL_PHOTO_HEIGHT)

if 1 == 2:
    # make mask of where the transparent bits are
    trans_mask = original_image[:, :, 2] == 0

    # replace areas of transparency with white and not transparent
    original_image[trans_mask] = [255, 255, 255]

    # new image without alpha channel...
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)


image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# convert the image to grayscale
gray = cv2.cvtColor((image), cv2.COLOR_BGR2GRAY)

dets = detector(gray, 1)

for i, det in enumerate(dets):
    shape = predictor(image, det)
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)

    M = get_rotation_matrix(left_eye, right_eye)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# get contours of the face
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(contours)

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
image = image[y:y + h, x:x + w]
gray = gray[y:y + h, x:x + w]

topHead = ((0, 0), (w, 0))
center_of_face = 0


# apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
gauss = cv2.GaussianBlur(gray, (3, 3), 0)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gauss,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Detect faces in the image
print("found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    # draw a rectangle around the faces
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # converting the opencv rectangle coordinates to Dlib rectangle
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    # detecting landmarks
    detected_landmarks = predictor(image, dlib_rect).parts()
    # converting to np matrix
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    # landmarks array contains indices of landmarks.
    """
    #copying the image so we can we side by side
    landmark = image.copy()
    for idx, point in enumerate(landmarks):
            pos = (point[0,0], point[0,1] )
            #annotate the positions
            cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
            #draw points on the landmark positions 
            cv2.circle(landmark, pos, 3, color=(0,255,255))
    
cv2.imshow("Landmarks by DLib", landmark)
"""
# making another copy  for showing final result
result = image.copy()

for (x, y, w, h) in faces:

    # making temporary copy
    temp = image.copy()
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
    # Steps to get the length of forehead
    # 1.get midpoint of the forehead
    # 2.travel left side and right side
    # the idea here is to detect the corners of forehead which is the hair.
    # 3.Consider the point which has change in pixel value (which is hair)
    forehead_mid = [int(cols / 2), int(rows / 2)]  # midpoint of forehead
    lef = 0
    # gets the value of forehead point
    pixel_value = forehead[forehead_mid[1], forehead_mid[0]]
    for i in range(0, cols):
        # enters if when change in pixel color is detected
        if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
            lef = forehead_mid[0] - i
            break
    left = [lef, forehead_mid[1]]
    rig = 0
    for i in range(0, cols):
        # enters if when change in pixel color is detected
        if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
            rig = forehead_mid[0] + i
            break
    right = [rig, forehead_mid[1]]

# drawing line1 on forehead with circles
# specific landmarks are used.
line1 = np.subtract(right + y, left + x)[0]
# cv2.line(result, tuple(x + left), tuple(y + right), color=(0, 255, 0), thickness=2)

cv2.line(result, topHead[0], topHead[1], color=(0, 255, 0), thickness=2)

# drawing line 2 with circles
linepointleft = (landmarks[1, 0], landmarks[1, 1])
linepointright = (landmarks[15, 0], landmarks[15, 1])
line2 = np.subtract(linepointright, linepointleft)[0]
cv2.line(result, linepointleft, linepointright, color=(0, 255, 0), thickness=2)

cv2.circle(result, linepointleft, 5, color=(255, 0, 0), thickness=-1)
cv2.circle(result, linepointright, 5, color=(255, 0, 0), thickness=-1)

# drawing line 3 with circles
linepointleft = (landmarks[3, 0], landmarks[3, 1])
linepointright = (landmarks[13, 0], landmarks[13, 1])
line3 = np.subtract(linepointright, linepointleft)[0]
cv2.line(result, linepointleft, linepointright, color=(0, 255, 0), thickness=2)

cv2.circle(result, linepointleft, 5, color=(255, 0, 0), thickness=-1)
cv2.circle(result, linepointright, 5, color=(255, 0, 0), thickness=-1)

# drawing line 4 with circles
center_of_face = landmarks[8, 0]
linepointbottom = (landmarks[8, 0], landmarks[8, 1])
linepointtop = (landmarks[8, 0], topHead[0][1])
line4 = np.subtract(linepointbottom, linepointtop)[1]
cv2.line(result, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)

cv2.circle(result, linepointtop, 5, color=(255, 0, 0), thickness=-1)
cv2.circle(result, linepointbottom, 5, color=(255, 0, 0), thickness=-1)
# print(line1,line2,line3,line4)


# draw a rectangle around the face
cv2.rectangle(result, (x, topHead[0][1]), (x + w, linepointbottom[1]), (0, 255, 255), 2)

original_head_height = linepointbottom[1] - topHead[0][1]

# create blank image
background = 255 * np.ones(shape=[ptp.FINAL_PHOTO_HEIGHT, ptp.FINAL_PHOTO_WIDTH, 3], dtype=np.uint8)

# Draw top head line
cv2.line(background, (0, ptp.TOP_HEAD_LINE), (ptp.FINAL_PHOTO_WIDTH, ptp.TOP_HEAD_LINE), color=(0, 255, 0), thickness=2)
# Draw bottom head line
cv2.line(background, (0, ptp.BOTTOM_HEAD_LINE), (ptp.FINAL_PHOTO_WIDTH, ptp.BOTTOM_HEAD_LINE), color=(0, 255, 255),
         thickness=2)

# Draw vertical head line
cv2.line(background, (int(ptp.FINAL_PHOTO_WIDTH / 2), 0), (int(ptp.FINAL_PHOTO_WIDTH / 2), ptp.FINAL_PHOTO_HEIGHT),
         color=(0, 255, 255),
         thickness=2)

# background2[y_offset:y_offset+result.shape[0], x_offset:x_offset+result.shape[1]] = result

needed_head_height = ptp.BOTTOM_HEAD_LINE - ptp.TOP_HEAD_LINE
print(original_head_height, needed_head_height)
k = needed_head_height / original_head_height
print(k)
aspect_ratio = float(image.shape[1]) / float(image.shape[0])

# offset along the x-axis to place the face at the center of canvas
result = imutils.translate(result, int(result.shape[1] / 2 - center_of_face), 0)
# result = cv2.rectangle(result, (0, 0), (result.shape[1], ptp.TOP_HEAD_LINE), color=(255, 255, 255), thickness=-1)

# resize image by the highest side
result = imutils.resize(result, height=int(result.shape[0] * k))

# cv2.line(result, (0, ptp.TOP_HEAD_LINE), (ptp.FINAL_PHOTO_WIDTH, ptp.TOP_HEAD_LINE), color=(0, 0, 255), thickness=2)
# cv2.line(result, (0, ptp.BOTTOM_HEAD_LINE), (ptp.FINAL_PHOTO_WIDTH, ptp.BOTTOM_HEAD_LINE), color=(0, 255, 255), thickness=2)


rows, cols, channels = result.shape
offset_x = int((result.shape[1] - ptp.FINAL_PHOTO_WIDTH) / 2)
offset_y = 0
# result = result[offset_y:ptp.FINAL_PHOTO_HEIGHT, offset_x:ptp.FINAL_PHOTO_WIDTH + offset_x]

# result = cv2.addWeighted(background, 0, face, 1, 0)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
cv2.imwrite("./samples/me-1.jpg", result)

result = PillowImage.fromarray(result)
background = PillowImage.fromarray(background)

background.paste(result, (-offset_x, ptp.TOP_HEAD_LINE))

d = ImageDraw.Draw(background)
d.line([(0, ptp.BOTTOM_HEAD_LINE), (ptp.FINAL_PHOTO_WIDTH, ptp.BOTTOM_HEAD_LINE)], fill='red', width=2)

#background.save("./samples/me-1.jpg")

print("--- %s seconds ---" % (time.time() - start_time))
plt.imshow(background)
plt.colorbar()
plt.show()

# retval, buffer = cv2.imencode('.jpg', background2)
# b64 = base64.b64encode(buffer)
# print(b64)
