from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps
import math
from scipy import ndimage
import tensorflow as tf

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test, y_test)

model = tf.keras.models.load_model('handwriting_digits.h5')


img = cv2.imread('pictures/7.jpg')

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)

# cv2.imshow('resize_image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' '''
#https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

img = cv2.merge(result_planes)
img_norm = cv2.merge(result_norm_planes)
''' ''' ''' ''' ''' ''' ''' ''' ''' ''' ''' '''
gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_image',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #gray = cv2.resize(255-gray, (28, 28))


(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #makes the image binary

# cv2.imshow('gray_image',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




gray = cv2.bitwise_not(gray) #inverted image
cv2.imshow('final_image',gray) #image should look similar to the mnist images
cv2.waitKey(0)
cv2.destroyAllWindows()


##########################
#https://github.com/opensourcesblog/tensorflow-mnist/blob/master/mnist.py
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

while np.sum(gray[0]) == 0:
	gray = gray[1:]
while np.sum(gray[:,0]) == 0:
	gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
	gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
	gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
	factor = 20.0/rows
	rows = 20
	cols = int(round(cols*factor))
	# first cols than rows
	gray = cv2.resize(gray, (cols,rows))
else:
	factor = 20.0/cols
	cols = 20
	rows = int(round(rows*factor))
	# first cols than rows
	gray = cv2.resize(gray, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted
#this code snippet I copied helps center the number in the image
###########################
gray = np.expand_dims(gray, axis=0)
numbers = model.predict(gray)
print(tf.math.argmax(numbers,axis=0))
numbers = np.array(numbers).tolist()
numbers = numbers[0]
x = 0
maximum = 0
#tf.math.argmax doesn't work with floats so I had to resort to doing it my own way
for i in range(len(numbers)):
    if numbers[i] > maximum:
        x = i
        maximum = numbers[i]
print(f"Number: {x}") #hopefully predicts the correct number!!

