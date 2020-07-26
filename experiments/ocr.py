#%%

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

#%%

digits = [ "digits/" + d for d in os.listdir("./digits")]

#%%

def resize_image(img, size=(20,20)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = (
        cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
        )

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def deskew(img, size=28):
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    
    m = cv2.moments(img)
    
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*size*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(size, size),flags=affine_flags)
    
    return img

def find_center(img):
    M = cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def align_center(img, size=28):
    pt1 = find_center(img)
    pt2 = (size//2, size//2)
    dst = np.zeros((size, size))
    
    dx = abs(pt2[0] - pt1[0])
    dy = abs(pt2[1] - pt1[1])
    
    dst[dy:dy + img.shape[1], dx:dx + img.shape[0]] = img
    return dst

#%%
i = 19
path = digits[i]
print(path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

squared_img = resize_image(~img, size=(20, 20))
centered_img = align_center(squared_img, size=28)
deskewed = deskew(centered_img, size=28)

fig, ax = plt.subplots(1, 3, dpi=300)
ax[0].imshow(squared_img, cmap=cm.gray)
ax[1].imshow(centered_img, cmap=cm.gray)
ax[2].imshow(deskewed, cmap=cm.gray)

#%%

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# deskewing the digits
# x_train = np.array([deskew(img) for img in x_train])
# x_test = np.array([deskew(img) for img in x_test])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#%%

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%%

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                   activation='relu',
#                   input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# model.save("model")

from tensorflow.keras.models import load_model
model = load_model('model')

#%%

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%

# testing digits base

new_test_x = []
new_test_y = []
for path in digits:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    
    squared_img = resize_image(~img, size=(20, 20))
    centered_img = align_center(squared_img, size=28)
    deskewed = centered_img
    # deskewed = deskew(centered_img, size=28)
    new_test_x.append(deskewed)
    new_test_y.append(float(path[-5]))
new_test_x = np.array(new_test_x)
new_test_y = np.array(new_test_y)

if K.image_data_format() == 'channels_first':
    new_test_x = new_test_x.reshape(new_test_x.shape[0], 1, img_rows, img_cols)
else:
    new_test_x = new_test_x.reshape(new_test_x.shape[0], img_rows, img_cols, 1)
    
new_test_x = new_test_x.astype('float32')
new_test_x /= 255

new_test_y = keras.utils.to_categorical(new_test_y, num_classes)

score = model.evaluate(new_test_x, new_test_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%

pred = model.predict_classes(new_test_x)
original = [int(path[-5]) for path in digits]
pairs = list(zip(original, pred))

for (ori, pred), path in zip(pairs, digits):
    if ori != pred:
        print((ori, pred), path)
