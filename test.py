from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time
import re
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
im_width = 50
im_height = 50

# src_path = "/home/train/alireza/class/tensor2/recognition/meli/TrainData"
dst_path = "/home/train/alireza/class/tensor2/noise_platetype_classifier/resized"
checkpoint_path = "./"
modelpath = './first'
imgpath = './redtest/R51.jpg'
model = Sequential()
inputShape = (im_height, im_width,3)
chanDim = -1
model.add(Conv2D(8, (5, 9), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(16, (5, 9), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation("sigmoid"))
model.load_weights(modelpath)
model.summary()


labels_list = ["blue","red"]

x = cv2.imread(imgpath)
x = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
x = cv2.resize(x, (im_width,im_height), interpolation=cv2.INTER_AREA)
x = np.array(x, dtype="float") / 255.0
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
firstrime = time.time()
for i in range(10):
    classes = model.predict(images, batch_size=1)
secondtime = time.time()

print(secondtime - firstrime)
print(classes)

