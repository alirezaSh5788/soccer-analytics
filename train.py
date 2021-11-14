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
import cv2 as cv
im_width = 50
im_height = 50
epcs=20
lrt=0.0001
bs=128
# src_path = "/home/train/alireza/class/tensor2/recognition/meli/TrainData"
dst_path = "./dataset/"
checkpoint_path = "./cp.ckpt"
# log_dir = "/home/train/alireza/class/tensor2/noise_platetype_classifier/tensorboard"

def getlabel(imagepath):
    label = imagepath.split("/")
    tmp = label[-2]

    return label

tmp = os.listdir(dst_path)

final = tmp

print(len(final))
labels_list = ["blue","red"]

print(len(labels_list))
print(labels_list)
mlb = MultiLabelBinarizer()
data = []
label_hot = []

label_hot = mlb.fit([labels_list])

for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
test = [['red']]
test = mlb.transform(test)
print(test)
test = [['blue']]
test = mlb.transform(test)
print(test)


def create(finalims):
    labels = []
    data = []
    for dirname in finalims:
        for image in os.listdir(dst_path + "//"+dirname):
            data.append(cv.imread(dst_path + "//" + dirname+"//"+image))
            imlabel = []
            imlabel.append(str(dirname))
            labels.append(imlabel)
    return data , labels
data , labels = create(finalims=final)
labels = mlb.transform(labels)

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
model.add(Dense(len(mlb.classes_)))
model.add(Activation("sigmoid"))
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.2, random_state=42)
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.10,
                             height_shift_range=0.10, shear_range=0.10, zoom_range=0.10,
                                horizontal_flip=False, fill_mode="nearest")
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

opt = Adam(lr=lrt, decay=lrt / epcs)
model.compile(loss="MSE", optimizer=opt,
                  metrics=METRICS)
model.summary()

checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    verbose=1)


# t_c = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


H = model.fit(
        x=aug.flow(trainX, trainY, batch_size=bs),
        validation_data=(testX, testY),
        batch_size=bs,
        callbacks=[cp_callback],
        epochs=epcs, verbose=1)
model.save("first", save_format="h5")
