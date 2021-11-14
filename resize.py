import cv2
import os
path = "./dataset/blue/"
files = os.listdir(path)
for im in files:
    tmpim = cv2.imread(path + im)
    tmp = cv2.resize(tmpim, (50,50))
    cv2.imwrite(path + im, tmp)