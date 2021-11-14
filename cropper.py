import glob
import cv2
import numpy as np

fnamesJPG = glob.glob('./Vision Images/*.jpg')
fnamesJPG.sort()
fnamesTXT = glob.glob('./Vision Images/*.txt')
fnamesTXT.sort()

image = cv2.imread('./Vision Images/0.jpg')
n = 0

for i in range(0,254):
    file = open('./Vision Images/' + str(i) + '.txt', 'r')
    image = cv2.imread('./Vision Images/' + str(i) + '.jpg')
    for line in file.readlines():
        fields = line.split()
        right = int((float(fields[1]) + float(fields[3])/2) * image.shape[1]) + np.random.randint(10)
        if right > image.shape[1] : right = image.shape[1]
        left = int((float(fields[1]) - float(fields[3])/2) * image.shape[1]) - np.random.randint(10)
        if left < 0 : left = 0
        top = int((float(fields[2]) - float(fields[4])/2) * image.shape[0]) - np.random.randint(10)
        if top < 0 : top = 0
        bottom = int((float(fields[2]) + float(fields[4])/2) * image.shape[0]) + np.random.randint(10)
        if bottom > image.shape[0] : bottom = image.shape[0]
        im = image[top:bottom,left:right]
        if fields[0] == '0': 
            cv2.imwrite('./images/R' + str(n) + '.jpg', im)
        elif fields[0] == '1':
            cv2.imwrite('./images/B' + str(n) + '.jpg', im)
        else:
            cv2.imwrite('./images/Y' + str(n) + '.jpg', im)
        n = n + 1




    
