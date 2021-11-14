import os
import random
from shutil import copyfile

imagesPath = "./images/"
files = os.listdir(imagesPath)
Rfiles = [x for x in files if "R" in x]
Bfiles = [x for x in files if "B" in x]
random.shuffle(Rfiles)
random.shuffle(Bfiles)
Rindex = len(Rfiles)
Bindex = len(Bfiles)
RtrainIndex = int(Rindex*0.9)
BtrainIndex = int(Bindex*0.9)
Rtrain = Rfiles[:RtrainIndex]
Btrain = Bfiles[:BtrainIndex]

Rtest = Rfiles[RtrainIndex:]
Btest = Bfiles[BtrainIndex:]

if(not os.path.isdir("red")):
    os.mkdir("red")
if(not os.path.isdir("blue")):
    os.mkdir("blue")
for redim in Rtrain:
    src = imagesPath + redim
    dst = "./red/" + redim
    copyfile(src, dst)
for blueim in Btrain:
    src = imagesPath + blueim
    dst = "./blue/" + blueim
    copyfile(src, dst)

if(not os.path.isdir("redtest")):
    os.mkdir("redtest")
if(not os.path.isdir("bluetest")):
    os.mkdir("bluetest")
for redim in Rtest:
    src = imagesPath + redim
    dst = "./redtest/" + redim
    copyfile(src, dst)
for blueim in Btest:
    src = imagesPath + blueim
    dst = "./bluetest/" + blueim
    copyfile(src, dst)