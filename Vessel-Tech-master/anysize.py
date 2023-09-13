import training
import prepare_datasets_DRIVE
import predicting
import feature_extraction
from os import listdir
from os.path import isfile, join
import shutil
from PIL import Image
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import cv2
import time
import tensorflow as tf

# create all the folders
tandp = 'static\I'
sep = 'static\images'
comb = 'combination'
if os.path.isdir(tandp):
    shutil.rmtree(tandp)
    print("Folder Cleared!")
    os.mkdir(tandp)
else:
    os.mkdir(tandp)
    print("New Folder Created!")

if os.path.isdir(sep):
    shutil.rmtree(sep)
    print("Folder Cleared!")
    os.mkdir(sep)
else:
    os.mkdir(sep)
    print("New Folder Created!")

if os.path.isdir(comb):
    shutil.rmtree(comb)
    print("Folder Cleared!")
    os.mkdir(comb)
else:
    os.mkdir(comb)
    print("New Folder Created!")

# read the picture, create a blank picture and combine them on it
path = "static/Whole"
path_dest = "static/images"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
im = Image.open(path + "/" + onlyfiles[0])
# get the size of the image
imarray = np.array(im)
height, width, channel = imarray.shape
print(height, width)
inteh = int(height/520)
intew = int(width/520)
imarray = imarray[int((height-inteh*520)/2):int((height-inteh*520)/2)+1+inteh*520,int((width-intew*520)/2):int((width-intew*520)/2)+1+intew*520, :]
# blank array
tol = np.zeros((inteh*500, intew*500))
print(tol.shape)
for i in range(inteh):
    for j in range(intew):
        print(i, j)
        if (i == 0 and j != 0):
            imneed = imarray[0:520, 500*j-10:500*(j+1) +10, :]
        elif (i != 0 and  j == 0):
            imneed = imarray[500*i-10:500*(i+1)+10, 0:520, :]
        elif (i == 0 and j == 0):
            imneed = imarray[0:520, 0:520, :]
        else:
            imneed = imarray[500*i-10:500*(i+1)+10,500*j-10:500*(j+1) +10,:]
        
        final = Image.fromarray(imneed)
        final.save(path_dest + "/" + onlyfiles[0])

        # prepare dataset
        prepare_datasets_DRIVE.Prepare()
        print("$$$$$$$Finish Preparation$$$$$$$")

        # 1. do the up part prediction

        predicting.Prediction()
        print("$$$$$$$Finish Prediction of the Upper Part$$$$$$$")

        target = "combination/"
        path_or = "static/I/"+"all_originals.jpg"
        path_pred = "static/I/"+"all_predictions.jpg"
        ori = Image.open(path_or)
        pred = Image.open(path_pred)
        ori_arr = np.asarray(ori)
        pred_arr = np.asarray(pred)
        time.sleep(1)

        if(i == 0 and j == 0):
            tol[i*500:500+i*500,j*500:j*500+500] = pred_arr[0:500, 0:500]
        elif(i == 0 and j != 0):
            tol[i*500:500+i*500,j*500:j*500+500] = pred_arr[0:500, 9:509]
        elif(i != 0 and j == 0):
            tol[i*500:500+i*500,j*500:j*500+500] = pred_arr[9:509, 0:500]
        else:
            tol[i*500:500+i*500,j*500:j*500+500] = pred_arr[9:509, 9:509]


tol = Image.fromarray(tol.astype(np.uint8))
tol.save("combination/tol_predictions.jpg")
