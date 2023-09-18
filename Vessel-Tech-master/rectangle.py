from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

path = "static/Whole"
path_dest = "static/images"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
im = Image.open(path + "/" + onlyfiles[0])
img = np.array(im)
h, w, c = img.shape
print(img.shape)
nh = h + 500
nw = w + 500

zeros = np.zeros((nh, nw, c))
print(zeros[255:(h+255), 255:(w+255),:].shape)
zeros[255:(h+255), 255:(w+255),:] = img

im = Image.fromarray(zeros.astype(np.uint8))

im.save(path+'/'+ onlyfiles[0])
