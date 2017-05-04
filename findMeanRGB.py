import os
import scipy.misc as im
import numpy as np

files = os.listdir("dataset")

r = 0
g = 0
b = 0

for f in files:
    image = im.imread("dataset/"+f,mode='RGB')
    r += image[:,:,0]
    g += image[:,:,1]
    b += image[:,:,2]

print (image.shape)
r = np.sum(r)
g = np.sum(g)
b = np.sum(b)

print(r,g,b)

r = r/(256*256)
g = g/(256*256)
b = b/(256*256)
print(r,g,b)