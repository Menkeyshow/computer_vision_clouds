#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:17:44 2018

@author: ali
"""

import numpy as np
from skimage.io import imread, imshow, imshow_collection
import matplotlib.pyplot as plt
import glob
import skimage.color


img = glob.glob("../temp/stratocumuliform/stratocumuliform122.jpg")
for x in img:
    pic = imread(x)


plt.close("all")

x = binarized_crop(pic, 0.35)

collection = [pic, binarize(pic), pic[0:x[0],0:x[1],:]]
for e in collection:
    print(e.shape)
plt.close()
imshow_collection(collection)



def cut_black_out(img, thresh): #wir nehmen 0.2 als treshold?
    '''
    Cut out the bottom of the image which is presumably blacker than the top.
    '''
    box_not_cloud= True
    dont_crop = False
    box_height = np.shape(img)[0] * 0.1 #3% des Bildes werden als Box getestet
    y = np.shape(img)[0]
    while(box_not_cloud):
        image_box = img[int(y) - int(box_height) : int(y),:]
        box_mean = np.mean(image_box)
        y -= box_height/2
        if y < 0:
            box_not_cloud = False
            dont_crop = True
        if box_mean > thresh:
            box_not_cloud = False
    if dont_crop:
        return img
    return img[0:int(y),:]