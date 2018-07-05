#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:44:21 2018

@author: ali
"""

import numpy as np
from skimage.io import imread, imsave, imshow
from scipy import misc
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import transform
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt
import glob
import pdb
import skimage.color

''' Aufgabe 2 '''

plt.close("all")

img = glob.glob("../temp/cirriform/cirriform29.jpg")
for x in img:
    pic = imread(x)

# 2.1
# Binarize image with every value mean within the threshold being in the foreground
def binarize(img):
    pic = skimage.color.rgb2grey(img)
    image_mean = np.mean(pic) 
    print(image_mean,np.max(pic),pic.shape)
    if (image_mean > .4):
        image_mean = 0.03
    return pic > image_mean
    

def cut_black_out(img, thresh): #wir nehmen 0.2 als treshold?
    dont_crop = False
    box_height = np.shape(img)[0] * 0.03
    bool_= True
    y = np.shape(img)[0]
    while(bool_):
        image_box = img[int(y) - int(box_height) : int(y),:]
        box_mean = np.mean(image_box)
        y -= box_height/2
        print (y,box_height)
        if y < 0:
            bool_ = False
            dont_crop = True
        if box_mean > thresh:
            bool_ = False
    if dont_crop:
        return img
    return img[0:int(y),:]
        
def binarized_crop(img,thresh):
    pic1 = binarize(img).astype(np.float)
    pic2 = cut_black_out(pic1, 0.2)
    return pic2.shape #??