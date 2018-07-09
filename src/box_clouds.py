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

def binarize(img):
    '''
    Binarize an image via its greyscaled mean value.
    '''
    pic = skimage.color.rgb2grey(img)
    image_mean = np.mean(pic) 
    #print(image_mean,np.max(pic),pic.shape)
    if (image_mean > .4):
        image_mean = 0.03
    return pic > image_mean
    

def cut_black_out(img, thresh): #wir nehmen 0.2 als treshold?
    '''
    Cut out the bottom of the image which is presumably blacker than the top.
    '''
    box_not_cloud= True
    dont_crop = False
    box_height = np.shape(img)[0] * 0.03 #3% des Bildes werden als Box getestet
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

def binarized_crop(img, thresh):
    '''
    Returns the shape an image has to be cropped to cut out non-sky objects.
    Use 0.2 as a treshold.
    '''
    binarized_img = binarize(img).astype(np.float)
    cropped_img = cut_black_out(binarized_img, 0.2)
    return cropped_img.shape

''' debug stuff

plt.close("all")

img = glob.glob("../temp/classic/cirriform/cirriform29.jpg")
for x in img:
    pic = imread(x)
'''