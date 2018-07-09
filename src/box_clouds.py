#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:44:21 2018

@author: ali
"""

import numpy as np
from skimage.io import imread, imsave, imshow, imshow_collection
from scipy import misc
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import transform
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt
import glob
import pdb
import skimage.color
from matplotlib.colors import hsv_to_rgb


img = glob.glob("../temp/classic/cirriform/cirriform18.jpg")
for x in img:
    pic = imread(x)


def hsv(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x,y,0]+ 20
        
def binarize(img):
    '''
    Binarize an image via its greyscaled mean value.
    '''
    pic = skimage.color.rgb2grey(img)
    image_mean = np.mean(pic)     
    img_hsv = skimage.color.rgb2hsv(img)
    bin_pic = np.zeros_like(pic)
    #wozu hatten wir das?
   # if (image_mean > .4):
    #    image_mean = 0.03
    
    #gehen jetzt über alle farbkanäle und schauen uns im hsv den Hue an
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x][y][0] > image_mean + 30 and img[x][y][1] > image_mean + 30 and img[x][y][2] > image_mean) and (img_hsv[x][y][0] > 0.4 or img_hsv[x][y][0] < 0.16):
                bin_pic[x][y] = 1
    return bin_pic

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
    cropped_img = cut_black_out(binarized_img, thresh)
    return cropped_img.shape

''' debug stuff

plt.close("all")

img = glob.glob("../temp/classic/cirriform/cirriform29.jpg")
for x in img:
    pic = imread(x)
'''

x = binarized_crop(pic, 0.3)

collection = [pic, binarize(pic), pic[0:x[0],0:x[1],:]]
for e in collection:
    print(e.shape)
plt.close()
imshow_collection(collection)