#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:42:29 2018

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



img = glob.glob("../temp/classic/cirriform/cirriform24.jpg")
for x in img:
    pic = imread(x)
    
    
gray_pic = skimage.color.rgb2grey(pic)    
plt.close()


# neighbours of a pixel (clockwise)
def get_neighbours(x,y):
    return [[x-1,y-1],[x-1,y],[x-1,y+1],
            [x,y+1],[x+1,y+1],
            [x+1,y],[x+1,y-1],[x,y-1]]

# region growing
def region_growing(img, r_tolerance, g_tolerance, b_tolerance):
    plt.close()
    #img = skimage.color.rgb2grey(img)
    
    # dimensions
    max_x = img.shape[0]
    max_y = img.shape[1]
    
    output = np.ones((max_x, max_y))
    output[0,0] = 0
    
    to_do = [[0,0]]
    visited = []
    
    # process as long as there is something to process
    while to_do != []:
        current_i = to_do[0]
        current_val = img[current_i[0],current_i[1]]
        visited.append(current_i)
        
        for n in get_neighbours(current_i[0], current_i[1]):
            #neighbour has a valid value
            if ((n not in visited and 0 <= n[0] < max_x) and (0 <= n[1] < max_y) ):
                    n_val = img[n[0],n[1]] 
                    if (abs(int(current_val[0]) - int(n_val[0])) <= r_tolerance)\
                    and (abs(int(current_val[1]) - int(n_val[1])) <= g_tolerance)\
                    and (abs(int(current_val[2]) - int(n_val[2])) <= b_tolerance):
                   # if (abs(float(current_val) - float(n_val)) <= tolerance):
                        output[n[0],n[1]] = 0
                        if n not in to_do:
                            to_do.append(n)
                    
        to_do.pop(0)
    
    return output

imshow_collection([region_growing(pic, 10,10,30), pic])