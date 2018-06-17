#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:55:11 2018

@author: ali
"""
import os
import glob
import PIL
from PIL import Image
        
        
altocumulus_path = "../data/altocumulus/*"
altocumulus_pics = glob.glob(altocumulus_path)

altostratus_path = "../data/altostratus/*"
altostratus_pics = glob.glob(altostratus_path)

cirrocumulus_path = "../data/cirrocumulus/*"
cirrocumulus_pics = glob.glob(cirrocumulus_path)

cirrostratus_path = "../data/cirrostratus/*"
cirrostratus_pics = glob.glob(cirrostratus_path)

cirrus_path = "../data/cirrus/*"
cirrus_pics = glob.glob(cirrus_path)

cumulonimbus_path = "../data/cumulonimbus/*"
cumulonimbus_pics = glob.glob(cumulonimbus_path)

cumulus_path = "../data/cumulus/*"
cumulus_pics = glob.glob(cumulus_path)

nimbostratus_path = "../data/nimbostratus/*"
nimbostratus_pics = glob.glob(nimbostratus_path)

stratocumulus_path = "../data/stratocumulus/*"
stratocumulus_pics = glob.glob(stratocumulus_path)

stratus_path = "../data/stratus/*"
stratus_pics = glob.glob(stratus_path)

stratiform = []
stratiform = cirrostratus_pics + altostratus_pics + nimbostratus_pics + stratus_pics

cirriform = []
cirriform = cirrus_pics

stratocumuliform = []
stratocumuliform = cirrocumulus_pics + altocumulus_pics + stratocumulus_pics

cumuliform = []
cumuliform = cumulus_pics + cumulonimbus_pics


def check_directories():
    '''Checks, if directories exist, and if not creates them'''    
    directories = ["../temp/stratiform", "../temp/cirriform",
                   "../temp/stratocumuliform", "../temp/cumuliform" ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
def resize_pics(array,height, widt, foldername):
    '''function to resize pics to a size''' 
    check_directories()
    counter = 0
    
    
    
    for element in array:
        img =  Image.open(element)    
        resized = img.resize((height, widt), PIL.Image.ANTIALIAS)
        path = "../temp/%s/%s%s.jpg" % (foldername,foldername, counter)
        resized.save(path)
        counter += 1
        

resize_pics(stratiform, 500, 500, "stratiform")
resize_pics(cirriform, 500, 500, "cirriform")
resize_pics(stratocumuliform, 500, 500, "stratocumuliform")
resize_pics(cumuliform, 500, 500, "cumuliform")



