# -*- coding: utf-8 -*-

from scipy import misc
import glob

import util

#TODO chosing label numbers globally for this project
#TODO save only descriptor data (missing metric functions for that)
#TODO save and load funktions for compressed *.npz instead of prepare_data()

#make sure to have run loadImgs.py atleast once for data!

trainData = [] #das ist gegeben
valiData = []  #das ist gesucht

trainLabels = []
valiLabels = []

labels = {}
for i, cloud_kind in util.cloud_kinds:
    labels[cloud_kind] = i + 1


def prepare_data():
    '''
    Builds two arrays with images - one for the training data, one for the 
    validation data - out of our 4 big cloud categories and builds the 
    correspondent label arrays for each. 
    '''  
    
    for cloud_kind in util.cloud_kinds:
        pics = []
        for imgpath in glob.globg("../temp/classic/" + cloud_kind + "/*"):
            pics.append(misc.imread(imgpath))

        numberValiImages = round(len(pics) * 0.8, 0)
        numberTrainImages = len(pics) - numberValiImages
        print(numberValiImages, numberTrainImages, len(pics))

        for x in range(int(numberValiImages)):
            trainData.append(pics[x][:][:][:])
            trainLabels.append(labels[cloud_kind])

        for x in range(int(numberValiImages), int(len(pics))):
            valiData.append(pics[x][:][:][:])
            valiLabels.append(labels[cloud_kind])
