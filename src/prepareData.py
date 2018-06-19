# -*- coding: utf-8 -*-

from scipy import misc
import glob

#TODO chosing label numbers globally for this project
#TODO save only descriptor data (missing metric functions for that)
#TODO save and load funktions for compressed *.npz instead of prepare_data()

#make sure to have run loadImgs.py atleast once for data!

trainData = [] #das ist gegeben
valiData = []  #das ist gesucht

trainLabels = []
valiLabels = []

stratiform_paths = glob.glob("../temp/stratiform/*")
stratiform_pics = []
stratiform_label = 1

cirriform_paths = glob.glob("../temp/cirriform/*")
cirriform_pics = []
cirriform_label = 2

stratocumuliform_paths = glob.glob("../temp/stratocumuliform/*")
stratocumuliform_pics = []
stratocumuliform_label = 3

cumuliform_paths = glob.glob("../temp/cumuliform/*")
cumuliform_pics = []
cumuliform_label = 4



def prepare_data():
    '''
    Builds two arrays with images - one for the training data, one for the 
    validation data - out of our 4 big cloud categories and builds the 
    correspondent label arrays for each. 
    '''  
    
	#stratiform
    for imgpath in stratiform_paths:
        stratiform_pics.append(misc.imread(imgpath))

    numberValiImages = round(len(stratiform_pics) * 0.8, 0)
    numberTrainImages = len(stratiform_pics) - numberValiImages
    print (numberValiImages,numberTrainImages,len(stratiform_pics))
    
    for x in range(int(numberValiImages)):
        trainData.append(stratiform_pics[x][:][:][:])
        trainLabels.append(stratiform_label)
		
    for x in range(int(numberValiImages), int(len(stratiform_pics))):
        valiData.append(stratiform_pics[x][:][:][:])
        valiLabels.append(stratiform_label)
        
    #cirriform
    for imgpath in cirriform_paths:
        cirriform_pics.append(misc.imread(imgpath))

    numberValiImages = round(len(cirriform_pics) * 0.8, 0)
    numberTrainImages = len(cirriform_pics) - numberValiImages
    print (numberValiImages,numberTrainImages,len(cirriform_pics))
    
    for x in range(int(numberValiImages)):
        trainData.append(cirriform_pics[x][:][:][:])
        trainLabels.append(cirriform_label)
		
    for x in range(int(numberValiImages), int(len(stratiform_pics))):
        valiData.append(stratiform_pics[x][:][:][:])
        valiLabels.append(stratiform_label)
    
    #stratocumuliform
    for imgpath in stratocumuliform_paths:
        stratocumuliform_pics.append(misc.imread(imgpath))

    numberValiImages = round(len(stratocumuliform_pics) * 0.8, 0)
    numberTrainImages = len(stratocumuliform_pics) - numberValiImages
    print (numberValiImages,numberTrainImages,len(stratocumuliform_pics))
    
    for x in range(int(numberValiImages)):
        trainData.append(stratocumuliform_pics[x][:][:][:])
        trainLabels.append(stratocumuliform_label)
		
    for x in range(int(numberValiImages), int(len(stratocumuliform_pics))):
        valiData.append(stratocumuliform_pics[x][:][:][:])
        valiLabels.append(stratocumuliform_label)
        
    #cumuliform
    for imgpath in cumuliform_paths:
        cumuliform_pics.append(misc.imread(imgpath))

    numberValiImages = round(len(cumuliform_pics) * 0.8, 0)
    numberTrainImages = len(cumuliform_pics) - numberValiImages
    print (numberValiImages,numberTrainImages,len(cumuliform_pics))
    
    for x in range(int(numberValiImages)):
        trainData.append(cumuliform_pics[x][:][:][:])
        trainLabels.append(cumuliform_label)
		
    for x in range(int(numberValiImages), int(len(cumuliform_pics))):
        valiData.append(cumuliform_pics[x][:][:][:])
        valiLabels.append(cumuliform_label)
    

        
		
	