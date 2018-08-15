import prepareData
from prepareData import trainData, trainLabels, valiData, valiLabels

import numpy as np

import box_clouds as box
import util
import os.path
from scipy.ndimage.filters import gaussian_filter
from skimage import filters
from skimage import color

'''      CONFIGURATION      '''
'''      PLEASE ADJUST      '''
#Which features should be used and how should they be weighted?
#available features: mean, std, histogram1D, histogram3D, histogramG, edge_count
#shape: [[feature, weight],[feature2, weight2], ...]
'''edge_count does not work yet'''
config = [['mean',1],
          ['std',1]]

#following are the arrays needed to save the data
bin_trainData = []
bin_valiData = []
edgy_trainData = []
edgy_valiData = []

tr_mean = [] 
va_mean = []
tr_std = []
va_std = []
tr_histogram1D = [] 
va_histogram1D = []
tr_histogram3D = []
va_histogram3D = []
tr_histogramG = []
va_histogramG = []
tr_edge_count = []
va_edge_count = []

#Number of Bins for histograms
nbins=7

#Dictionary to convert strings to variable names
stringconverter = {'tr_mean':tr_mean,'va_mean':va_mean, 
                   'tr_std':tr_std,'va_std':va_std, 
                   'tr_histogram1D':tr_histogram1D,'va_histogram1D':va_histogram1D,
                   'tr_histogram3D':tr_histogram3D,'va_histogram3D':va_histogram3D,
                   'tr_histogramG':tr_histogramG,'va_histogramG':va_histogramG,
                   'tr_edge_count':tr_edge_count,'va_edge_count':va_edge_count}


def load_data():
    if not os.path.exists('../temp/classic/trainData.npy'):
        #ASSUMPTION: trainData exists implicates that valiData, trainLabels and valiLabels exist
        global trainData, valiData, trainLabels, valiLabels
        prepareData.prepare_data() 
        bin_trainData = util.cropImageArray(trainData)
        print('done cropping TrainImages')
        bin_valiData = util.cropImageArray(valiData)
        print('done cropping ValiImages')
        
        trainData = bin_trainData
        valiData = bin_valiData
        np.save('../temp/classic/trainData', trainData)
        np.save('../temp/classic/valiData', valiData)
        np.save('../temp/classic/trainLabels', trainLabels)
        np.save('../temp/classic/valiLabels', valiLabels)
    else:
        trainData = np.load('../temp/classic/trainData.npy')
        valiData = np.load('../temp/classic/valiData.npy')
        trainLabels = np.load('../temp/classic/trainLabels.npy')
        valiLabels = np.load('../temp/classic/valiLabels.npy')
        
        
#Calculates all features that haven't been saved yet
#Want something recalculated? Just delete the file
def calculateFeatures():
    
    for feat in config:
        if os.path.exists('../temp/classic/tr_'+feat[0]+'.npy'):
            stringconverter['tr_'+feat[0]] = np.load('../temp/classic/tr_'+feat[0]+'.npy')
        else:    
            for img in trainData:
                stringconverter['tr_'+feat[0]].append(getFeature(img, feat[0], nbins))
            np.save('../temp/classic/tr_'+feat[0]+'.npy',stringconverter['tr_'+feat[0]])
            print('saved new tr_'+feat[0])
        if os.path.exists('../temp/classic/va_'+feat[0]+'.npy'):
            stringconverter['va_'+feat[0]] = np.load('../temp/classic/va_'+feat[0]+'.npy')
        else:    
            for img in valiData:
                stringconverter['va_'+feat[0]].append(getFeature(img, feat[0], nbins))
            np.save('../temp/classic/va_'+feat[0]+'.npy',stringconverter['va_'+feat[0]])
            print('saved new va_'+feat[0])


def getFeature(img, Merkmal, nbins): #returns the given feature for a picture
    if Merkmal == 'mean':
        return np.mean(img, axis=(0,1))
    if Merkmal == 'std':
        return np.std(img, axis=(0,1))
    if Merkmal == 'histogram1D':
        rHist = np.histogram(img[:,:,0], bins = nbins, range=(0,1))[0] 
        gHist = np.histogram(img[:,:,1], bins = nbins, range=(0,1))[0]
        bHist = np.histogram(img[:,:,2], bins = nbins, range=(0,1))[0]
        return np.hstack((rHist, gHist, bHist)) #hstack verbindet die Eingabe (hier die einzelnen Historgamme als Array) zu einem Array, indem es die Elemente horizontal stapelt
    if Merkmal == 'histogram3D':
        imgReshaped = img.reshape((img.shape[0]*img.shape[1],3)) #Reshapen, damit jedes Pixek in einer Zeile liegt
        return np.histogramdd(imgReshaped, bins = [nbins,nbins,nbins], range=((0,1),(0,1),(0,1)))[0].flatten()
    if Merkmal == 'histogramG' :
        return np.histogram(img, bins = nbins)[0]
    if Merkmal == 'edge_count' :
        sums = np.array([0])
        for zeile in img:
            sums = np.append(sums, zeile.sum())
        return sums 
        

def edges(): #Findet die Kanten
    for img in trainData:
        g_img = color.rgb2gray(img)
        f_img = gaussian_filter(g_img, 2) #Wendet den gaussschen Weichzeichner auf das Bild an mit Sigma = 2
        sobel_h = filters.sobel_h(f_img) #Findet die horizontalen Kanten
        sobel_v = filters.sobel_v(f_img) #Findet die vertikalen Kanten
        intensity = np.linalg.norm(np.stack((sobel_h, sobel_v)), axis=0) #Kombiniert h & v und zeigt den absoluten Kantenwert
        edgy_trainData.append(intensity)
    for img in valiData:
        g_img = color.rgb2gray(img)
        f_img = gaussian_filter(g_img, 2)
        sobel_h = filters.sobel_h(f_img)
        sobel_v = filters.sobel_v(f_img)
        intensity = np.linalg.norm(np.stack((sobel_h, sobel_v)), axis=0)
        edgy_valiData.append(intensity)   
    

def create_confusion_matrix():
    '''
    Creates a confusion matrix, after caclulating train and vali data
    '''
    #first number = calculated, second number = trueLabel
    #1 = stratiform
    #2 = cirriform
    #3 = stratocumuliform
    #4 = cumuliform
    O1_1 = 0
    O1_2 = 0
    O1_3 = 0
    O1_4 = 0
    
    O2_1 = 0
    O2_2 = 0
    O2_3 = 0
    O2_4 = 0
    
    O3_1 = 0
    O3_2 = 0
    O3_3 = 0
    O3_4 = 0
    
    O4_1 = 0
    O4_2 = 0
    O4_3 = 0
    O4_4 = 0
    
    for calculatedLabel, trueLabel in zip(result,valiLabels):
        if calculatedLabel == 1 and trueLabel == 1:
            O1_1 += 1
        if calculatedLabel == 1 and trueLabel == 2:
            O1_2 += 1
        if calculatedLabel == 1 and trueLabel == 3:
            O1_3 += 1
        if calculatedLabel == 1 and trueLabel == 4:
            O1_4 += 1
            
        if calculatedLabel == 2 and trueLabel == 1:
            O2_1 += 1
        if calculatedLabel == 2 and trueLabel == 2:
            O2_2 += 1
        if calculatedLabel == 2 and trueLabel == 3:
            O2_3 += 1
        if calculatedLabel == 2 and trueLabel == 4:
            O2_4 += 1
            
        if calculatedLabel == 3 and trueLabel == 1:
            O3_1 += 1
        if calculatedLabel == 3 and trueLabel == 2:
            O3_2 += 1
        if calculatedLabel == 3 and trueLabel == 3:
            O3_3 += 1
        if calculatedLabel == 3 and trueLabel == 4:
            O3_4 += 1
            
        if calculatedLabel == 4 and trueLabel == 1:
            O4_1 += 1
        if calculatedLabel == 4 and trueLabel == 2:
            O4_2 += 1
        if calculatedLabel == 4 and trueLabel == 3:
            O4_3 += 1
        if calculatedLabel == 4 and trueLabel == 4:
            O4_4 += 1
            
    gesamt = O1_1 + O1_2 + O1_3 + O1_4 + O2_1 + O2_2 + O2_3 + O2_4 + O3_1 + O3_2 + O3_3 + O3_4 + O4_1 + O4_2 + O4_3 + O4_4
    print ('stratiform:        '+ str(O1_1) +'|'+ str(O1_2) +'|'+ str(O1_3) +'|'+ str(O1_4))
    print ('cirriform:         '+ str(O2_1) +'|'+ str(O2_2) +'|'+ str(O2_3) +'|'+ str(O2_4))
    print ('stratocummuliform: '+ str(O3_1) +'|'+ str(O3_2) +'|'+ str(O3_3) +'|'+ str(O3_4))
    print ('cummuliform:       '+ str(O4_1) +'|'+ str(O4_2) +'|'+ str(O4_3) +'|'+ str(O4_4))
    print ('Anzahl klassifizierter Bilder: ',gesamt)


#MAIN PROGRAMM
if __name__ == '__main__':
    load_data()
    calculateFeatures()
    result = []
    #Distanzvergleich
    for x in range(len(valiLabels)):
        distances = []
        for y in range(len(trainLabels)):
            dist = 0
            for feat in config:
                dist += np.linalg.norm(stringconverter['va_'+feat[0]][x] 
                        -stringconverter['tr_'+feat[0]][y]) * feat[1]    #(val-train)*weight
            distances.append(dist) 
        result.append(trainLabels[np.argmin(distances)])

    #Accuracy Berechnung
    correct = 0.0
    for calculatedLabel, trueLabel in zip(result,valiLabels):
        if calculatedLabel == trueLabel: 
            correct+=1 
        
    accuracy = correct/len(valiLabels)
    print ('Accuracy: ',accuracy)
    
    create_confusion_matrix()
