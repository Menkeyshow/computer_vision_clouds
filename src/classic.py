import prepareData
from prepareData import trainData, trainLabels, valiData, valiLabels

import numpy as np

import util
import os.path
from scipy.ndimage.filters import gaussian_filter
from skimage import filters
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns

'''      CONFIGURATION      
         PLEASE ADJUST      
Erfolgreiche Gewichtungen:
[['mean', 1],['std', 1500]] --> 40% 
[['mean', 100],['std', 100],["edge_count", 1]] -->48%
[['edge_count', 1500],['histogram3D', 1]] -->49%
'''
#Which features should be used and how should they be weighted?
#available features: mean, std, histogram1D, histogram3D, histogramG, edge_count
#shape: [[feature, weight],[feature2, weight2], ...]
config = [['edge_count', 1500],['histogram3D', 1]]

#following are the arrays needed to save the features
#ATTENTION: they are only accessible via the stringconverter dict
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
    global feats_all_saved, trainData, valiData, trainLabels, valiLabels
    feats_all_saved = True
    for feat in config:#Are all features already saved? Then we don't need to load the data, only the labels
        if not os.path.exists('../temp/classic/tr_'+feat[0]+'.npy'):
            feats_all_saved = False
        if not os.path.exists('../temp/classic/va_'+feat[0]+'.npy'):
            feats_all_saved = False
    if feats_all_saved:
        trainLabels = np.load('../temp/classic/trainLabels.npy')
        valiLabels = np.load('../temp/classic/valiLabels.npy')
        return
    if not os.path.exists('../temp/classic/trainData.npy'):
        #ASSUMPTION: trainData exists implicates that valiData, trainLabels and valiLabels exist
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
        print('done saving Images')
    else:
        trainData = np.load('../temp/classic/trainData.npy')
        valiData = np.load('../temp/classic/valiData.npy')
        trainLabels = np.load('../temp/classic/trainLabels.npy')
        valiLabels = np.load('../temp/classic/valiLabels.npy')
        print('done loading Images')
        
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
        g_img = color.rgb2gray(img)
        f_img = gaussian_filter(g_img, 2) #Wendet den gaussschen Weichzeichner auf das Bild an mit Sigma = 2
        sobel_h = filters.sobel_h(f_img) #Findet die horizontalen Kanten
        sobel_v = filters.sobel_v(f_img) #Findet die vertikalen Kanten
        intensity = np.linalg.norm(np.stack((sobel_h, sobel_v)), axis=0) #Kombiniert h & v und zeigt den absoluten Kantenwert
        sums = np.array([0])
        for zeile in intensity:
            sums = np.append(sums, zeile.sum())
        return sums 
    

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
    print ('stratocumuliform: '+ str(O3_1) +'|'+ str(O3_2) +'|'+ str(O3_3) +'|'+ str(O3_4))
    print ('cumuliform:       '+ str(O4_1) +'|'+ str(O4_2) +'|'+ str(O4_3) +'|'+ str(O4_4))
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
    
    #SCATTERPLOTS! eventuell auskommentieren
    '''
    #config = [['mean', 1],['std', 1500]]
    #following is a scatterplot for mean and std
    x = stringconverter['tr_mean']
    y = stringconverter['tr_std']
    a = np.append(trainLabels,np.append(trainLabels,trainLabels))
    b = np.sort(a)
    c = np.reshape(b, (633, 3))
    plt.scatter(x, y, 5, c)
    plt.xlabel('Mean \n lila:stratiform    blau:cirriform    grün:stratocumuliform    gelb:cumuliform')
    plt.ylabel('Standard Deviation')
    plt.show

    #config = [['edge_count', 1]]
    #following is a box/swarmplot for edge_count
    x = [] 
    for arr in stringconverter['tr_edge_count']:
        x.append(sum(arr))
    y = trainLabels
    sns.swarmplot(y,x)  
    #In dieser und der nächsten Zeile sind boxplot/violinplot/swarmplot austauschbar
    sns.swarmplot(y,x,order=['stratiform','cirriform','stratocumuliform','cumuliform'])
'''
    