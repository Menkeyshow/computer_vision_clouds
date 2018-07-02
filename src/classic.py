import prepareData
from prepareData import trainData, trainLabels, valiData, valiLabels
import numpy as np

""" 
trainData = [] #das ist gegeben
valiData = []  #das ist gesucht

trainLabels = []
valiLabels = []
"""
trMerkmale = [] 
vaMerkmale = []
trMerkmale2 = [] 
vaMerkmale2 = []
trMerkmale3 = []
vaMerkmale3 = []
trMerkmale4 = []
vaMerkmale4 = []

nbins=7

def getMerkmal(img, Merkmal, nbins):
    if Merkmal == 'mean':
        return np.mean(img, axis=(0,1))
    if Merkmal == 'std':
        return np.std(img, axis=(0,1))
    if Merkmal == 'histogram1D':
        rHist = np.histogram(img[:,:,0], bins = nbins, range=(0,256))[0] 
        gHist = np.histogram(img[:,:,1], bins = nbins, range=(0,256))[0]
        bHist = np.histogram(img[:,:,2], bins = nbins, range=(0,256))[0]
        return np.hstack((rHist, gHist, bHist)) #hstack verbindet die Eingabe (hier die einzelnen Historgamme als Array) zu einem Array, indem es die Elemente horizontal stapelt
    if Merkmal == 'histogram3D':
        imgReshaped = img.reshape((img.shape[0]*img.shape[1],3)) #Reshapen, damit jedes Pixek in einer Zeile liegt
        return np.histogramdd(imgReshaped, bins = [nbins,nbins,nbins], range=((0,256),(0,256),(0,256)))[0].flatten()

#TODO: binarisierung einbauen, sonst macht das Histogramm noch nicht so viel Sinn...
if __name__ == '__main__':
    prepareData.prepare_data() #speichert Daten in trainData, trainLabels, valiData, valiLabels
    
    for img in trainData: 
        trMerkmale.append(getMerkmal(img, 'histogram3D', nbins))
        trMerkmale2.append(getMerkmal(img, 'std', 0))
        trMerkmale3.append(getMerkmal(img, 'mean', 0))
    
    for img in valiData:
        vaMerkmale.append(getMerkmal(img, 'histogram3D', nbins))
        vaMerkmale2.append(getMerkmal(img, 'std', 0))
        vaMerkmale3.append(getMerkmal(img, 'mean', 0))
        
    #Gewichte
    W0=1
    W1=50
    W2=50
    result = []
    
    #Distanzvergleich
    for vaM,vaM2,vaM3 in zip(vaMerkmale,vaMerkmale2,vaMerkmale3):
        distances = [] 
        for trM,trM2,trM3 in zip(trMerkmale,trMerkmale2,trMerkmale3): 
            distances.append(W0*np.linalg.norm(vaM-trM)+
                             W1*np.linalg.norm(vaM2-trM2)+
                             W2*np.linalg.norm(vaM3-trM3)) 
        result.append(trainLabels[np.argmin(distances)]) 

    #Accuracy Berechnung
    correct = 0.0
    for calculatedLabel, trueLabel in zip(result,valiLabels):
        if calculatedLabel == trueLabel: 
            correct+=1 
        
    accuracy = correct/len(valiLabels)
    print (accuracy)