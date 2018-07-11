import prepareData
from prepareData import trainData, trainLabels, valiData, valiLabels
import numpy as np
import box_clouds as box
import util
from scipy.ndimage.filters import gaussian_filter
from skimage import filters
from skimage import color
from skimage import io

from skimage.io import imread, imsave, imshow
import skimage


'''      KONFIGURATION      '''
'''BIITE UNBGEDINGT ANPASSEN'''
#Use saved data or create new ones
use_saved_data = False #only works, if you saved data once!
save_data = True

#Number of Bins for histograms
nbins=7

#Weights for the distance calculation
W0=1 #1D-Histogramm
W1=0 #std
W2=0 #mean
W3=500 #edge_count



""" 
trainData = [] #das ist gegeben
valiData = []  #das ist gesucht

trainLabels = []
valiLabels = []
"""

bin_trainData = []
bin_valiData = []
edgy_trainData = []
edgy_valiData = []

trMerkmale = [] 
vaMerkmale = []
trMerkmale2 = [] 
vaMerkmale2 = []
trMerkmale3 = []
vaMerkmale3 = []
trMerkmale4 = []
vaMerkmale4 = []


    
def getMerkmal(img, Merkmal, nbins): #Gibt das angegebene Merkmal für ein Bild zurück.
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
    print ("stratiform:        "+ str(O1_1) +"|"+ str(O1_2) +"|"+ str(O1_3) +"|"+ str(O1_4))
    print ("cirriform:         "+ str(O2_1) +"|"+ str(O2_2) +"|"+ str(O2_3) +"|"+ str(O2_4))
    print ("stratocummuliform: "+ str(O3_1) +"|"+ str(O3_2) +"|"+ str(O3_3) +"|"+ str(O3_4))
    print ("cummuliform:       "+ str(O4_1) +"|"+ str(O4_2) +"|"+ str(O4_3) +"|"+ str(O4_4))
    print ("Anzahl klassifizierter Bilder: ",gesamt)
        

        
def euclidean_hist_dist(hist1, hist2):
    """Computes the Euclidean distance between two histograms."""

    delta_hist_squared = (hist2 - hist1) ** 2
    return np.sqrt(delta_hist_squared.sum())

#MAIN PROGRAMM
if __name__ == '__main__':
    #speichert Daten in trainData, trainLabels, valiData, valiLabels
    prepareData.prepare_data() 
    if use_saved_data == False:
        
        
        
        bin_trainData = util.cropImageArray(trainData)
        print("done cropping TrainImages")
        bin_valiData = util.cropImageArray(valiData)
        print("done cropping ValiImages")
        
        trainData = bin_trainData
        valiData = bin_valiData
        edges()
    
        #berechnet Merkmale in den zugehörigen Arrays
        for img in trainData: 
            trMerkmale.append(getMerkmal(img, 'histogram1D', nbins))
            trMerkmale2.append(getMerkmal(img, 'std', 0))
            trMerkmale3.append(getMerkmal(img, 'mean', 0))   
        for img in edgy_trainData:
            trMerkmale4.append(getMerkmal(img, 'edge_count', 0))
        print("done calculating TrainFeatures")
        
        for img in valiData:
            vaMerkmale.append(getMerkmal(img, 'histogram1D', nbins))
            vaMerkmale2.append(getMerkmal(img, 'std', 0))
            vaMerkmale3.append(getMerkmal(img, 'mean', 0))       
        for img in edgy_valiData:
            vaMerkmale4.append(getMerkmal(img, 'edge_count', 0))
        print("done calculating ValiFeatures")
    
    #Gewichte
    W0=0.8 #0
    W1=0.35#1000000
    W2=0.45#1000000
    W3=0.8#6 #Erhöht man W3, geht die Genauigkeit gegen 39%, verringert man W3 ist es, als wäre es 0.
    #So wie es jetzt gerade ist, verschlechtert W3 leicht das Ergebnis

    if save_data == True:
        np.save("../temp/classic/trMerkmale", trMerkmale)
        np.save("../temp/classic/trMerkmale2", trMerkmale2)
        np.save("../temp/classic/trMerkmale3", trMerkmale3)
        np.save("../temp/classic/trMerkmale4", trMerkmale4)
        
        np.save("../temp/classic/vaMerkmale", vaMerkmale)
        np.save("../temp/classic/vaMerkmale2", vaMerkmale2)
        np.save("../temp/classic/vaMerkmale3", vaMerkmale3)
        np.save("../temp/classic/vaMerkmale4", vaMerkmale4)

    if use_saved_data == True:
        print("skipping calculating features and instead using saved ones")
        trMerkmale = np.load("../temp/classic/trMerkmale.npy")
        trMerkmale2 = np.load("../temp/classic/trMerkmale2.npy")
        trMerkmale3 = np.load("../temp/classic/trMerkmale3.npy")
        trMerkmale4 = np.load("../temp/classic/trMerkmale4.npy")
        
        vaMerkmale = np.load("../temp/classic/vaMerkmale.npy")
        vaMerkmale2 = np.load("../temp/classic/vaMerkmale2.npy")
        vaMerkmale3 = np.load("../temp/classic/vaMerkmale3.npy")
        vaMerkmale4 = np.load("../temp/classic/vaMerkmale4.npy")
        
    result = []
    #Distanzvergleich
    for vaM,vaM2,vaM3,vaM4 in zip(vaMerkmale,vaMerkmale2,vaMerkmale3,vaMerkmale4):
        distances = [] 
        for trM,trM2,trM3,trM4 in zip(trMerkmale,trMerkmale2,trMerkmale3,trMerkmale4): 
            distances.append(W0*np.linalg.norm(vaM-trM)+
                             W1*np.linalg.norm(vaM2-trM2)+
                             W2*np.linalg.norm(vaM3-trM3)+
                             W3*np.linalg.norm(euclidean_hist_dist(vaM4, trM4)))
        result.append(trainLabels[np.argmin(distances)]) 


    #Accuracy Berechnung
    correct = 0.0
    for calculatedLabel, trueLabel in zip(result,valiLabels):
        if calculatedLabel == trueLabel: 
            correct+=1 
        
    accuracy = correct/len(valiLabels)
    print ("Accuracy: ",accuracy)
    
    create_confusion_matrix()
