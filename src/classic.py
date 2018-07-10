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

nbins=3

    
def getMerkmal(img, Merkmal, nbins): #Gibt das angegebene Merkmal für ein Bild zurück.
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
    if Merkmal == 'histogramG' :
        return np.histogram(img, bins = nbins)[0]

def create_cropped_images():
    for img in trainData:
        image_shape = box.binarized_crop(img, 0.2)
        #print("image_shape:" ,image_shape)
        #print("img_shape:" , img.shape)
        image = img[0:image_shape[0],0:image_shape[1],:]
        #print("image:",image.shape)
        resized = skimage.transform.resize(image, (500,500))
        #print("resized:",resized.shape)
        bin_trainData.append(resized)
    print("done cropping traindata")
    for img in valiData:
        image_shape = box.binarized_crop(img, 0.2)
        #print("image_shape:" ,image_shape)
        #print("img_shape:" , img.shape)
        image = img[0:image_shape[0],0:image_shape[1],:]
        #print("image:",image.shape)
        resized = skimage.transform.resize(image, (500,500))
        #print("resized:",resized.shape)
        bin_valiData.append(resized)
    print("done cropping validata")
    
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
        
def euclidean_hist_dist(hist1, hist2):
    """Computes the Euclidean distance between two histograms."""

    delta_hist_squared = (hist2 - hist1) ** 2
    return np.sqrt(delta_hist_squared.sum())

#MAIN PROGRAMM
if __name__ == '__main__':
    #speichert Daten in trainData, trainLabels, valiData, valiLabels
    prepareData.prepare_data() 
    create_cropped_images()
    #the copImageArray doesnt work yet :c
    #bin_trainData = util.cropImageArray(trainData)
    #bin_valiData = util.cropImageArray(valiData)

    trainData = bin_trainData
    valiData = bin_valiData
    edges()

    #berechnet Merkmale in den zugehörigen Arrays
    for img in trainData: 
        trMerkmale.append(getMerkmal(img, 'histogram3D', nbins))
        trMerkmale2.append(getMerkmal(img, 'std', 0))
        trMerkmale3.append(getMerkmal(img, 'mean', 0))
    
    for img in valiData:
        vaMerkmale.append(getMerkmal(img, 'histogram3D', nbins))
        vaMerkmale2.append(getMerkmal(img, 'std', 0))
        vaMerkmale3.append(getMerkmal(img, 'mean', 0))
       
    for img in edgy_trainData:
        trMerkmale4.append(getMerkmal(img, 'histogramG', 16))
        
    for img in edgy_valiData:
        vaMerkmale4.append(getMerkmal(img, 'histogramG', 16))
        
    #Gewichte
    W0=0
    W1=1000000
    W2=1000000
    W3=6 #Erhöht man W3, geht die Genauigkeit gegen 39%, verringert man W3 ist es, als wäre es 0.
    #So wie es jetzt gerade ist, verschlechtert W3 leicht das Ergebnis
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
    print (accuracy)