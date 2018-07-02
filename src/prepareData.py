from scipy import misc
import glob
import logging
import PIL
from PIL import Image

import util


logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


#TODO chosing label numbers globally for this project
#TODO save only descriptor data (missing metric functions for that)
#TODO save and load funktions for compressed *.npz instead of prepare_data()


def save_resized_pictures(height, width):
    for cloud_kind, subkinds in util.cloud_kinds.items():
        if util.ensure_directory_exists("../temp/classic/" + cloud_kind):
            # directory exists already, we assume that the resized pictures
            # are in there
            continue

        logger.info("saving resized %s pictures" % cloud_kind)

        counter = 0

        for subkind in subkinds:
            for element in glob.glob("../data/" + subkind + "/*"):
                img = Image.open(element)
                resized = img.resize((height, width), PIL.Image.ANTIALIAS)
                path = "../temp/classic/%s/%s%s.jpg" % (cloud_kind, cloud_kind, counter)
                resized.save(path)
                counter += 1



trainData = [] #das ist gegeben
valiData = []  #das ist gesucht

trainLabels = []
valiLabels = []

labels = {}
for i, cloud_kind in enumerate(util.cloud_kinds):
    labels[cloud_kind] = i + 1


def prepare_data():
    '''
    Builds two arrays with images - one for the training data, one for the 
    validation data - out of our 4 big cloud categories and builds the 
    correspondent label arrays for each. 
    '''

    logger.info("preparing classic classification data")
    
    for cloud_kind in util.cloud_kinds:
        pics = []
        for imgpath in glob.glob("../temp/classic/" + cloud_kind + "/*"):
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
            
if __name__ == '__main__':
    save_resized_pictures(500, 500)
    prepare_data()
