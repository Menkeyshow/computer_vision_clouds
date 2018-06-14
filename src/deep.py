from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil


def move_data_into_categorized_directories():
    dirname = "./temp/categorized/"
    if os.path.exists(dirname):
        return dirname

    os.makedirs(dirname)

    cloud_kinds = {"stratiform": ["cirrostratus", "altostratus", "nimbostratus", "stratus"],
                   "cirriform": ["cirrus"],
                   "stratocumuliform": ["cirrocumulus", "altocumulus", "stratocumulus"],
                   "cumuliform": ["cumulus", "cumulonimbus"]}

    for label, sublabels in cloud_kinds.items():
        labeldirname = dirname + label + "/"
        os.makedirs(labeldirname)
        for cloud_type in sublabels:
            sublabeldirname = "./data/" + cloud_type + "/"
            for filename in os.listdir(sublabeldirname):
                shutil.copy(sublabeldirname + filename, labeldirname)
    
    return dirname


extractor = Xception(include_top=False,
                     weights='imagenet',
                     input_shape=(256, 256, 3),
                     pooling='max')

image_dir = move_data_into_categorized_directories()

# mehr data augmentation optionen:
    # rotation_range
    # width_shift_range
    # height_shift_range
    # shear_range
    # zoom_range
    # vertical_flip
train_datagen = ImageDataGenerator(
        fill_mode='nearest',
        horizontal_flip=True,
        rescale=1./255,
        validation_split=0.2)

# TODO: seed...
train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(256, 256),
        batch_size=32,
        subset="training")

# TODO: feature extraction
# TODO: creating model & training it
# TODO: testing performance

# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
