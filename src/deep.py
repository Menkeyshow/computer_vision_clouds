from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks, layers, models, optimizers
import logging
import numpy as np
import os
import shutil



logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

logger.info("logger says hi!")


def move_data_into_categorized_directories():
    dirname = "./temp/categorized/"
    if os.path.exists(dirname):
        logger.info("found categorized images in " + dirname + ", proceeding")
        return dirname

    logger.info("couldn't find categorized images in {}, copying images now"
                .format(dirname))

    os.makedirs(dirname)

    cloud_kinds = {
        "stratiform":
            ["cirrostratus", "altostratus", "nimbostratus", "stratus"],
        "cirriform": ["cirrus"],
        "stratocumuliform": ["cirrocumulus", "altocumulus", "stratocumulus"],
        "cumuliform": ["cumulus", "cumulonimbus"]
    }

    for label, sublabels in cloud_kinds.items():
        logger.debug("processing {}".format(label))
        labeldirname = dirname + label + "/"
        os.makedirs(labeldirname)
        for cloud_type in sublabels:
            sublabeldirname = "../data/" + cloud_type + "/"
            for filename in os.listdir(sublabeldirname):
                shutil.copy(sublabeldirname + filename, labeldirname)

    logger.info("finished copying images")

    return dirname


extractor = Xception(include_top=False,
                     weights='imagenet',
                     input_shape=(256, 256, 3),
                     pooling='max')


def extract_features(datagen, image_dir, mode, num):
    # TODO: seed...
    generator = datagen.flow_from_directory(
            image_dir,
            target_size=(256, 256),
            batch_size=32,
            subset=mode)

    features = np.empty(shape=(num, 2048))
    labels = np.empty(shape=(num, 4))

    i = 0
    for inputs_batch, labels_batch in generator:
        batch_size = min(num - i, inputs_batch.shape[0])
        features_batch = extractor.predict(inputs_batch[:batch_size])
        features[i:(i + batch_size)] = features_batch
        labels[i:(i + batch_size)] = labels_batch[:batch_size]

        i += batch_size
        logger.debug("extracting features: {}/{}".format(i, num))
        if i >= num:
            break

    return features, labels


def load_extracted_features():
    dirname = "./temp/deep-features/"
    if os.path.exists(dirname):
        logger.info("found extracted features in " + dirname + ", proceeding")
        loaded = np.load(dirname + "training.npz")
        tr_features = loaded['tr_features']
        tr_labels = loaded['tr_labels']
        val_features = loaded['val_features']
        val_labels = loaded['val_labels']
        return tr_features, tr_labels, val_features, val_labels

    logger.info("did not find extracted features in " + dirname)

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
            rotation_range=10.0,
            shear_range=5.0,
            zoom_range=0.2,
            rescale=1./255,
            validation_split=0.2)

    num_train = 512
    logger.info("extracting {} training features".format(num_train))
    tr_features, tr_labels = extract_features(train_datagen, image_dir,
                                              "training", num_train)

    num_val = int(num_train * 0.2)
    logger.info("extracting {} validation features".format(num_val))
    val_features, val_labels = extract_features(train_datagen, image_dir,
                                                "validation", num_val)

    os.makedirs(dirname)
    np.savez_compressed(dirname + "training.npz",
                        tr_features=tr_features,
                        tr_labels=tr_labels,
                        val_features=val_features,
                        val_labels=val_labels)

    return tr_features, tr_labels, val_features, val_labels


tr_features, tr_labels, val_features, val_labels = load_extracted_features()

logger.info("creating model")

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=2048,
                       name='class-fc1'))
model.add(layers.Dense(4, activation='softmax', name='output'))

logger.info("compiling model")
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=['acc'])

logger.info("fitting model")

early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                              patience=3)
history = model.fit(tr_features, tr_labels,
                    epochs=20, batch_size=32,
                    verbose=1,
                    validation_data=(val_features, val_labels),
                    callbacks=[early_stop_callback])


# TODO: testing performance

# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
