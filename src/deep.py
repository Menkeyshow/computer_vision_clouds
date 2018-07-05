from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks, layers, models, optimizers
import logging
import numpy as np
from scipy.misc import imread, imsave
import os

import util



logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

logger.info("logger says hi!")


def move_data_into_categorized_directories():
    dirname = "../temp/deep/categorized/"
    if util.ensure_directory_exists(dirname):
        logger.info("found categorized images in " + dirname + ", proceeding")
        return dirname

    logger.info("couldn't find categorized images in {}, copying images now"
                .format(dirname))

    for label, sublabels in util.cloud_kinds.items():
        logger.debug("processing {}".format(label))
        labeldirname = dirname + label + "/"
        os.makedirs(labeldirname)
        for cloud_type in sublabels:
            logger.debug("current cloud type: {}".format(cloud_type))
            sublabeldirname = "../data/" + cloud_type + "/"
            for filename in os.listdir(sublabeldirname):
                img = imread(sublabeldirname + filename)
                height = img.shape[0]
                width = img.shape[1]
                imsave(labeldirname + "l-" + filename, img[:(3 * height // 4), :(width // 2), :])
                imsave(labeldirname + "r-" + filename, img[:(3 * height // 4), (width // 2):, :])

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
    dirname = "../temp/deep/features/"
    if util.ensure_directory_exists(dirname):
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

    num_train = 4096
    logger.info("extracting {} training features".format(num_train))
    tr_features, tr_labels = extract_features(train_datagen, image_dir,
                                              "training", num_train)

    num_val = int(num_train * 0.2)
    logger.info("extracting {} validation features".format(num_val))
    val_features, val_labels = extract_features(train_datagen, image_dir,
                                                "validation", num_val)

    np.savez_compressed(dirname + "training.npz",
                        tr_features=tr_features,
                        tr_labels=tr_labels,
                        val_features=val_features,
                        val_labels=val_labels)

    return tr_features, tr_labels, val_features, val_labels


tr_features, tr_labels, val_features, val_labels = load_extracted_features()

logger.info("creating model")

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=2048,
                       name='class-fc1'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(128, activation='relu', name='class-fc2'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(4, activation='softmax', name='output'))

logger.info("compiling model")
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
              metrics=['acc'])

logger.info("fitting model")

early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                              patience=30)
checkpoint_callback = callbacks.ModelCheckpoint('../temp/deep/best_model.h5',
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='auto', period=1)
history = model.fit(tr_features, tr_labels,
                    epochs=500, batch_size=64,
                    verbose=1,
                    validation_data=(val_features, val_labels),
                    callbacks=[early_stop_callback, checkpoint_callback])

model.load_weights('../temp/deep/best_model.h5')
# TODO: true train/validate/test data split
score = model.evaluate(val_features, val_labels, verbose=1)
print("test loss: ", score[0], "accuracy: ", score[1])

predictions = model.predict_classes(val_features, verbose=1)

ground_truth = []
for label in val_labels:
    for i in range(len(label)):
        if label[i] > 0.:
            ground_truth.append(i)
            break

assert len(ground_truth) == len(predictions)

confusion_matrix = np.zeros(16).reshape((4, 4))
for true, prediction in zip(ground_truth, predictions):
    confusion_matrix[true][prediction] += 1

cm = confusion_matrix
for i in range(4):
    cm[i] /= np.sum(cm[i])

cm *= 100

print("\t".join(["", "cir", "cum", "str", "s-c"]))
print("cir\t%.2f\t%.2f\t%.2f\t%.2f" % (cm[0][0], cm[0][1], cm[0][2], cm[0][3]))
print("cum\t%.2f\t%.2f\t%.2f\t%.2f" % (cm[1][0], cm[1][1], cm[1][2], cm[1][3]))
print("str\t%.2f\t%.2f\t%.2f\t%.2f" % (cm[2][0], cm[2][1], cm[2][2], cm[2][3]))
print("s-c\t%.2f\t%.2f\t%.2f\t%.2f" % (cm[3][0], cm[3][1], cm[3][2], cm[3][3]))
