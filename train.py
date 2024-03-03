import matplotlib
matplotlib.use("Agg")

from pyimagesearch.simple_dataset_loader import SimpleDatasetLoader
from pyimagesearch.simple_preprocessor import SimplePreprocessor
from pyimagesearch.miniVGGNet import MiniVGGNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical # for one hot encoding
from tensorflow.keras.layers.experimental import preprocessing
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--augment", action="store_true", help="whether or not to augment the image dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 1e-2
BS = 32
EPOCHS = 50
class_labels = os.listdir(args["dataset"])
num_classes = len( class_labels )
print("NUM CLASSES:", num_classes)

def load_images(imagePath, label):
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (64, 64))

    # label = tf.strings.split(imagePath, os.path.sep)[-2]

    return (image, label)

def augment(image, label, aug):
    image = aug(image)

    return (image, label)

print("[INFO] loading dataset")
if args["augment"]:
    print("[INFO] performing data augmentation on the fly")
    allImages = list(paths.list_images(args["dataset"]))
    random.shuffle(allImages)
    i = int(len(allImages) * 0.25)
    trainPaths = allImages[i:]
    trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
    print("TRAIN LABELS", trainLabels)

    testPaths = allImages[:i]
    testLabels = [p.split(os.path.sep)[-2] for p in testPaths]

    labelEncoder = LabelEncoder()
    trainLabels = labelEncoder.fit_transform(trainLabels)
    # trainLabels = trainLabels.reshape(len(trainLabels), 1)
    trainLabels = to_categorical(trainLabels)

    testLabels = labelEncoder.fit_transform(testLabels)
    testLabels = to_categorical(testLabels)

    classTotals = trainLabels.sum(axis=0)
    classWeights = {}

    for i in range(0, len(classTotals)):
        classWeights[i] = classTotals.max() / classTotals[i]


    trainDS = tf.data.Dataset.from_tensor_slices((trainPaths, trainLabels))
    trainDS = (
        trainDS
        .shuffle(len(trainPaths), seed=42)
        .map(load_images, num_parallel_calls=AUTOTUNE)
        .batch(BS)
        .cache()
    )

    #     rotation_range=20,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     validation_split=0.25

    # default shear method does not exist in "layers.preprocessing"
    trainAug = tf.keras.Sequential(
        [
            preprocessing.Rescaling(scale=1.0/255),
            preprocessing.RandomRotation(20.0/360),
            preprocessing.RandomZoom(height_factor=(0, 0.15), fill_mode="nearest"),
            preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2),
            preprocessing.RandomFlip(mode="horizontal")

        ]
    )
    
    trainDS = (
        trainDS
        .map(lambda x, y: augment(x, y, trainAug), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    testDS = tf.data.Dataset.from_tensor_slices(( testPaths, testLabels ))
    testDS = (
        testDS
        .shuffle(len(testPaths))
        .map(load_images, num_parallel_calls=AUTOTUNE)
        .batch(BS)
        .cache()
    )

    testAug = tf.keras.Sequential(
        [
            preprocessing.Rescaling(scale=1.0/255),
        ]
    )

    testDS = (
        testDS
        .map(lambda x, y: augment(x, y, testAug), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # aug = ImageDataGenerator(
    #     rotation_range=20,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     validation_split=0.25
    # )
    # train_iterator = aug.flow_from_directory(
    #     directory=args["dataset"],
    #     seed=42,
    #     shuffle=True,
    #     subset="training",
    #     class_mode="categorical",
    #     color_mode="rgb",
    #     target_size=(64, 64),
    #     batch_size=BS
    # )
    # test_iterator = aug.flow_from_directory(
    #     directory=args["dataset"],
    #     seed=42,
    #     shuffle=True,
    #     subset="validation",
    #     class_mode="categorical",
    #     color_mode="rgb",
    #     target_size=(64, 64),
    #     batch_size=BS
    # )

    print("[INFO] compiling model")
    sgd = SGD(learning_rate=INIT_LR, momentum=0.9, weight_decay=INIT_LR/EPOCHS)
    model = MiniVGGNet.build(64, 64, 3, num_classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    print(f"[INFO] training network for {EPOCHS} epocs")
    training_history = model.fit(
        x=trainDS,
        validation_data=testDS,
        epochs=EPOCHS
    )
    model.save(os.path.join(args["plot"], f"augmented_weights_epochs_{EPOCHS}.hdf5"))
    print("[INFO] evaluating network")
    (_, acc) = model.evaluate(testDS)
    print(f"[INFO] test accuracy: {acc * 100}%")

else:
    preprocessors = [SimplePreprocessor(64, 64)]
    loader = SimpleDatasetLoader(preprocessors=preprocessors)
    image_paths = list(paths.list_images(args["dataset"]))
    (data, labels) = loader.load(imagePaths=image_paths)

    data = data.astype("float") / 255.0

    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(labels)
    labels = to_categorical(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("[INFO] compiling model")
    sgd = SGD(learning_rate=INIT_LR, momentum=0.9, weight_decay=INIT_LR/EPOCHS)
    model = MiniVGGNet.build(64, 64, 3, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    print(f"[INFO] training network for {EPOCHS} epocs")

    training_history = model.fit(
        x=trainX,
        y=trainY,
        validation_data=(testX, testY),
        batch_size=BS,
        epochs=EPOCHS
    )
    model.save(os.path.join(args["plot"], f"non_augmented_weights_epochs_{EPOCHS}.hdf5"))

    print("[INFO] evaluating network")
    predictions = model.predict(x=testX.astype("float32"), batch_size=BS)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelEncoder.classes_))

epoch_linspace = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(epoch_linspace, training_history.history["loss"], label="train_loss")
plt.plot(epoch_linspace, training_history.history["val_loss"], label="val_loss")
plt.plot(epoch_linspace, training_history.history["accuracy"], label="train_acc")
plt.plot(epoch_linspace, training_history.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy on dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
if args["augment"]:
    plt.savefig(os.path.join(args["plot"], f"augmented_training_history.png"))
else:
    plt.savefig(os.path.join(args["plot"], f"non_augmented_training_history.png"))

