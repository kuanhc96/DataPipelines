import matplotlib
matplotlib.use("Agg")

from pyimagesearch.simple_dataset_loader import SimpleDatasetLoader
from pyimagesearch.simple_preprocessor import SimplePreprocessor
from pyimagesearch.miniVGGNet import MiniVGGNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--augment", action="store_true", help="whether or not to augment the image dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 1e-1
BS = 8
EPOCHS = 50
class_labels = os.listdir(args["dataset"])
num_classes = len( class_labels )

print("[INFO] loading dataset")
if args["augment"]:
    print("[INFO] performing data augmentation on the fly")
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.25
    )
    train_iterator = aug.flow_from_directory(
        directory=args["dataset"],
        seed=42,
        shuffle=True,
        subset="training",
        class_mode="categorical",
        color_mode="rgb",
        target_size=(64, 64),
        batch_size=BS
    )
    test_iterator = aug.flow_from_directory(
        directory=args["dataset"],
        seed=42,
        shuffle=True,
        subset="validation",
        class_mode="categorical",
        color_mode="rgb",
        target_size=(64, 64),
        batch_size=BS
    )
    print(test_iterator.labels)

    print("[INFO] compiling model")
    sgd = SGD(learning_rate=INIT_LR, momentum=0.9, weight_decay=INIT_LR/EPOCHS)
    model = MiniVGGNet.build(64, 64, 3, num_classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    print(f"[INFO] training network for {EPOCHS} epocs")
    training_history = model.fit(
        x=train_iterator,
        validation_data=test_iterator,
        epochs=EPOCHS
    )
    print("[INFO] evaluating network")
    predictions = model.predict(x=test_iterator)
    print(classification_report(test_iterator.labels, predictions.argmax(axis=1), target_names=class_labels))

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

    print("[INFO] evaluating network")
    predictions = model.predict(x=testX.astype("float32"), batch_size=BS)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelEncoder.classes_))

epoch_linspace = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(epoch_linspace, training_history["loss"], label="train_loss")
plt.plot(epoch_linspace, training_history["val_loss"], label="val_loss")
plt.plot(epoch_linspace, training_history["accuracy"], label="train_acc")
plt.plot(epoch_linspace, training_history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy on dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

