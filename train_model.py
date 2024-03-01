import matplotlib
matplotlib.use("Agg")


from pyimagesearch.cancernet import CancerNet
from pyimagesearch import config
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.data import AUTOTUNE
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os

def load_images(imagePath):
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, config.IMAGE_SIZE)

    label = tf.strings.split(imagePath, os.path.sep)[-2]
    label = tf.strings.to_number(label, tf.int32)

    return (image, label)

@tf.function # the tf decorator converts a python function into a tensorflow-callable "graph"
def augment(image, label):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    return (image, label)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# datasets/idc/{train, validation, test}/{0,1}/*.png
trainPaths = list(paths.list_images(config.TRAIN_PATH))
valPaths = list(paths.list_images(config.VALIDATION_PATH))
testPaths = list(paths.list_images(config.TEST_PATH))

# p.split(os.path.sep)[-2] will get {0, 1}
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels) # one hot encoding
classTotals = trainLabels.sum(axis=0) # [22513, 8910]
classWeight = {}

# classWight = [1.0 2.5]
# Idea: the class with more representation will be weighted 1.0 (no weights)
# the class with less representation will be weighted as (count of class weighted 1.0) / (count of class with less representation)
# in this case, the class with less representation will be ~2.5
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
trainDS = (
    trainDS
    .shuffle(len(trainPaths))
    .map(load_images, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(config.BS)
    .prefetch(AUTOTUNE)
)
print(trainDS)

valDS = tf.data.Dataset.from_tensor_slices(valPaths)
valDS = (
    valDS
    .map(load_images, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(config.BS)
    .prefetch(AUTOTUNE)
)

testDS = tf.data.Dataset.from_tensor_slices(testPaths)
testDS = (
    testDS
    .map(load_images, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(config.BS)
    .prefetch(AUTOTUNE)
)

model = CancerNet.build(width=48, height=48, depth=3, classes=1) # one class, since CancerNet uses the sigmoid function as output
adagrad = Adagrad(learning_rate=config.INIT_LR, weight_decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=adagrad, metrics=["accuracy"])

earlyStopping = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

trainingHistory = model.fit(
    x=trainDS,
    validation_data=valDS,
    class_weight=classWeight,
    epochs=config.NUM_EPOCHS,
    callbacks=[earlyStopping],
    verbose=1
)

(_, acc) = model.evaluate(testDS)
print(f"[INFO] test accuracy: {acc * 100}%")

plt.style.use("ggplot")
plt.figure()
plt.plot(trainingHistory.history["loss"], label="train_loss")
plt.plot(trainingHistory.history["val_loss"], label="val_loss")
plt.plot(trainingHistory.history["accuracy"], label="train_acc")
plt.plot(trainingHistory.history["val_accuracy"], label="val_acc")
plt.title("Training loss and Accuracy on Dataset")
plt.xlabel("epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])