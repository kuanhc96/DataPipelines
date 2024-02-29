from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

BS = 64
NUM_STEPS = 1000
print("[INFO] loading image paths")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = np.array(sorted(os.listdir(args["dataset"])))

def load_images(imagePath):
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (96, 96)) / 255.0

    label = tf.strings.split(imagePath, os.path.sep)[-2]
    oneHot = label == classNames
    encodedLabel = tf.argmax(oneHot)

    return (image, encodedLabel)

print("[INFO] creating a df.data input pipeline")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = (
    dataset
    .shuffle(1024)
    .map(load_images, num_parallel_calls=AUTOTUNE)
    .cache()
    .repeat()
    .batch(BS)
    .prefetch(AUTOTUNE)
)

print("[INFO] creating an ImageDataGenerator object")
imageGen = ImageDataGenerator(rescale=1.0/255)
dataGen = imageGen.flow_from_directory(
    args["dataset"],
    target_size=(96, 96),
    batch_size=BS,
    class_mode="categorical",
    color_mode="rgb"
)

totalTime = benchmark(dataGen, NUM_STEPS)
print(f"[INFO] ImageDataGenerator generated {BS * NUM_STEPS} images in {totalTime} seconds")

datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
print(f"[INFO] tf.data generated {BS * NUM_STEPS} images in {totalTime} seconds")