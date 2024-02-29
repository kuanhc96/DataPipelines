from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100
from tensorflow.data import AUTOTUNE
import tensorflow as tf

BS = 64
NUM_STEPS = 5000

print("[INFO] loading the cifar100 dataset")
((trainX, trainY), (testX, testY)) = cifar100.load_data()

print("[INFO] creating ImageDataGenerator object")
imageGen = ImageDataGenerator()
dataGen = imageGen.flow(
    x=trainX,
    y=trainY,
    batch_size=BS,
    shuffle=True
)

print("[INFO] creating tf.data pipeline")
dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
dataset = (
    dataset
    .shuffle(1024)
    .cache()
    .repeat()
    .batch(BS)
    .prefetch(AUTOTUNE)
)

totalTime = benchmark(dataGen, NUM_STEPS)
print(f"[INFO] ImageDataGenerator generated {BS * NUM_STEPS} in {totalTime} seconds")

datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
print(f"[INFO] tf.data generated {BS * NUM_STEPS} images in {totalTime} seconds")