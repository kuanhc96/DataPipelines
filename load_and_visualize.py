from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os

def load_images(imagePath):
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3) 
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (156, 156))

    label = tf.strings.split(imagePath, os.path.sep)[-2]

    return (image, label)

# aud is a data augmentation object, which is assumed to be an instance of Sequential
def augment_using_layers(images, labels, aug):
    # Pass a batch of images through the data augmentation pipeline 
    # and return the augmented images
    images = aug(images)

    return (images, labels)

def augment_using_ops(images, labels):
    # randomly flip images;
    # rotate images counter-clockwise by 90 degrees
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)

    return (images, labels)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input images dataset")
ap.add_argument("-a", "--augment", action="store_true", help="flag indicating whether or not augmentation should be applied")
ap.add_argument("-t", "--type", choices=["layers", "ops"], help="method to be used to perform data augmentation")
args = vars(ap.parse_args())

BATCH_SIZE = 8
imagePaths = list(paths.list_images(args["dataset"]))

print("[INFO] loading the dataset")
ds = tf.data.Dataset.from_tensor_slices(imagePaths)
ds = (
    ds
    .shuffle(len(imagePaths), seed=42)
    .map(load_images, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
)

if args["augment"]:
    if args["type"] == "layers":
        aug = tf.keras.Sequential(
            [
                preprocessing.RandomFlip("horizontal_and_vertical"),
                preprocessing.RandomZoom(
                    height_factor=(-0.05, -0.15),
                    width_factor=(-0.05, -0.15)
                ),
                preprocessing.RandomRotation(0.3)
            ]
        )

        ds = (
            ds
            .map(lambda x, y: augment_using_layers(x, y, aug), num_parallel_calls=AUTOTUNE)
        )
    else:
        ds = (
            ds
            .map(augment_using_ops, num_parallel_calls=AUTOTUNE)
        )
ds = (
    ds
    .prefetch(AUTOTUNE)
)

batch = next(iter(ds))

print("[INFO] visualizing the first batch of the dataset")
appliedOrNot = f"applied ({args['type'] if args['augment'] else 'not applied'})"
title = f"with data augmentation {appliedOrNot}"
fig = plt.figure(figsize=(BATCH_SIZE, BATCH_SIZE))
fig.suptitle(title)

for i in range(0, BATCH_SIZE):
    (image, label) = (batch[0][i], batch[1][i])

    ax = plt.subplot(2, 4, i+1)
    plt.imshow(image.numpy())
    plt.title(label.numpy().decode("UTF-8"))
    plt.axis("off")

plt.tight_layout()
plt.show()