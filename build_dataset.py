from pyimagesearch import config
from imutils import paths
import random
import shutil
import os

# imagePaths = dataset/orig/**/*.png
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths) # shuffle the ordering in the list

i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i] # trainPaths = dataset/orig/**/*.png
testPaths = imagePaths[i:]

i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = imagePaths[i:]

# config.TRAIN_PATH, config.VALIDATION_PATH, config.TEST_PATH
# are the paths that will be created
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VALIDATION_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# create paths for training, validation, and testing
# datasets/idc/training
# datasets/idc/validation
# datasets/idc/testing
for (dType, imagePaths, baseOutput) in datasets:
    print(f"[INFO] building {dType} split")

    if not os.path.exists(baseOutput):
        print(f"[INFO] creating {baseOutput} directory")
        os.makedirs(baseOutput)
    
    for inputPath in imagePaths:
        # *.png
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4] # skips the .png, results in either 0/1

        # create datasets/idc/training/0; datasets/idc/training/1
        # create datasets/idc/validation/0; datasets/idc/validation/1
        # create datasets/idc/testing/0; datasets/idc/testing/1
        labelPath = os.path.sep.join([baseOutput, label]) 

        if not os.path.exists(labelPath):
            print(f"[INFO] creating {labelPath} directory")
            os.makedirs(labelPath)

        # datasets/idc/training/0/*.png
        # datasets/idc/validation/0/*.png
        # datasets/idc/testing/0/*.png
        # datasets/idc/training/1/*.png
        # datasets/idc/validation/1/*.png
        # datasets/idc/testing/1/*.png
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)