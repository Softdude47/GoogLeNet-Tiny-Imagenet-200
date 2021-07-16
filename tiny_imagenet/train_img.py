
import os
import sys
import json
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append("../")
from mas_lib.nn.conv.deepergooglenet import DeeperGoogLeNet
from mas_lib.callbacks.epochcheckpoint import EpochCheckpoint
from mas_lib.callbacks.trainingmonitor import TrainingMonitor
from mas_lib.io.imagedatasetgenerator import ImageDatasetGenerator
from mas_lib.preprocessing.meanpreprocessor import MeanPreprocessor
from mas_lib.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from tiny_imagenet.configs import tiny_imagenet_config as config

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="path to load specific model")
ap.add_argument("-start", "--start-at", help="epoch at which model training resumes", type=int)
ap.add_argument(
    "-c",
    "--checkpoint",
    required=True,
)
args = vars(ap.parse_args())

# load datasets RGB mean values
mean = json.loads(open(config.DATASET_MEAN).read())

# initiailze preprocessors
iap = ImageToArrayPreprocessor().preprocess
mp = MeanPreprocessor(rMean=mean["R"], gMean=mean["G"], bMean=mean["B"]).preprocess

# initialize dataset generators
aug = ImageDataGenerator(
    zoom_range = 0.2,
    shear_range = 0.2,
    rotation_range = 30,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
)
gen = ImageDatasetGenerator(
    path=config.TRAIN_IMAGE,
    label_index=-3, # index of label in list generated path(os.path.sep)
    batch_size=config.BATCH_SIZE,
    preprocessors=[mp, iap],
    aug=aug.flow,
    target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
    validation_split=0.1,
    val_preprocessors=[mp, iap]
)

# create model callbacks
callbacks=[
    LearningRateScheduler(config.poly_weight_decay),
    TrainingMonitor(config.FIG_PATH, config.JSON_PATH, args.get("start_at", 0)),
    EpochCheckpoint(path=args["checkpoint"], interval=5)
]

# creates new model if there wasn't a pevious one
if args["model"] is None:
    # initialize and tune model optimizer
    sgd = SGD(learning_rate=config.LEARNING_RATE, momentum=0.9)

    # build and compile model
    model = DeeperGoogLeNet.build(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3, config.NUM_CLASSES, 0.000503)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

# loads previously saved model(if any)
else:
    # load model
    model = load_model(args["model"])
    prev_lr = K.get_value(model.optimizer.lr)
    
    # change learning rate of model optimizer
    print(f"[INFO] Previous learning rate {prev_lr}")
    K.set_value(model.optimizer.lr, 0.00005)
    print(f"[INFO] New learning rate set to {K.get_value(model.optimizer.lr)}")

# train model
model.fit(
    gen.generate("train"),
    callbacks=callbacks,
    epochs=config.EPOCHS,
    validation_data=gen.generate("val"),
    steps_per_epoch=gen.get_steps("train"),
    validation_steps=gen.get_steps("val"),
)
