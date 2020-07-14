print("[*] Importing packages...")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import cv2

# Keras
import tensorflow
from tensorflow import keras
tensorflow.__version__

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json


debug = False
retrain = False

for args in sys.argv[1:]:
    if(args == "-r" or args == "--retrain"):
        retrain = True
    elif(args == "-d" or args == "--debug"):
        debug = True

epochs = 50

mjson_file = "model.json"
mweights_file = "model_weights.h5"

# size of the image: 48*48 pixels
pic_size = 48

# input path for the images
base_path = "../RawData/facial-expression/images/"

EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

plt.figure(0, figsize=(12,20))
cpt = 0

def load_model(model_json_file, model_weights_file):
    # load model from JSON file
    with open(model_json_file, "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    # load weights into the new model
    loaded_model.load_weights(model_weights_file)
    return loaded_model

def predict_emotion(loaded_model, img):
    preds = loaded_model.predict(img)
    return EMOTIONS_LIST[np.argmax(preds)]


if(retrain):

    if(debug):
        for expression in os.listdir(base_path + "train/"):
            for i in range(1,6):
                cpt = cpt + 1
                plt.subplot(7,5,cpt)
                img = load_img(base_path + "train/" + expression + "/" +os.listdir(base_path + "train/" + expression+"/")[i], target_size=(pic_size, pic_size))            
                plt.imshow(img, cmap="gray")

        plt.tight_layout()
        plt.show()

    if(debug):
        for expression in os.listdir(base_path + "train"):
            print(str(len(os.listdir(base_path + "train/" + expression))) + " " + expression + " images")

    # number of images to feed into the NN for every batch
    print("[*] Training the model...")
    batch_size = 128

    datagen_train = ImageDataGenerator()
    datagen_validation = ImageDataGenerator()

    train_generator = datagen_train.flow_from_directory(base_path + "train",
                                                        target_size=(pic_size,pic_size),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = datagen_validation.flow_from_directory(base_path + "validation",
                                                        target_size=(pic_size,pic_size),
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=False)

    # number of possible label values
    nb_classes = 7

    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes, activation='softmax'))

    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=100,#train_generator.n//train_generator.batch_size,
                                    epochs=epochs,
                                    validation_data = validation_generator,
                                    validation_steps = 2,#validation_generator.n//validation_generator.batch_size,
                                    callbacks=callbacks_list
                                    )
else:
    print("[*] Loading the model...")
    loaded_model = load_model(mjson_file, mweights_file)
    
    if(debug):
        loaded_model.summary()
    
    facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video = cv2.VideoCapture(0)

    print("[*] Running recog")
    while True:
        _, fr = cv2.rotate(video.read(), cv2.ROTATE_180)
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        print("FAces: ", faces)

        if(debug):
            cv2.imwrite("original_fr.png", fr)
            cv2.imwrite("grayscale_fr.png", gray_fr)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = predict_emotion(loaded_model, roi[np.newaxis, :, :, np.newaxis])
            print(pred)




