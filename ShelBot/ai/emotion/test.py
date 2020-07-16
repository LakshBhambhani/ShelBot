print("[*] Importing packages...")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

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

from sklearn.neural_network import MLPClassifier # multi-layer perceptron model

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,  save_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn.neighbors import KNeighborsClassifier




debug = False
retrain = False

for args in sys.argv[1:]:
    if(args == "-r" or args == "--retrain"):
        retrain = True
    elif(args == "-d" or args == "--debug"):
        debug = True

# size of the image: 48*48 pixels
pic_size = 48

# input path for the images
base_path = "../RawData/facial-expression/images/"

img_files=[]
labels=[]

for expression in os.listdir(base_path + "train/"):
    for f in os.listdir(base_path+"train/"+expression+"/"):
        img_files.append(f)
        labels.append(expression)

print("Working with {0} images".format(len(img_files)))

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(img_files), (pic_size*pic_size*channels)),
                     dtype=np.float32)

i = 0
for i, _file in enumerate(img_files):
    img = load_img(base_path+"train/"+labels[i]+"/"+_file)  # this is a PIL image
    img.thumbnail((pic_size, pic_size))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((pic_size*pic_size*channels))
    # Normalize
    # x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")

# print(dataset.shape)
# number, width, height, layers = dataset.shape
# dataset = dataset.reshape(number, (width*height*layers))
# print(dataset.shape)


#Splitting 
print(dataset.shape)
print(dataset)
print(len(labels))
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)
print("Train set size: {0}, Test set size: {1}".format(len(X_train), len(X_test)))
print(X_train.shape)
print(X_train)
print(len(y_train))

print(X_train[0])

model = KNeighborsClassifier(10)#MLPClassifier(hidden_layer_sizes=(691), activation="logistic", learning_rate='constant', learning_rate_init=0.0006, max_iter= 1000)#= CNNClassifier(num_epochs=500, layers=2, dropout=0.00001)

print("[*] Training the model... ")
history = model.fit(X_train, y_train)

# Testing
print("[*] Testing the model... ")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

