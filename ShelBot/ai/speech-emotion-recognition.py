print("[*] Importing the packages...")

import sounddevice as sd
import soundfile as sf
import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import sys
import pickle # to save model after training
from os import path
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn.linear_model import LogisticRegression

debug = False
retrain = False

fs = 48000
sd.default.samplerate=fs
duration=5

modelName = "speech-emotion-model.sav"

for args in sys.argv[1:]:
    if(args == "-r" or args == "--retrain"):
        retrain = True
    elif(args == "-d" or args == "--debug"):
        debug = True

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    # with soundfile.SoundFile(file_name) as sound_file:
        # X = sound_file.read(dtype="float32")
    X, sample_rate = librosa.load(file_name)
        # sample_rate = sound_file.samplerate
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result   

int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy",
    "suprised"
}

def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("RawData/speech-emotion/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def get_sample():
    myrec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write('realtime_aud.wav', myrec, fs)

if(not path.exists(modelName) or retrain):
    print("[*] Loading the data...")
    X_train, X_test, y_train, y_test = load_data(test_size=0.25)

    if(debug):
        print("[+] Number of training samples:", X_train.shape[0])
        print("[+] Number of testing samples:", X_test.shape[0])
    

    model = MLPClassifier(hidden_layer_sizes=(691), activation="logistic", learning_rate='constant', learning_rate_init=0.0006, max_iter= 1000)#= CNNClassifier(num_epochs=500, layers=2, dropout=0.00001)
    print("[*] Training the model... ")
    model.fit(X_train, y_train)

    # Testing
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))

    # Saving the model
    pickle.dump(model, open(modelName, 'wb'))

model = pickle.load(open(modelName, 'rb'))

while True:
    get_sample()
    feature = extract_feature("realtime_aud.wav", mfcc=True, mel=True)
    feature = feature.reshape(1, -1)
    emotion_pred = model.predict(feature)
    print(emotion_pred)


