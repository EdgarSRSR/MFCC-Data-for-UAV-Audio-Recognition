 #records once and analyses
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import cross_val_score #new
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix


import pyaudio
import wave

## STEP 1: PREPARES THE MODEL

#takes training data from the mfcc csv 
data = pd.read_csv("results.csv")

#prepares the data for fitting it in the model
x = data.drop('label', axis=1)
y = data['label']
#shuffles it
X_sparse = coo_matrix(x)
x, X_sparse, y = shuffle(x, X_sparse, y, random_state=0)

#divide data into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

## STEP 2: TRAINS THE MODEL
#training
svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(x_train, y_train)

## STEP 3: CREATES FUNCTION FOR USING THE MODEL AND PRINTING THE RESULTS

#extracts the mfccs features of the audio sample
def extract_feature(file_name):
    try:
        y, sr = librosa.load(filename)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=40).T,axis=0) 
        feature = mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None
    return feature

#prints the description of the audio sample
def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 
    prediction_feature.reshape(1,-1)
    y_pred = svclassifier.predict([prediction_feature])

    if (y_pred[0] == 1):
        print('Drone Detected')
    else:
        print('No Drone Detected')

## STEP 4: RECORDS 

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 5
audiofile = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print(' Recording '.center(40,'#'))

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print(' Finished recording '.center(40,'#'))

# Save the recorded data as a WAV file
wf = wave.open(audiofile, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

## STEP 5: ANALYSES THE RECORDING AND PRINTS RESULT

filename = 'output.wav' 
print_prediction(filename) 
