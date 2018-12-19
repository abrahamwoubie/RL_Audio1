import numpy as np
import scipy
import scipy.io
from scipy import signal
import scipy.io.wavfile
# Read the .wav file
#sample_rate, data = scipy.io.wavfile.read('Hello.wav')
# Spectrogram of .wav file
#sample_freq, segment_time, spec_data = signal.spectrogram(data, sample_rate)
#print(np.mean(spec_data))

from keras.models import Sequential
from keras.layers import Dense

def model():
    model=Sequential
    model.add(Dense(20,input_dim=20,activation='relu'))

