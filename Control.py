from keras.models import Sequential
from keras.layers import Convolution2D,Activation,MaxPooling2D,BatchNormalization,Dropout,Flatten,Dense,Conv2D
from keras import  optimizers
from ExtractFeatures import Extract_Features
d=Extract_Features
data=d.Extract_Spectrogram(0,0)
print(data.shape)
model = Sequential()
model.add(Conv2D(64, (1, 1), activation='relu', input_shape=data.shape))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='sgd')

# import numpy as np
# import scipy
# import scipy.io
# from scipy import signal
# import scipy.io.wavfile
# # #Read the .wav file
# # sample_rate, data = scipy.io.wavfile.read('Hello.wav')
# # #Spectrogram of .wav file
# # sample_freq, segment_time, spec_data = signal.spectrogram(data, sample_rate)
# # print(np.mean(spec_data))
#
# from ExtractFeatures import Extract_Features
# rdat=Extract_Features
# raw_data=rdat.Extract_Raw_Data(4,4)
# spectrogram=rdat.Extract_Spectrogram(0,0)
# print(len(raw_data))
# #print(len(spectrogram[0]))
