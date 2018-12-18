import numpy as np
import scipy
import scipy.io
from scipy import signal
import scipy.io.wavfile
# Read the .wav file
sample_rate, data = scipy.io.wavfile.read('test.wav')
# Spectrogram of .wav file
sample_freq, segment_time, spec_data = signal.spectrogram(data, sample_rate)
print(len(spec_data[0]))