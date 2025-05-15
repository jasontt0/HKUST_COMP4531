import numpy as np
from scipy import signal
import wave
import matplotlib.pyplot as plt

# read audio file recorded by Raspberry pi
file = wave.open('channel_1.wav', 'rb')
# get sampling frequency
sf = file.getframerate()
# get audio data total length
n_length = file.getnframes()
# read audio data
audio_data_raw = file.readframes(n_length)
# transfer to python list
audio_data_raw = list(audio_data_raw)
# transfer to numpy array
audio_data_raw = np.asarray(audio_data_raw, np.int8)
# set the data type to int16
audio_data_raw.dtype = 'int16'
# calculate audio length in second
audio_data_raw_total_time = n_length/sf
# close the file
file.close()

# cut the middle part of the audio data
time_offset = 2
total_time = np.int32(np.ceil(audio_data_raw_total_time - time_offset - 2))
total_point = total_time * sf
time_offset_point = time_offset * sf
audio_data = audio_data_raw[range(time_offset_point,time_offset_point+total_point)]

# set frequency
freq = 200
# calculate time t
t = np.arange(total_point)/sf
# get the cos and sin used in demodulation
signal_cos = np.cos(2*np.pi*freq*t)
signal_sin = np.sin(2*np.pi*freq*t)
# get a low-pass filter
b, a = signal.butter(3, 50/(sf/2), 'lowpass')
# Todo: multiply received signal and demodulate signal (Hint use scipy.signal.filtfilt method)
signalI = signal.filtfilt(b, a, audio_data*signal_cos)
signalQ = signal.filtfilt(b, a, audio_data*signal_sin)
# remove the static vector
signalI = signalI - np.mean(signalI)
signalQ = signalQ - np.mean(signalQ)
# calculate the phase angle
phase = np.arctan(signalQ/signalI)
# unwrap the phase
phase = np.unwrap(phase*2)/2
# Todo: calculate the distance
wavelength = 342/freq
distance = phase/2/np.pi*wavelength/2




# plot the distance
plt.plot(t, distance)
plt.xlabel('time/s')
plt.ylabel('distance/m')
plt.show()
plt.savefig("position.png")

print(distance)