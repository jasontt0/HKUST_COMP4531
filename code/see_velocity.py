import numpy as np
import scipy
import wave
import matplotlib.pyplot as plt

from scipy import signal as scipysignal
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
audio_data_raw = np.asarray(audio_data_raw, np.uint8)
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

# Please notice the sample rate of the audio, normally 44100Hz/48000Hz
f, t, Zxx = scipy.signal.stft(audio_data, fs=44100, window='hann', nperseg=4096, noverlap=1024, nfft=8192)
freq_mask = (f < 17000) | (f > 19000)
Zxx[freq_mask, :] = 0

# TODO: Complete this section to calculate velocity from Fourier Results
# HINTS:
# 1. Find the peak frequencies in each time frame in Zxx. Zxx contains the amplitude of the frequency components in each time frame.
# 2. Calculate the velocity based on the peak frequencies and Doppler Effect formula.

# Constants (You might need to adjust these based on your setup)
SOUND_SPEED = 343  # Speed of sound in air (m/s)
EMITTED_FREQUENCY = 200 # 发射频率，需要根据实际情况修改，这里假设发射频率为18000Hz

# 1. Find the peak frequencies in each time frame
frequencies = []
for i in range(Zxx.shape[1]):  # Iterate over time frames (columns)
    # Find the index of the maximum amplitude in the current time frame
    peak_index = np.argmax(np.abs(Zxx[:, i]))  # Use np.abs to handle complex numbers
    # Get the corresponding frequency
    peak_frequency = f[peak_index]
    frequencies.append(peak_frequency)

frequencies = np.array(frequencies)

# 由频率计算当前速度
velocity = SOUND_SPEED * ((frequencies-EMITTED_FREQUENCY) / (frequencies+EMITTED_FREQUENCY))


# Correct the length of velocity array
velocity = velocity[:len(t)-1]
plt.plot(t[:-1], velocity)
plt.savefig("velocity.png")
