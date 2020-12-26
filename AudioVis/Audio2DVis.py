'''
Audio Visualizer
https://analyticsindiamag.com/step-by-step-guide-to-audio-visualization-in-python/
'''

# Imports
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# PyAudio Instance
p = pyaudio.PyAudio()  # Create an interface to PortAudio

# Main Functions
# Basic Audio Functions
def RecordAudio(time, chunkSize=1024, sample_format=pyaudio.paInt16, channels=2, frame_rate=44100, savePath=None):
    print('Recording')
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=frame_rate,
                    frames_per_buffer=chunkSize,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for time seconds
    for i in range(0, int(frame_rate / chunkSize * time)):
        data = stream.read(chunkSize)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    if savePath is not None:
        SaveAudio(savePath, frames, channels, sample_format, frame_rate)

    return frames

def SaveAudio(filePath, frames, channels, sample_format, frame_rate):
    # Save the recorded data as a WAV file
    wf = wave.open(filePath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(frame_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Librosa Visualisation
def LoadAudio(filePath, sample_rate=None, mono=True, offset=0.0, duration=None, res_type='kaiser_best', dtype=np.float32):
    data, sample_rate = librosa.load(filePath, sr=sample_rate, mono=mono, offset=offset, duration=duration, res_type=res_type, dtype=dtype)
    return data, sample_rate

def DisplayAudio_WavePlot(audio, sample_rate):
    librosa.display.waveplot(audio, sr=sample_rate, max_points=50000.0, x_axis='time', offset=0.0, max_sr=1000)
    plt.show()

def GetFrequencyData(audio, sample_rate):
    frequencies, times, spectrogram = signal.spectrogram(audio, sample_rate)
    return frequencies, times, spectrogram

def DisplayFrequencyData(frequencies, times, spectrogram):
    plt.imshow(spectrogram)
    # plt.pcolormesh(times, frequencies, spectrogram)

    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# Driver Code
# Params
mainPath = 'TestAudio/'
fileName = 'Iruvar.mp3'
offset = 0.0#np.random.randint(1, 300) / 10
duration = 5.0

# Load Audio
print("Loading", fileName, "from", str(offset), "-", str(offset + duration))
audio, sample_rate = LoadAudio(mainPath + fileName, duration=duration, offset=offset)
print("Audio Data Shape:", audio.shape)

# Visualise
# Amplitudes
DisplayAudio_WavePlot(audio, sample_rate)

# Frequencies
frequencies, times, spectrogram = GetFrequencyData(audio, sample_rate)
DisplayFrequencyData(frequencies, times, spectrogram)