import scipy.io.wavfile
import numpy as np
def write_audio(filename, samplerate, audio):
    audio_data = np.frombuffer(audio, dtype=np.int16)
    scipy.io.wavfile.write(filename, samplerate, audio_data)
    