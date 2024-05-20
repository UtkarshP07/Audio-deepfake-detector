import librosa
from scipy.io import wavfile

def convert_mp3_to_wav(mp3_file, wav_file):
    y, sr = librosa.load(mp3_file, sr=None)
    wavfile.write(wav_file, sr, y)