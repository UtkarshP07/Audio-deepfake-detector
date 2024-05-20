import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def preprocess_and_save_spectrogram(input_file, output_dir):
    y, sr = librosa.load(input_file)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # Turn off axis
    output_file = os.path.join(output_dir, os.path.basename(input_file)[:-4] + '.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Spectrogram image saved at: {output_file}")
    spectrogram_image = plt.imread(output_file)
    return spectrogram_image
