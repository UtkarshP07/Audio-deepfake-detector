#file imports karenge
from convert_to_wav import convert_mp3_to_wav
from create_spectrogram import preprocess_and_save_spectrogram
from model_predict import predict_real_or_fake

#Libraries and module imports karenge
import os 
from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__)

def convert_to_wav_if_needed(file_path):
    base_dir = 'uploads/wav'
    if file_path.endswith('.wav'):
        return os.path.basename(file_path)
    wav_file = os.path.join(base_dir, os.path.splitext(os.path.basename(file_path))[0] + '.wav')
    convert_mp3_to_wav(file_path, wav_file)

    return os.path.basename(wav_file)

def convert_and_save_spectrogram(input_file):
    base_dir = 'uploads/spectrograms'
    wav_file = convert_to_wav_if_needed(input_file)
    wav_file_dir = "uploads/wav/"+wav_file
    img = preprocess_and_save_spectrogram(wav_file_dir, base_dir)

    return os.path.basename(wav_file)[:-4] + '.png'

def process_audio_file(input_file):
    spectrogram_file = convert_and_save_spectrogram(input_file)
    spectrogram_file_saved = "uploads/spectrograms/"+spectrogram_file
    fake_probability, real_probability = predict_real_or_fake(spectrogram_file_saved)
    with open(spectrogram_file_saved, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    return img_base64,fake_probability, real_probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join('uploads/mp3', file.filename)
    file.save(file_path)
    
    img, fake_probability, real_probability = process_audio_file(file_path)
    return jsonify({'img': img, 'fake_probability': round(fake_probability,2), 'real_probability': (real_probability,2)})

if __name__ == '__main__':
    app.run(debug=True)