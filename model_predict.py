from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def load_model_for_prediction():
    model_path = 'deepfake_final.h5'
    model = load_model(model_path)
    return model

def predict_real_or_fake(image_path):
    model = load_model_for_prediction()
    img = load_img(image_path, target_size=(775, 385))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    probabilities = model.predict(img_array)[0]

    if len(probabilities) == 1:
        fake_probability = 1 - probabilities[0]
        real_probability = probabilities[0]
        print(f"Probability of being Fake: {fake_probability * 100:.2f}%")
        print(f"Probability of being Real: {real_probability * 100:.2f}%")
    else:
        fake_probability = probabilities[0]
        real_probability = probabilities[1]
        print(f"Probability of being Fake: {fake_probability * 100:.2f}%")
        print(f"Probability of being Real: {real_probability * 100:.2f}%")

    probabilities = 0
    return fake_probability * 100, real_probability * 100