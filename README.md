# Audio-deepfake-detector
This is an ML project which is build to tackle the upcoming threat of deepfake audios on social media. It aims to build a model and employ it to detect deepfake audio files.

# Deepfake Audio Detection Project

## Description

This machine learning project is designed to detect deepfake audio files. Built using Flask, it features a simple web interface where users can upload audio files and receive a percentage score indicating the likelihood of the audio being fake. The core of this project is a deep learning CNN model trained on spectrogram images, achieving an accuracy of 85%.

## Steps to Build

1. **Clone the Repository**
   ```bash
   git clone [your-repo-link]
   cd [your-repo-directory]
   ```

2. **Dataset Access**
   - The dataset is available via a Google Drive link. Request access to the dataset from us.
   - Once granted, download the dataset and ensure it is structured correctly in your local environment.

3. **Model Training**
   - Use the provided Jupyter Notebook (`train_model.ipynb`) to train the model. Ensure the dataset is placed in the correct directories (`Train_Audio` and `Test_Audio`).
   - The model architecture is a simple CNN built using Keras with the following key layers:
     ```python
     model = Sequential([
         Flatten(input_shape=(775, 385, 3)),
         Dense(512, activation='relu'),
         Dropout(0.5),
         Dense(256, activation='relu'),
         Dropout(0.3),
         Dense(128, activation='relu'),
         Dropout(0.2),
         Dense(1, activation='sigmoid')
     ])
     ```

   - The model is compiled and trained as follows:
     ```python
     model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
     model.fit(train_audio, train_labels, validation_data=(val_audio, val_labels), epochs=15, batch_size=32)
     ```

   - Save the trained model as `deepfake_final.h5`:
     ```python
     model.save('deepfake_final.h5')
     ```

4. **Model Usage**
   - After training, ensure the model is named `deepfake_final.h5`.
   - If you wish to use a different model name, update the `model_predict.py` file accordingly.

5. **Run the Application**
   - Execute the following command to start the Flask application:
     ```bash
     python app.py
     ```

   - Upon running, you might see the following messages indicating detected changes:
     ```
     * Detected change in 'D:\\deepfake_final_Night\\final_deepfake.py', reloading
     * Detected change in 'D:\\deepfake_final_Night\\final_deepfake.py', reloading
     * Detected change in 'D:\\deepfake_final_Night\\final_deepfake.py', reloading
     * Detected change in 'D:\\deepfake_final_Night\\final_deepfake.py', reloading
     ```

     These messages indicate that the server is reloading due to code changes. Once the reloading completes, the application will run smoothly.

## Overview of Model Training

The training process involves the following steps:

- **Data Loading:** Images of spectrograms from the dataset are loaded and preprocessed.
- **Model Architecture:** A CNN model is constructed using Keras, featuring multiple dense layers with dropout for regularization.
- **Training:** The model is trained on the preprocessed data for 15 epochs with a batch size of 32.
- **Evaluation:** The model's performance is evaluated on the test set, achieving an accuracy of 85%.
- **Saving the Model:** The trained model is saved as `deepfake_final.h5`.

By following these steps, users can train and deploy their own deepfake audio detection model using this project.
