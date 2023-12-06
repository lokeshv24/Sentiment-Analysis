import os
from flask import Flask, render_template, request
import numpy as np
import librosa
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model('Emotion_Voice_Detection_Model.h5')  # Replace with the actual path to your model
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def prediction(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

def predict_emotion_from_features(features):
    if features is not None:
        # Transpose the features to match the (None, 40, 1) shape expected by the model
        features = np.transpose(features)

        # Expand dimensions to match the input shape expected by the model
        features = np.expand_dims(features, axis=-1)

        # Make predictions
        predictions = model.predict(features)

        # Get the emotion label with the highest probability
        predicted_emotion_index = np.argmax(predictions)

        # Check if the predicted_emotion_index is within the valid range
        if 0 <= predicted_emotion_index < len(emotion_labels):
            predicted_emotion = emotion_labels[predicted_emotion_index]
            return predicted_emotion
        else:
            return 'Invalid emotion index'
    else:
        return "Unable to extract features from the audio file"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the 'audio' file is in the request
        if 'audio' not in request.files:
            return "No audio file provided."

        audio_file = request.files['audio']

        # Check if the file has a valid extension
        if audio_file and audio_file.filename.lower().endswith(('.wav')):
            # Save the uploaded audio file
            audio_file_path = os.path.join('static/uploads', 'uploaded_audio.wav')
            audio_file.save(audio_file_path)

            # Extract features and predict emotion
            features = prediction(audio_file_path)
            predicted_emotion = predict_emotion_from_features(features)
            return render_template('submit.html', result=predicted_emotion)

    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        print(file_path)
        prection1 = prediction(file_path)
        final_result = predict_emotion_from_features(prection1)
        return render_template('submit.html', result=+final_result)

if __name__ == '__main__':
    app.run(debug=True)
