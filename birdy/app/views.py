from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import librosa
import json
import tensorflow as tf
import os
from django.conf import settings
def prediction(audio_file):
    # Define the path to your app directory
    app_directory = os.path.join(settings.BASE_DIR, 'app')

    # Load the Prediction JSON File to Predict Target_Label
    prediction_file_path = os.path.join(app_directory, 'prediction.json')
    with open(prediction_file_path, mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate = librosa.load(audio_file)

    # Extract the MFCC Features and Aggregate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model_file_path = os.path.join(app_directory, 'Birdmodel.h5')
    model = tf.keras.models.load_model(model_file_path)
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction) * 100, 2)

    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence}%')
    return predicted_class , confidence
def upload_audio_page(request):
    return render(request, 'app/index.html')

def upload_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file:
            app_directory = os.path.join(settings.BASE_DIR, 'app', 'audio_uploads')

            # Make sure the directory exists
            if not os.path.exists(app_directory):
                os.makedirs(app_directory)

            # Save the file in the specified directory
            fs = FileSystemStorage(location=app_directory)
            filename = fs.save(audio_file.name, audio_file)

            # Create the file path and return it
            file_path = os.path.join(app_directory, filename)
            print(f'Received file: {audio_file.name} ,{file_path }')
            name , confidence = prediction(file_path)
            return JsonResponse({'message': 'File received' ,"name":name ,"confidence":confidence})
        else:
            return JsonResponse({'message': 'No file received'}, status=400)
    return JsonResponse({'message': 'Invalid request'}, status=400)
