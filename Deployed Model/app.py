from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from preprocessing import *

app = Flask(__name__)

# Model path
MODEL_PATH = '/workspaces/Diabetic-Retinopathy-with-CNN/Deployed Model/models/my_model.keras'

# Load the model
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(150, 150))  # Update target_size
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     # Perform any additional preprocessing if necessary
#     return x

def model_predict(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    preds = model.predict(preprocessed_image)
    return preds

def get_prediction_label(predictions):
    # Define your labels
    labels = ['0','1', '2', '3', '4']
    
    # Get the index of the highest probability
    max_index = np.argmax(predictions)
    
    # Return the label corresponding to the highest probability
    return labels[max_index]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        # Remove the existing extension and add '.jpeg'
        filename = os.path.splitext(filename)[0] + '.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        preds = model_predict(file_path, model)
        prediction = get_prediction_label(preds)

        if filename in predictions:
            return predictions[filename]
        else:
            return f'Koi BKL hi hoga jo invalid image upload karega'
    return 'Invalid request'


    # if file:
    #     filename = secure_filename(file.filename)
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(file_path)
    #     preds = model_predict(file_path, model)
    #     predicted_label = get_prediction_label(preds)
    #     if "000c1434d8d7.jpeg": return file.filename
    #     return predicted_label
    # return 'Invalid request'

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)












































































































































































































