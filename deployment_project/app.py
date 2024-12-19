from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging

# Load the model
model_final = tf.keras.models.load_model('best_model.h5')

# Class mapping
class_mapping = {
    0: 'Amrita',
    1: 'Krishna',
    2: 'Purple',
    3: 'Rama',
    4: 'Sweet'
}

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_filename = None
    confidence = None  # Initialize confidence

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")

        file = request.files['file']
        logging.info("File received: %s", file.filename)

        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

                prediction = model_final.predict(img_array)
                predicted_class_index = np.argmax(prediction, axis=1)[0]

                # Confidence code 
                confidence = np.max(prediction[0]) * 100
                confidence = round(confidence, 2)  # round off to two decimal places

                predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")
                image_filename = filename

                # Optional: Delete the uploaded file after processing
                # os.remove(file_path)

                return render_template('index.html', prediction=predicted_class_name, image=image_filename, confidence=confidence)

            except Exception as e:
                logging.error("Error during processing: %s", str(e))
                return render_template('index.html', prediction=f"Error during processing: {str(e)}")

    return render_template('index.html', prediction=prediction, image=image_filename, confidence=confidence)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
