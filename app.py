from flask import Flask, request, render_template
from flask import redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from utils import clean_image, get_prediction, make_results

app = Flask(__name__)

# Load Model
def load_model(path):
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(
            include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights(path)
    return model

model = load_model('model.h5')

# Homepage Route - Start Page
@app.route('/')
def start():
    return render_template('start.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('feature'))
    return render_template('index.html')

# Sign Up Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('index1.html', error="Passwords do not match.")

        return redirect(url_for('feature'))

    return render_template('index1.html')

# Feature Page - With Buttons for Detection and Prediction
@app.route('/feature')
def feature():
    return render_template('feature.html')

# Detect Page - For Image Upload & Prediction
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detect.html', result="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('detect.html', result="No file selected.")
        try:
            image = Image.open(file.stream)
        except Exception:
            return render_template('detect.html', result="Invalid image format.")
        image = clean_image(image)

        # Prediction
        predictions, predictions_arr = get_prediction(model, image)
        result = make_results(predictions, predictions_arr)

        return render_template('detect.html', result=f"The plant {result['status']} with {result['prediction']} prediction.")
    
    return render_template('detect.html', result="")

# Crop yield prediction
import pickle
import sklearn
print(sklearn.__version__)

# Loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    result = ""
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        result = prediction[0][0]

    return render_template('predict.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
