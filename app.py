from flask import Flask, request, render_template

import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

import os



app = Flask(__name__)



# Load the trained model


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'cats_vs_dogs_model.h5'))

@app.route('/', methods=['GET', 'POST'])

def index():

    prediction = None

    if request.method == 'POST':

        # Get the uploaded image

        img_file = request.files['image']

        img_path = os.path.join('static', img_file.filename)

        img_file.save(img_path)


        # Prepare image for prediction

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0


        # Make prediction

        result = model.predict(img_array)

        prediction = "Dog 🐶" if result[0][0] > 0.5 else "Cat 🐱"


    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':

    app.run(debug=True)