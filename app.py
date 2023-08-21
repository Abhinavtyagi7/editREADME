from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        image = Image.open(image_file)
        
        # Preprocess the image
        image = image.resize((544, 544))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make a prediction
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=-1)
        
        # Post-process the output
        prediction = prediction[0]  # Get the first (and only) prediction result
        atherosclerosis_percentage = np.mean(prediction == 1) * 1000  # Calculate the percentage of pixels labeled as class 1 (atherosclerosis)
        
        # Return the prediction result
        return jsonify({'prediction': f"There are {atherosclerosis_percentage:.2f}% chances of Atherosclerosis."})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

