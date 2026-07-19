import numpy as np
from PIL import Image
import io
import base64
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details for the model shape
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(base64_string):
    # Decode base64 image string
    img_data = base64.b64decode(base64_string.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert('L') # Convert to grayscale
    img = img.resize((28, 28)) # Resize to MNIST standard
    
    # Normalize pixel values to [0.0, 1.0] and match shape (1, 28, 28, 1)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_img = preprocess_image(data['image'])
        
        # Feed the processed image into the TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        
        # Extract the prediction percentages
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_digit = int(np.argmax(predictions))
        
        return jsonify({'prediction': predicted_digit, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500




@app.route('/')
def home():
    return render_template('index.html')

    

if __name__=='__main__':
    app.run(debug=True)