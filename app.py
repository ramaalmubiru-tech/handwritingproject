import os
import base64
import keras
from flask import Flask, request, jsonify,render_template
import tensorflow as tf
import io
import numpy as np
from PIL import Image

app=Flask(__name__)

MODEL_PATH ="my_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_b64):


    image_data=base64.b64decode(image_b64.split(',')[1])


    img=Image.open(io.BytesIO(image_data))


    img=img.convert('L').resize((28,28))

    img_array=np.array(img)


    img_array=img_array.astype('float32')/255.0


    img_array=img_array.reshape(1,28,28,1)

    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()


    if not data or 'image' not in data:
        return jsonify({'error':'No image data provided'}),400
    
    try:
        processed_img=preprocess_image(data['image'])
        

        debug_img_array=(processed_img[0, :, :, 0]*255).astype(np.uint8)
        Image.fromarray(debug_img_array).save('debug_view.png')
        predictions=model.predict(processed_img)
        predicted_digit=int(np.argmax(predictions[0]))
        confidence=float(np.max(predictions[0]))

        return jsonify({
            'prediction':predicted_digit,
            'confidence':confidence
        })
    except Exception as e:
        return jsonify({'error':str(e)}),500
    

if __name__=='__main__':
    app.run(debug=True)