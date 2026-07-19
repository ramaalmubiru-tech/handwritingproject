import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('my_model.h5')

# Convert it to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the lightweight model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! Your model is now lightweight.")