import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix

print("Loading test data...")
# We only need the test data for evaluation
(_, _), (features_test, labels_test) = mnist.load_data()

# Preprocess it exactly the same way we did in training
features_test = features_test.reshape(features_test.shape[0], 28, 28, 1).astype('float32') / 255.0

print("Loading model...")
# Ensure this matches your actual saved model name
model = tf.keras.models.load_model("my_model.h5")

print("Generating predictions...")
# Predict all 10,000 test images
predictions = model.predict(features_test)

# Convert probabilities (like 0.99 for digit '3') into strict integer predictions
y_pred = np.argmax(predictions, axis=1)

print("\n=== CLASSIFICATION REPORT ===")
# This gives you Recall (TPR), Precision, and F1-Score for every single digit
print(classification_report(labels_test, y_pred))

print("\n=== CONFUSION MATRIX ===")
# This shows exactly which digits are being confused with which
print(confusion_matrix(labels_test, y_pred))