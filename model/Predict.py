import numpy as np
import cv2
import tensorflow as tf
import os

# ✅ Load trained model (change name as per your model)
model = tf.keras.models.load_model('model/vgg_ulcer_model.h5')  # Update path if needed

# ✅ Class names in the correct order (must match training order)
class_names = ['Calculus', 'Healthy', 'Ulcer']

def predict_image(image_path):
    print(f"📸 Loading: {image_path}")
    
    if not os.path.exists(image_path):
        print("❌ File not found. Please check the path.")
        return

    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load the image. Check format or corruption.")
        return
    
    # ✅ Resize and normalize
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if trained that way
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ✅ Predict
    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    predicted_class = class_names[class_index]
    print(f"✅ Predicted: {predicted_class} ({confidence * 100:.2f}%)")

# ✅ Test the prediction with sample image
test_image_path = 'Test_images/1.jpg'  # Make sure this path exists
predict_image(test_image_path)
