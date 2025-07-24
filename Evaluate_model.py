import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os

# 1️⃣ Load trained model
model = tf.keras.models.load_model("model/vgg_ulcer_model.h5")

# 2️⃣ Load saved training history
with open("train_history.pkl", "rb") as f:
    history = pickle.load(f)

# 3️⃣ Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# 4️⃣ Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Training Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# 5️⃣ Evaluate on test set (you must define the test generator same way as training)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

