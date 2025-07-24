from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your model
model = load_model(r"C:\Users\amanp\C and C++ Programs\Mouth_Ulcers\model\vgg_ulcer_model.h5")
print("✅ Model loaded.")

# Setup test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_dir = r"C:\Users\amanp\C and C++ Programs\Mouth_Ulcers\Test_images"

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Should match your model input
    batch_size=32,
    class_mode='categorical',  # or 'binary' if only 2 classes
    shuffle=False
)


print("✅ Found", test_generator.samples, "test images.")
true_classes = test_generator.classes
predicted_classes = np.argmax(model.predict(test_generator), axis=1)
class_labels = list(test_generator.class_indices.keys())
print("✅ Predictions made on test set.")
# Classification report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()  # <-- Make sure this is here
plt.savefig("confusion_matrix.png")
print("📸 Confusion matrix saved as confusion_matrix.png")