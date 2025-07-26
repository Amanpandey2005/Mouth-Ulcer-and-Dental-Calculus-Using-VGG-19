![ChatGPT Image Jul 26, 2025, 02_08_55 PM](https://github.com/user-attachments/assets/05cfc119-6725-4e2e-bc2b-e104e37ae7f1)🦷 Dental Calculus and Mouth Ulcer Detection Using Deep Learning (VGG19)
📌 Introduction
Oral health issues such as dental calculus (tartar) and mouth ulcers are widespread and often go undetected in early stages, leading to painful outcomes and long-term dental complications. This project introduces a deep learning-based image classification system using a fine-tuned VGG19 model to automate the detection of:

Healthy oral condition

Mouth Ulcer

Dental Calculus

The aim is to support dental professionals with a fast, accurate, and scalable solution for early screening and diagnostics using intraoral images.

🛠️ Tech Stack
Python

TensorFlow / Keras

NumPy, Matplotlib, Scikit-learn

VGG19 (Transfer Learning)

📂 Dataset
Images categorized into three classes:

Healthy

Ulcer

Calculus

Format: .jpg or .png

Organized in subfolders for training and validation:


📌 You can collect oral images from sources like Kaggle or clinical datasets. Data should be preprocessed and resized to 224x224 for VGG19.

⚙️ Installation & Setup
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/dental-detection-vgg19.git
cd dental-detection-vgg19

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🧠 Model Architecture
Base model: VGG19 (pre-trained on ImageNet)

Custom top layers:

GlobalAveragePooling2D

Dense (ReLU)

Dropout

Output layer (Softmax - 3 units)

Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

🏃‍♂️ Training
python
Copy
Edit
python train_model.py
Includes:

Image augmentation

80-20 train-validation split

Accuracy, loss, and confusion matrix tracking

📈 Results
Validation Accuracy: ~80–90% (depending on dataset quality)

Metrics Tracked: Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix

Sample output:

makefile
Copy
Edit
Healthy: 95% accuracy
Ulcer: 88% accuracy
Calculus: 86% accuracy
📊 Evaluation
Run evaluation with:

python
Copy
Edit
python evaluate_model.py
Visualizes:

Confusion Matrix

Classification Report

ROC Curve (optional)
💬 Future Improvements
Deploy as a web app using Flask or Streamlit

Expand dataset with more diverse images

Integrate real-time webcam input

Add segmentation for precise lesion/calcification area detection

🤝 Contributors
Aman Pandey – Model development, training, documentation
