import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from twilio.rest import Client

# Load model
model = load_model(r"C:\Users\amanp\C and C++ Programs\Mouth_Ulcers\model\vgg_ulcer_model.h5")

# Class labels
class_labels = ['Healthy', 'Ulcer', 'Calculus']

# Streamlit UI
st.title("🦷 Dental Disease Detection using VGG19")
st.markdown("Upload an image of your teeth or oral cavity to detect conditions like Ulcers or Calculus.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Phone number input (optional)
phone_number = st.text_input("📱 Enter your mobile number to receive results (optional):", placeholder="+91XXXXXXXXXX")

# Only process if an image is uploaded
if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    prediction_label = class_labels[predicted_class_index]

    # Show prediction
    st.subheader(f"🔍 Prediction: {prediction_label}")

    if prediction_label == "Healthy":
        st.success("✅ Your teeth appear to be healthy!")
    elif prediction_label == "Ulcer":
        st.warning("⚠️ Mouth Ulcers detected. Consult a dentist.")
    elif prediction_label == "Calculus":
        st.error("❌ Dental Calculus detected! Visit a dentist.")

    # Twilio credentials (replace with your actual credentials)
    account_sid = "XXXXXXXXXXXXXXXXXXXXXXX"
    auth_token = "XXXXXXXXXXXXXXXXXXXXXXXX"
    twilio_number = "+XXXXXXXXXXXXXXXX"  # Your Twilio phone number

    client = Client(account_sid, auth_token)

    # Send result to phone
    if st.button("📤 Send result to phone"):
        if phone_number:
            try:
                message = client.messages.create(
                    body=f"🦷 Dental Scan Result: {prediction_label}",
                    from_=twilio_number,
                    to=phone_number
                )
                st.success("✅ Result sent to your phone!")
            except Exception as e:
                st.error(f"❌ Failed to send SMS: {e}")
        else:
            st.warning("⚠️ Please enter a valid phone number.")
