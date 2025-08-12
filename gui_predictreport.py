import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from twilio.rest import Client
import io

# Load model
model = load_model(r"C:\Users\amanp\C and C++ Programs\Mouth_Ulcers\model\vgg_ulcer_model.h5")

# Class labels
class_labels = ['Calculus', 'Healthy', 'Ulcer']

# Streamlit UI
st.title("🦷 Dental Disease Detection using VGG19")
st.markdown("Upload an image of your teeth or oral cavity to detect conditions like Ulcers or Calculus.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

# Phone number input (optional)
phone_number = st.text_input("📱 Enter your mobile number to receive results (optional):", placeholder="+91XXXXXXXXXX")

if uploaded_file is not None:
    try:
        # ✅ Load image using PIL
        pil_image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img = pil_image.resize((224, 224))
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

        # Twilio credentials (use environment variables or secrets in production)
        account_sid = "AC16bea1596a7d82c5a7093c03efe24801"
        auth_token = "d8f90c3bb9447871ae77a7c236becd36"
        twilio_number = "+12314121727"

        # Send SMS on button click
        if st.button("📤 Send result to phone"):
            if phone_number:
                try:
                    client = Client(account_sid, auth_token)
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

    except Exception as e:
        st.error(f"❌ Error: Could not process the image.\n\n{e}")
