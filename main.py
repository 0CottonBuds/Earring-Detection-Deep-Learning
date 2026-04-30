import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
import os

st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="centered"
)

# 19Aj-nLSCV5_i3Yr5FRr4B7tBhJFZJbw_

st.title("Earrings Detection System")
st.markdown("""
This app detects whether a motorcyclist is:
- *Male wearing earrings*
- *Male not wearing earrings*
- *Female wearing earrings*
- *Female not wearing earrings*


Trained on ~173 images across 4 classes.
""")

@st.cache_resource
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_classifier.h5")
  
# def load_model():
#     model = tf.keras.models.load_model("helmet_detectorV3.h5")
#     return model

model = load_model()

class_names = ['Male wearing earrings', 'Male not wearing earrings', 'Female wearing earrings', 'Female not wearing earrings']

tab1, tab2 = st.tabs(["📤 Upload Image", "📹 Live Camera"])

# TAB 1: file uploader 
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image of a motorcyclist",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((128, 128))
        img = img.convert('RGB') 
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class] * 100

        st.subheader("Results")
        
        predicted_label = class_names[predicted_class]
        
        if predicted_label == "Male wearing earrings":
            st.success(f"✅ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Male not wearing earrings":
            st.warning(f"⚠️ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Female wearing earrings":
            st.info(f"ℹ️ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Female not wearing earrings":
            st.warning(f"⚠️ *{predicted_label}* ({confidence:.1f}% confidence)")
        else:
            st.error(f"❌ *{predicted_label}* ({confidence:.1f}% confidence)")

        # probabilities
        st.write("### Prediction Probabilities")
        for label, prob in zip(class_names, prediction[0]):
            st.write(f"{label}: {prob * 100:.1f}%")
            st.progress(float(prob))

# TAB 2: live Camera 
with tab2:
    st.write("Point your camera at a motorcyclist to get real-time predictions!")
    
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        st.image(camera_photo, caption="Camera Capture", use_column_width=True)

        image = Image.open(camera_photo)
        img = image.resize((128, 128))
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class] * 100

        st.subheader("Results")
        
        predicted_label = class_names[predicted_class]
        
        if predicted_label == "Male wearing earrings":
            st.success(f"✅ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Male not wearing earrings":
            st.warning(f"⚠️ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Female wearing earrings":
            st.info(f"ℹ️ *{predicted_label}* ({confidence:.1f}% confidence)")
        elif predicted_label == "Female not wearing earrings":
            st.warning(f"⚠️ *{predicted_label}* ({confidence:.1f}% confidence)")
        else:
            st.error(f"❌ *{predicted_label}* ({confidence:.1f}% confidence)")

        st.write("### Prediction Probabilities")
        for label, prob in zip(class_names, prediction[0]):
            st.write(f"{label}: {prob * 100:.1f}%")
            st.progress(float(prob))