import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ear_extraction import extract_ear_region

st.set_page_config(
    page_title="Earrings Detection",
    layout="centered"
)

# 19Aj-nLSCV5_i3Yr5FRr4B7tBhJFZJbw_

st.title("Earrings Detection System")
st.markdown("""
This app detects whether a earring is:
- *Male wearing earrings*
- *Male not wearing earrings*
- *Female wearing earrings*
- *Female not wearing earrings*

""")

@st.cache_resource
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_finetune.h5")
    # return tf.keras.models.load_model("cnn_classifier.h5")
  
# def load_model():
#     model = tf.keras.models.load_model("helmet_detectorV3.h5")
#     return model

model = load_model()

class_names = ['Male wearing earrings', 'Male not wearing earrings', 'Female wearing earrings', 'Female not wearing earrings']

tab1, tab2 = st.tabs(["📤 Upload Image", "📹 Live Camera"])


def prepare_model_input(image: Image.Image) -> np.ndarray:
    ear_crop = extract_ear_region(image, target_size=(128, 128))
    ear_array = np.array(ear_crop, dtype=np.float32) / 255.0
    return np.expand_dims(ear_array, axis=0)


def show_extraction_preview(image: Image.Image) -> None:
    st.write("### Ear Extraction Preview")
    original_column, extracted_column = st.columns(2)

    with original_column:
        st.image(image, caption="Original image", use_column_width=True)

    with extracted_column:
        extracted = extract_ear_region(image, target_size=(256, 256)).astype(np.uint8)
        st.image(extracted, caption="Extracted ear region", use_column_width=True)
        st.caption("This is the cropped region used for prediction.")

# TAB 1: file uploader 
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image of a earring",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        show_extraction_preview(image)

        img_array = prepare_model_input(image)

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
    st.write("Point your camera at a earring to get real-time predictions!")
    
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        image = Image.open(camera_photo)
        show_extraction_preview(image)

        img_array = prepare_model_input(image)

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