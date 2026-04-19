import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_vegetable_freshness_model.h5")

model = load_model()


fresh_classes = [0, 1, 2, 3]
medium_classes = [4, 5, 6, 7]
rotten_classes = [8, 9, 10, 11]


st.title("Fruit & Vegetable Freshness Detector")
st.write("Upload an image (even from Google) to detect freshness.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
    
        
    
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=100)

    
       
    
        img = image.resize((224, 224))  # change if your model uses different size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    
        
    
        prediction = model.predict(img_array)[0]

    
        
    
        fresh_score = sum(prediction[i] for i in fresh_classes)
        medium_score = sum(prediction[i] for i in medium_classes)
        rotten_score = sum(prediction[i] for i in rotten_classes)

    
        
    
        scores = {
            "Fresh": fresh_score,
            "Medium Fresh": medium_score,
            "Rotten": rotten_score
        }

        final_category = max(scores, key=scores.get)
        final_confidence = scores[final_category] * 100

    
       
    
        st.subheader("Prediction Result")
        st.write(f"**{final_category}**")
        st.write(f"Accuracy: {final_confidence:.2f}%")

    except Exception as e:
        st.error("Error processing image. Try another image.")
        st.write(str(e))