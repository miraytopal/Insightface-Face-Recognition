import streamlit as st
import numpy as np
from PIL import Image
import insightface
from model import process_images
from insightface.app import FaceAnalysis

@st.cache_resource
def detector_model():
    detector = insightface.model_zoo.get_model('model.onnx', download=True)
    return detector

@st.cache_resource
def app_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    return app


st.title("Face Recognition App")

col1, col2 = st.columns(2)

with col1:

    file_1 = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif"],key='file_1')

    if file_1:
        st.image(file_1)

with col2:

    file_2 = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif"],key='file_2')

    if file_2:
        st.image(file_2)


if st.button("Predict"):

    if not (file_1 and file_2):
        st.error("Please upload two image files")

    else:
        # File to numpy array
        image_1 = np.asarray(Image.open(file_1))
        image_2 = np.asarray(Image.open(file_2))

        score = process_images(image_1, image_2, detector_model(), app_model())

        st.metric("Similarity", f"{score:.2f}")



