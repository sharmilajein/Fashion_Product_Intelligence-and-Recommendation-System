import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD MODEL & FILES
# -----------------------------
@st.cache_resource
def load_model_and_data():
    model = tf.keras.models.load_model("/Users/jein/env/env/Fashion_product_intelligence/model/classifier_model_clean.h5")

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    with open("/Users/jein/env/env/Fashion_product_intelligence/model/image_embeddings.pkl", "rb") as f:
        embeddings, image_paths = pickle.load(f)

    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )

    index_to_class = {v: k for k, v in class_mapping.items()}

    return model, feature_extractor, embeddings, image_paths, index_to_class


model, feature_extractor, embeddings, image_paths, index_to_class = load_model_and_data()

# -----------------------------
# UI
# -----------------------------
st.title("👕 Image-Based Fashion Recommendation")
st.write("Upload a fashion image to get visually similar product recommendations.")

uploaded_file = st.file_uploader(
    "Upload a fashion product image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Show uploaded image
    st.image(img, caption="Uploaded Image", width=250)

    # Predict category
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_label = index_to_class[pred_index]

    st.subheader(f"✅ Predicted Category: **{pred_label}**")

    # Extract embedding
    query_embedding = feature_extractor.predict(img_array, verbose=0)

    # Compute similarity
    similarity = cosine_similarity(query_embedding, embeddings)[0]

    # Filter by same predicted category
    filtered_indices = [
        i for i, path in enumerate(image_paths)
        if pred_label in path
    ]

    filtered_similarities = similarity[filtered_indices]

    top_indices = np.argsort(filtered_similarities)[-5:][::-1]

    st.subheader("🔍 Top-5 Recommended Similar Products")

    cols = st.columns(5)

    for col, idx in zip(cols, top_indices):
        img_path = image_paths[filtered_indices[idx]]
        rec_img = image.load_img(img_path, target_size=(224, 224))
        col.image(rec_img)