import streamlit as st

st.title("👕 Fashion Product Intelligence System")

st.markdown("""
### 📌 Project Overview
This project builds an **end-to-end fashion product intelligence and recommendation system**
using **Deep Learning and Computer Vision**.

The system:
- Understands complex fashion catalog data
- Classifies fashion product images
- Recommends visually similar products

---

### 📊 Dataset Description
- Product metadata from real-world e-commerce JSON files
- Product images downloaded using image URLs
- Multiple fashion categories (clothing & footwear)

---

### High-Level Architecture 
###  Technologies Used
- Python
- TensorFlow / Keras
- Transfer Learning (MobileNetV2)
- Cosine Similarity
- Streamlit

### Work Flow ###
User Image Upload
↓
Image Classification (CNN - Transfer Learning)
↓
Image Embedding Extraction
↓
Cosine Similarity Search
↓
Top-N Similar Product Recommendations

""")
