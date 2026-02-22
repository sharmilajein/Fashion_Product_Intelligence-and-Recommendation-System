# Fashion_Product_Intelligence-and-Recommendation-System
This project is an end-to-end Fashion Product Intelligence and Recommendation System built using Deep Learning and Computer Vision. The goal is to analyze real-world e-commerce fashion data and recommend visually similar products based on images.

The project starts with flattening complex JSON product data, followed by data cleaning and Exploratory Data Analysis (EDA) to understand trends across gender, category, color, season, and brand. Based on EDA insights, a focused set of fashion categories is selected for modeling.

A transfer learning–based CNN model is trained to classify fashion product images into product types. The trained model is then used as a feature extractor to generate image embeddings. Using cosine similarity, the system recommends visually similar products from the catalog.

Finally, a Streamlit web application integrates EDA insights, image classification, and similarity-based recommendations into an interactive user interface where users can upload an image and receive product recommendations.


