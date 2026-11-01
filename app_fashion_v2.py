from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "/Users/jein/env/env/Fashion_product_v2/src/efficientnetb0_best.keras"
EMBEDDINGS_PATH = BASE_DIR / "/Users/jein/env/env/Fashion_product_v2/src/women_product_embeddings_v2.pkl"
LABEL_MAPPING_PATH = BASE_DIR / "/Users/jein/env/env/Fashion_product_v2/src/article_type_label_mapping_v2.csv"
IMAGE_DIR = BASE_DIR / "/Users/jein/env/env/Fashion_product_v2/src/images"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fashion Product Intelligence",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def load_classification_model():

    model = load_model(MODEL_PATH)

    return model


# ============================================================
# LOAD EMBEDDINGS + METADATA
# ============================================================

@st.cache_resource
def load_embedding_data():

    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    metadata = data["metadata"]

    return embeddings, metadata


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

@st.cache_data
def load_label_mapping():

    mapping = pd.read_csv(LABEL_MAPPING_PATH)

    return mapping


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def create_feature_extractor(model):

    # EfficientNetB0 architecture:
    # efficientnetb0
    # global_average_pooling2d
    # dropout
    # dense

    feature_extractor = Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
    )

    return feature_extractor


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_uploaded_image(uploaded_file):

    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    return img_array


# ============================================================
# CLASSIFICATION
# ============================================================

def predict_article_type(
    image_array,
    model,
    label_mapping
):

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_label = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        np.max(predictions[0])
    )

    predicted_row = label_mapping[
        label_mapping["label"] == predicted_label
    ]

    if predicted_row.empty:

        return "Unknown", confidence

    predicted_article_type = (
        predicted_row["articleTypeName"].iloc[0]
    )

    return predicted_article_type, confidence


# ============================================================
# GENERATE IMAGE EMBEDDING
# ============================================================

def generate_embedding(
    image_array,
    feature_extractor
):

    embedding = feature_extractor.predict(
        image_array,
        verbose=0
    )

    embedding = embedding.flatten()

    return embedding


# ============================================================
# RECOMMEND SIMILAR PRODUCTS
# ============================================================

def recommend_products(
    query_embedding,
    predicted_category,
    embeddings,
    metadata,
    top_n=5
):

    # Filter products belonging to predicted category

    category_mask = (
        metadata["articleTypeName"]
        == predicted_category
    )

    category_metadata = metadata[
        category_mask
    ].copy()

    category_indices = np.where(
        category_mask.values
    )[0]

    if len(category_indices) == 0:

        return pd.DataFrame()

    category_embeddings = embeddings[
        category_indices
    ]

    # Cosine similarity

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        category_embeddings
    )[0]

    # Add similarity score

    category_metadata["similarity"] = similarities

    # Sort highest similarity first

    recommendations = (
        category_metadata
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    return recommendations


# ============================================================
# DISPLAY PRODUCT IMAGE
# ============================================================

def get_image_path(relative_path):

    filename = Path(
        str(relative_path)
    ).name

    image_path = IMAGE_DIR / filename

    return image_path


# ============================================================
# LOAD DATA
# ============================================================

model = load_classification_model()

embeddings, metadata = load_embedding_data()

label_mapping = load_label_mapping()

feature_extractor = create_feature_extractor(model)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👗 Fashion Product Intelligence")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 EDA Insights",
        "🛍️ Image Recommendation"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("👗 Fashion Product Intelligence")

    st.subheader(
        "AI-Powered Fashion Classification & Recommendation"
    )

    st.markdown(
        """
        ### Project Overview

        This project uses **Deep Learning and Computer Vision**
        to classify fashion products and recommend visually
        similar products.

        ### 🔍 How it works

        **Upload an image → Classify the product → Extract visual
        features → Calculate similarity → Recommend similar products**
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Product Images",
            f"{len(metadata):,}"
        )

    with col2:
        st.metric(
            "Classifier Classes",
            f"{len(label_mapping)}"
        )

    with col3:
        st.metric(
            "Test Accuracy",
            "85.92%"
        )

    st.markdown("---")

    st.subheader("Technology Stack")

    st.markdown(
        """
        - Python
        - TensorFlow / Keras
        - EfficientNetB0
        - Computer Vision
        - Transfer Learning
        - Cosine Similarity
        - Pandas / NumPy
        - Streamlit
        """
    )


# ============================================================
# EDA PAGE
# ============================================================

elif page == "📊 EDA Insights":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Dataset overview and product distribution."
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Products",
            f"{len(metadata):,}"
        )

    with col2:
        st.metric(
            "Catologue Article Types",
            metadata["articleTypeName"].nunique()
        )

    with col3:
        st.metric(
            "Brands",
            metadata["brandName"].nunique()
        )

    with col4:
        st.metric(
            "Colours",
            metadata["baseColour"].nunique()
        )

    st.markdown("---")

    # --------------------------------------------------------
    # 1. ARTICLE TYPE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "1️⃣ Product Category Distribution"
    )

    category_counts = (
        metadata["articleTypeName"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(category_counts)

    # --------------------------------------------------------
    # 2. MASTER CATEGORY
    # --------------------------------------------------------

    st.subheader(
        "2️⃣ Master Category Distribution"
    )

    master_counts = (
        metadata["masterCategoryName"]
        .value_counts()
    )

    st.bar_chart(master_counts)

    # --------------------------------------------------------
    # 3. SUB CATEGORY
    # --------------------------------------------------------

    st.subheader(
        "3️⃣ Sub-Category Distribution"
    )

    subcategory_counts = (
        metadata["subCategoryName"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(subcategory_counts)

    # --------------------------------------------------------
    # 4. TOP BRANDS
    # --------------------------------------------------------

    st.subheader(
        "4️⃣ Top 15 Brands"
    )

    brand_counts = (
        metadata["brandName"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(brand_counts)

    # --------------------------------------------------------
    # 5. COLOUR DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "5️⃣ Popular Product Colours"
    )

    colour_counts = (
        metadata["baseColour"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(colour_counts)

    # --------------------------------------------------------
    # 6. SEASON DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "6️⃣ Products by Season"
    )

    season_counts = (
        metadata["season"]
        .value_counts()
    )

    st.bar_chart(season_counts)

    # --------------------------------------------------------
    # DATASET SCOPE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📌 Dataset Scope")

    st.info(
        """
        The recommendation dataset focuses on **Women fashion
        products**.

        This focused subset contains 18,630 product images across
        54 article types. Restricting the recommendation system to
        a single product domain helps provide more meaningful
        category-specific visual recommendations.
        """
    )


# ============================================================
# IMAGE RECOMMENDATION PAGE
# ============================================================

elif page == "🛍️ Image Recommendation":

    st.title(
        "🛍️ Image-Based Product Recommendation"
    )

    st.write(
        "Upload a fashion product image to identify "
        "its category and find visually similar products."
    )

    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        st.markdown("---")

        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        image_array = preprocess_uploaded_image(
            uploaded_file
        )

        # ----------------------------------------------------
        # DISPLAY UPLOADED IMAGE
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [1, 2]
        )

        with col1:

            st.image(
                uploaded_file,
                caption="Uploaded Image",
                use_container_width=True
            )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        predicted_category, confidence = (
            predict_article_type(
                image_array,
                model,
                label_mapping
            )
        )

        with col2:

            st.subheader(
                "Classification Result"
            )

            st.success(
                f"Predicted Category: "
                f"{predicted_category}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ----------------------------------------------------
        # GENERATE QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = generate_embedding(
            image_array,
            feature_extractor
        )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = recommend_products(
            query_embedding,
            predicted_category,
            embeddings,
            metadata,
            top_n=5
        )

        st.markdown("---")

        st.subheader(
            "🛍️ Top 5 Similar Products"
        )

        if recommendations.empty:

            st.warning(
                "No products found for this category."
            )

        else:

            cols = st.columns(5)

            for col, (_, row) in zip(
                cols,
                recommendations.iterrows()
            ):

                with col:

                    image_path = get_image_path(
                        row["image_path"]
                    )

                    # ----------------------------------------
                    # PRODUCT IMAGE
                    # ----------------------------------------

                    if image_path.exists():

                        st.image(
                            str(image_path),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            "Image unavailable"
                        )

                    # ----------------------------------------
                    # PRODUCT INFORMATION
                    # ----------------------------------------

                    st.write(
                        f"**{row['productDisplayName']}**"
                    )

                    st.caption(
                        f"Brand: {row['brandName']}"
                    )

                    st.caption(
                        f"Category: "
                        f"{row['articleTypeName']}"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{row['similarity']:.3f}"
                    )

# # ============================================================
# # HOME PAGE
# # ============================================================

# if page == "🏠 Home":

#     st.title("👗 Fashion Product Intelligence")

#     st.subheader(
#         "AI-Powered Fashion Classification & Recommendation"
#     )

#     st.markdown(
#         """
#         ### Project Overview

#         This project uses **Deep Learning and Computer Vision**
#         to classify fashion products and recommend visually
#         similar products.

#         ### 🔍 How it works

#         **Upload an image → Classify the product → Extract visual
#         features → Calculate similarity → Recommend similar products**
#         """
#     )

#     col1, col2, col3 = st.columns(3)

#     with col1:

#         st.metric(
#             "Product Images",
#             "18,630"
#         )

#     with col2:

#         st.metric(
#             "Classes",
#             "54"
#         )

#     with col3:

#         st.metric(
#             "Test Accuracy",
#             "85.92%"
#         )

#     st.markdown("---")

#     st.subheader("Technology Stack")

#     st.markdown(
#         """
#         - Python
#         - TensorFlow / Keras
#         - EfficientNetB0
#         - Computer Vision
#         - Transfer Learning
#         - Cosine Similarity
#         - Pandas / NumPy
#         - Streamlit
#         """
#     )


# # ============================================================
# # EDA PAGE
# # ============================================================

# elif page == "📊 EDA Insights":

#     st.title("📊 Exploratory Data Analysis")

#     st.write(
#         "Dataset overview and product distribution."
#     )
    
#     col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric(
#         "Total Products",
#         f"{len(metadata):,}"
#     )

# with col2:
#     st.metric(
#         "Article Types",
#         metadata["articleTypeName"].nunique()
#     )

# with col3:
#     st.metric(
#         "Brands",
#         metadata["brandName"].nunique()
#     )

# with col4:
#     st.metric(
#         "Colours",
#         metadata["baseColour"].nunique()
#     )
    
#     # Product Category Distribution
    
#     st.subheader("Product Category Distribution")

# category_counts = (
#     metadata["articleTypeName"]
#     .value_counts()
#     .head(15)
# )

# st.bar_chart(category_counts)

# # Master Category Distribution

# st.subheader("Master Category Distribution")

# master_counts = (
#     metadata["masterCategoryName"]
#     .value_counts()
# )

# st.bar_chart(master_counts)

# # Sub-Category Distribution

# st.subheader("Sub-Category Distribution")

# subcategory_counts = (
#     metadata["subCategoryName"]
#     .value_counts()
#     .head(15)
# )

# st.bar_chart(subcategory_counts)

# # Top Brands

# st.subheader("Top 15 Brands")

# brand_counts = (
#     metadata["brandName"]
#     .value_counts()
#     .head(15)
# )

# st.bar_chart(brand_counts)

# # . Base Colour Distribution
# st.subheader("Popular Product Colours")

# colour_counts = (
#     metadata["baseColour"]
#     .value_counts()
#     .head(15)
# )

# st.bar_chart(colour_counts)

# # Season Distribution

# st.subheader("Products by Season")

# season_counts = (
#     metadata["season"]
#     .value_counts()
# )

# st.bar_chart(season_counts)

# #     col1, col2, col3 = st.columns(3)

# #     with col1:

# #         st.metric(
# #             "Total Products",
# #             f"{len(metadata):,}"
# #         )

# #     with col2:

# #         st.metric(
# #             "Categories",
# #             metadata[
# #                 "articleTypeName"
# #             ].nunique()
# #         )

# #     with col3:

# #         st.metric(
# #             "Brands",
# #             metadata[
# #                 "brandName"
# #             ].nunique()
# #         )

# #     st.markdown("---")

# #     st.subheader(
# #         "Top Product Categories"
# #     )

# #     category_counts = (
# #         metadata[
# #             "articleTypeName"
# #         ]
# #         .value_counts()
# #         .head(15)
# #     )

# #     st.bar_chart(category_counts)

# #     st.subheader(
# #         "Gender Distribution"
# #     )

# #     gender_counts = (
# #         metadata[
# #             "gender"
# #         ]
# #         .value_counts()
# #     )

# #     st.bar_chart(gender_counts)




# # ============================================================
# # IMAGE RECOMMENDATION PAGE
# # ============================================================

#     elif page == "🛍️ Image Recommendation":
    
#     st.title(
#         "🛍️ Image-Based Product Recommendation"
#     )

#     st.write(
#         "Upload a fashion product image to identify "
#         "its category and find visually similar products."
#         )

#     uploaded_file = st.file_uploader(
#         "Upload Product Image",
#         type=[
#             "jpg",
#             "jpeg",
#             "png"
#         ]
#     )

#     if uploaded_file is not None:

#         st.markdown("---")

#         # ----------------------------------------------------
#         # PREPROCESS IMAGE
#         # ----------------------------------------------------

#         image_array = preprocess_uploaded_image(
#             uploaded_file
#         )

#         # ----------------------------------------------------
#         # DISPLAY UPLOADED IMAGE
#         # ----------------------------------------------------

#         col1, col2 = st.columns(
#             [1, 2]
#         )

#         with col1:

#             st.image(
#                 uploaded_file,
#                 caption="Uploaded Image",
#                 use_container_width=True
#             )

#         # ----------------------------------------------------
#         # CLASSIFICATION
#         # ----------------------------------------------------

#         predicted_category, confidence = (
#             predict_article_type(
#                 image_array,
#                 model,
#                 label_mapping
#             )
#         )

#         with col2:

#             st.subheader(
#                 "Classification Result"
#             )

#             st.success(
#                 f"Predicted Category: "
#                 f"{predicted_category}"
#             )

#             st.metric(
#                 "Confidence",
#                 f"{confidence * 100:.2f}%"
#             )

#         # ----------------------------------------------------
#         # GENERATE QUERY EMBEDDING
#         # ----------------------------------------------------

#         query_embedding = generate_embedding(
#             image_array,
#             feature_extractor
#         )

#         # ----------------------------------------------------
#         # RECOMMENDATIONS
#         # ----------------------------------------------------

#         recommendations = recommend_products(
#             query_embedding,
#             predicted_category,
#             embeddings,
#             metadata,
#             top_n=5
#         )

#         st.markdown("---")

#         st.subheader(
#             "🛍️ Top 5 Similar Products"
#         )

#         if recommendations.empty:

#             st.warning(
#                 "No products found for this category."
#             )

#         else:

#             cols = st.columns(5)

#             for col, (_, row) in zip(
#                 cols,
#                 recommendations.iterrows()
#             ):

#                 with col:

#                     image_path = get_image_path(
#                         row["image_path"]
#                     )

#                     if image_path.exists():

#                         st.image(
#                             str(image_path),
#                             use_container_width=True
#                         )

#                     else:

#                         st.warning(
#                             "Image unavailable"
#                         )

#                     st.write(
#                         f"**{row['productDisplayName']}**"
#                     )

#                     st.caption(
#                         f"Brand: {row['brandName']}"
#                     )

#                     st.caption(
#                         f"Category: "
#                         f"{row['articleTypeName']}"
#                     )

#                     st.caption(
#                         f"Similarity: "
#                         f"{row['similarity']:.3f}"
#                     )