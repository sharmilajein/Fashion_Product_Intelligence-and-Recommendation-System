import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load trained classification model
model = tf.keras.models.load_model("/Users/jein/env/env/Fashion_product_intelligence/model/classifier_model_clean.h5")

# Remove final classification layer
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-2].output  # Dense(128)
)

print("✅ Feature extractor ready")

IMAGE_SIZE = (224, 224)

embeddings = []
image_paths = []

BASE_DIR = "/Users/jein/env/env/Fashion_product_intelligence/dataset_clean"

for class_name in os.listdir(BASE_DIR):
    class_path = os.path.join(BASE_DIR, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            embedding = feature_extractor.predict(img_array, verbose=0)

            embeddings.append(embedding.flatten())
            image_paths.append(img_path)

        except:
            continue

embeddings = np.array(embeddings)

print("Total embeddings created:", embeddings.shape[0])

# Save embeddings
with open("image_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, image_paths), f)

print("✅ Embeddings saved as image_embeddings.pkl")
