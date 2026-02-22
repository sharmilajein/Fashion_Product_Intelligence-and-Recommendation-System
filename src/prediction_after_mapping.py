import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("/Users/jein/env/env/Fashion_product_intelligence/model/classifier_model_clean.h5")

# Load class mapping
with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Reverse mapping
index_to_class = {v: k for k, v in class_mapping.items()}

# Load test image
img_path = "/Users/jein/env/env/Fashion_product_intelligence/dataset_clean/Tshirts/12536.jpg"  # change to any image you have

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_index = np.argmax(pred)
pred_label = index_to_class[pred_index]

print("Predicted class:", pred_label)
