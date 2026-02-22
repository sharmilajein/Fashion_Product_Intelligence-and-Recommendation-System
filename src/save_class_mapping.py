import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset_clean",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

class_mapping = data.class_indices
print("Class mapping:", class_mapping)

with open("class_mapping.json", "w") as f:
    json.dump(class_mapping, f)

print("✅ class_mapping.json saved successfully")
