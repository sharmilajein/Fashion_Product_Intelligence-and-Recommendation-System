import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# --------------------
# BASIC SETTINGS
# --------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # start small, fast

# --------------------
# DATA GENERATOR
# --------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset_clean",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset_clean",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("Class mapping:", train_data.class_indices)

# --------------------
# LOAD PRETRAINED MODEL
# --------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# --------------------
# CUSTOM CLASSIFICATION HEAD
# --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# --------------------
# COMPILE MODEL
# --------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------
# TRAIN MODEL
# --------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# --------------------
# SAVE MODEL
# --------------------
model.save("classifier_model_clean.h5")
print("✅ Model saved as classifier_model_clean.h5")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset_clean",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Class mapping:", data.class_indices)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset_clean",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Class mapping:", data.class_indices)





# # Verification 2 To Save Mapping

# # prediction = model.predict(image)
# # predicted_index = argmax(prediction)
# # label = index_to_class[predicted_index]



# import json

# with open("class_mapping.json", "w") as f:
#     json.dump(train_data.class_indices, f)

# print("✅ class_mapping.json saved")

