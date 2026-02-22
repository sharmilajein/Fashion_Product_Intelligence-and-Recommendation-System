# import os
# import requests
# import pandas as pd

# df = pd.read_csv("/Users/jein/env/env/Fashion_product_intelligence/data/model_dataset_reduced.csv")

# BASE_DIR = "dataset_clean"
# os.makedirs(BASE_DIR, exist_ok=True)

# downloaded = 0

# for _, row in df.iterrows():

#     label = row["articleType_clean"]
#     image_url = row["imageURL"]
#     product_id = str(row["id"])

#     class_dir = os.path.join(BASE_DIR, label)
#     os.makedirs(class_dir, exist_ok=True)

#     image_path = os.path.join(class_dir, product_id + ".jpg")

#     if os.path.exists(image_path):
#         continue

#     try:
#         r = requests.get(image_url, timeout=5)
#         if r.status_code == 200:
#             with open(image_path, "wb") as f:
#                 f.write(r.content)
#             downloaded += 1

#             if downloaded % 50 == 0:
#                 print("Downloaded", downloaded)

#     except:
#         continue

# print("\n✅ Image download completed")
# print("Total images downloaded:", downloaded)


# Verification

import os

for folder in os.listdir("dataset_clean"):
    count = len(os.listdir(os.path.join("dataset_clean", folder)))
    print(folder, "->", count)

