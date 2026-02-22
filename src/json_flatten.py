import os
import json
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
json_folder = "/Users/jein/Downloads/styles"
image_csv_path = "/Users/jein/Downloads/images.csv"

# This list will store all product rows
product_list = []

# -----------------------------
# Read JSON files one by one
# -----------------------------
files = os.listdir(json_folder)

print("Total JSON files found:", len(files))

for file_name in files:
    if file_name.endswith(".json"):
        file_path = os.path.join(json_folder, file_name)

        try:
            with open(file_path, "r") as file:
                json_data = json.load(file)

            # Actual product data is inside "data"
            product = json_data.get("data", {})

            # Create one row
            row = {
                "id": str(product.get("id")),
                "productDisplayName": product.get("productDisplayName"),
                "brandName": product.get("brandName"),
                "gender": product.get("gender"),
                "masterCategory": product.get("masterCategory"),
                "subCategory": product.get("subCategory"),
                "articleType": product.get("articleType"),
                "baseColour": product.get("baseColour"),
                "season": product.get("season"),
                "year": product.get("year"),
                "usage": product.get("usage")
            }

            product_list.append(row)

        except:
            print("❌ Error reading file:", file_name)

# -----------------------------
# Convert list → DataFrame
# -----------------------------
df_products = pd.DataFrame(product_list)

print("Metadata rows:", df_products.shape)

# -----------------------------
# Load image mapping CSV
# -----------------------------
df_images = pd.read_csv(image_csv_path)

# filename = "12345.jpg" → id = "12345"
df_images["id"] = df_images["filename"].str.replace(".jpg", "", regex=False)

# -----------------------------
# Merge metadata + images
# -----------------------------
df_final = pd.merge(
    df_products,
    df_images,
    on="id",
    how="left"
)

# Rename column
df_final.rename(columns={"link": "imageURL"}, inplace=True)

# -----------------------------
# Save final dataset
# -----------------------------
df_final.to_csv("fashion_products.csv", index=False)

print("✅ Step 1 completed successfully")
print("Final dataset shape:", df_final.shape)
print(df_final.head())
