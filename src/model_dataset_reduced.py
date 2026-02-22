import pandas as pd

# Load dataset after EDA
df = pd.read_csv("/Users/jein/env/env/Fashion_product_intelligence/model/fashion_products_clean.csv")

print("Original dataset size:", df.shape)

# def normalize_article_type(x):
#     x = str(x).lower()

#     if "tshirt" in x:
#         return "Tshirts"
#     elif "shirt" in x:
#         return "Shirts"
#     elif "casual_shoe" in x:
#         return "Casual_Shoes"
#     elif "sports_shoe" in x:
#         return "Sports_Shoes"
#     elif "jean" in x:
#         return "Jeans"
#     else:
#         return "Other"

# df["articleType_clean"] = df["articleType"].apply(normalize_article_type)

# print("\nNormalized distribution:")
# print(df["articleType_clean"].value_counts())

# # Save intermediate file
# df.to_csv("fashion_products_normalized.csv", index=False)

# print("\n✅ fashion_products_normalized.csv saved")


# import pandas as pd

# # df = pd.read_csv("fashion_products_clean.csv")

# # # Load dataset after EDA
# df = pd.read_csv("/Users/jein/env/env/Fashion_product_intelligence/model/fashion_products_clean.csv")

# print(df["articleType"].value_counts().head(20))


import pandas as pd

# df = pd.read_csv("fashion_products_clean.csv")

print("Original dataset size:", df.shape)

def normalize_article_type(x):
    x = str(x).lower()

    # T-Shirts
    if "tshirt" in x or "t-shirt" in x:
        return "Tshirts"

    # Shirts
    elif "shirt" in x:
        return "Shirts"

    # Jeans
    elif "jean" in x:
        return "Jeans"

    # Shoes (catch ALL shoe types)
    elif (
        "shoe" in x or
        "sandal" in x or
        "flip_flop" in x or
        "flip flop" in x
    ):
        # Optional: split casual vs sports
        if "sport" in x:
            return "Sports_Shoes"
        else:
            return "Casual_Shoes"

    else:
        return "Other"

df["articleType_clean"] = df["articleType"].apply(normalize_article_type)

print("\nNormalized distribution:")
print(df["articleType_clean"].value_counts())

df.to_csv("fashion_products_normalized.csv", index=False)

print("\n✅ fashion_products_normalized.csv saved")

