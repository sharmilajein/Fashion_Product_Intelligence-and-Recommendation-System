# import pandas as pd

# df = pd.read_csv("/Users/jein/env/env/Fashion_product_intelligence/model/fashion_products_normalized.csv")

# KEEP_CLASSES = [
#     "Tshirts",
#     "Shirts",
#     "Casual_Shoes",
#     "Sports_Shoes",
#     "Jeans"
# ]

# df_reduced = df[df["articleType_clean"].isin(KEEP_CLASSES)]

# print("Reduced dataset size:", df_reduced.shape)
# print("\nFinal class distribution:")
# print(df_reduced["articleType_clean"].value_counts())

# df_reduced.to_csv("model_dataset_reduced.csv", index=False)

# print("\n✅ model_dataset_reduced.csv created")



# Verification

import pandas as pd

df = pd.read_csv("/Users/jein/env/env/Fashion_product_intelligence/data/model_dataset_reduced.csv")

print(df.columns)
print(df["articleType_clean"].value_counts())

