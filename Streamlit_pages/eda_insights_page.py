import streamlit as st

st.title("📊 Exploratory Data Analysis (EDA) Insights")

st.markdown("""
This page presents **key insights and visualizations obtained from EDA**.
The analysis helped understand the fashion catalog and guided important modeling decisions.
""")

st.divider()

# --------------------------------------------------
# 1️⃣ GENDER DISTRIBUTION
# --------------------------------------------------
st.header("1️⃣ Gender-wise Product Distribution")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/gender_distribution.png",
    caption="Gender-wise distribution of fashion products"
)

st.markdown("""
**Insight:**
- The dataset is dominated by **Men’s fashion products**
- A single gender category was selected to ensure focused and accurate modeling
""")

st.divider()

# --------------------------------------------------
# 2️⃣ CATEGORY DISTRIBUTION
# --------------------------------------------------
st.header("2️⃣ Category-wise Product Distribution")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/category_distribution.png",
    caption="Top product categories in the fashion catalog"
)

st.markdown("""
**Insight:**
- T-Shirts and Shirts are the most frequent categories
- Several categories have very few samples, indicating data imbalance
""")

st.divider()

# --------------------------------------------------
# 3️⃣ COLOR TRENDS
# --------------------------------------------------
st.header("3️⃣ Color Trends in Fashion Products")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/color_trends.png",
    caption="Most common product colors"
)

st.markdown("""
**Insight:**
- Neutral colors such as **Black, Blue, and White** dominate the dataset
- This reflects real-world fashion preferences
""")

st.divider()

# --------------------------------------------------
# 4️⃣ SEASONAL TRENDS
# --------------------------------------------------
st.header("4️⃣ Seasonal Distribution of Products")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/seasonal_trends.png",
    caption="Season-wise product distribution"
)

st.markdown("""
**Insight:**
- Most products are marked as **Summer and Fall season**
- Indicates a strong presence of casual fashion
""")

st.divider()

# --------------------------------------------------
# 5️⃣ BRAND DISTRIBUTION
# --------------------------------------------------
st.header("5️⃣ Brand-wise Product Distribution")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/brand_distribution.png",
    caption="Top brands by product count"
)

st.markdown("""
**Insight:**
- A few brands dominate the catalog
- Many brands appear infrequently, showing a long-tail distribution
""")

st.divider()

# --------------------------------------------------
# 6️⃣ Colour vs Gender distribution
# --------------------------------------------------
st.header("# 6️⃣ Colour vs Gender distribution")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/Colour_vs_gender.png",
    caption="Most usage colors by Gender in the fashion catalog"
)

st.markdown("""
**Insight:**
- Bright and Light Colours are higher usage
""")

st.divider()

# --------------------------------------------------
# 7️⃣ Season vs Gender distribution
# --------------------------------------------------
st.header("7️⃣Season vs Gender distribution")

st.image(
    "/Users/jein/env/env/Fashion_product_intelligence/eda_charts/season_vs_gender.png",
    caption="High usage Season"
)

st.markdown("""
**Insight:**
- Summer season probabably have high buyers
""")

st.divider()

# --------------------------------------------------
# ⭐ CATEGORY SELECTION JUSTIFICATION
# --------------------------------------------------
st.header("🎯 Category Selection for Modeling")

st.markdown("""
Based on the above EDA insights, the following **five categories** were selected:

- **Tshirts**
- **Shirts**
- **Casual Shoes**
- **Sports Shoes**
- **Jeans**

**Reasons for selection:**
- Sufficient number of images per category
- Clear visual patterns suitable for CNNs
- Balanced mix of apparel and footwear
- Faster and more stable training on CPU
""")

st.success("✅ EDA insights directly influenced model design and category selection.")