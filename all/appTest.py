import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

# =======================
# دالة لتحميل الملفات من Google Drive
# =======================
def download_from_gdrive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# =======================
# تحميل النماذج والمعالج
# =======================
download_from_gdrive("1ZyiR3ZiGNXzDihWBuTTIK8PnTa-C0Ew_", "preprocessor.pkl")
download_from_gdrive("1Hrj1_EKfwqozCTM0FaVi82Qg1nnyfThn", "best_ml_model.pkl")
download_from_gdrive("13eoCKB9sk3JqPq0qIynO05E_m1y5ebTU", "nn_model.h5")

# =======================
# إنشاء نسخة صغيرة من البيانات إذا لم تكن موجودة
# =======================
if not os.path.exists("zomato_sample.csv"):
    if os.path.exists("big_dataset.csv"):
        df_big = pd.read_csv("big_dataset.csv")
        sample_df = df_big.sample(frac=0.02, random_state=42)  # 2% sample
        sample_df.to_csv("zomato_sample.csv", index=False)
        print("✅ Sample dataset created: zomato_sample.csv")
    else:
        st.error("No dataset found! Please upload 'zomato_sample.csv' or 'big_dataset.csv'")
        st.stop()

# =======================
# تحميل النماذج والمعالج بعد التحميل
# =======================
preprocessor = joblib.load('preprocessor.pkl')
best_ml_model = joblib.load('best_ml_model.pkl')
nn_model = load_model('nn_model.h5')

# =======================
# تحميل البيانات للـ EDA
# =======================
df = pd.read_csv('zomato_sample.csv')

# =======================
# إعداد صفحة Streamlit
# =======================
st.set_page_config(page_title="Zomato Restaurant Rating Predictor", layout="wide")
page = st.sidebar.selectbox("Select Page", ["Analysis", "Prediction"])

# =======================
# صفحة التحليل
# =======================
if page == "Analysis":
    st.title("Zomato Bangalore Restaurants - EDA")
    
    st.header("Key Insights")
    st.write(f"1. Average rating: {df['rate'].mean():.2f}")
    st.write(f"2. Most common restaurant type: {df['rest_type'].mode()[0]}")
    st.write(f"3. Percentage offering online order: {(df['online_order'] == 'Yes').mean() * 100:.2f}%")
    st.write(f"4. Percentage offering table booking: {(df['book_table'] == 'Yes').mean() * 100:.2f}%")
    
    st.header("Visualizations")
    
    # Rating Distribution
    st.subheader("Distribution of Restaurant Ratings")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['rate'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Restaurant Ratings')
    st.pyplot(fig)
    
    # Top Locations
    st.subheader("Top 10 Locations by Number of Restaurants")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_locations = df['location'].value_counts().head(10)
    sns.barplot(x=top_locations.values, y=top_locations.index, ax=ax)
    ax.set_title('Top 10 Locations by Number of Restaurants')
    st.pyplot(fig)
    
    # Online Order vs Rating
    st.subheader("Rating Distribution by Online Order")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='online_order', y='rate', data=df, ax=ax)
    ax.set_title('Rating Distribution by Online Order Availability')
    st.pyplot(fig)

# =======================
# صفحة التنبؤ
# =======================
else:
    st.title("Zomato Restaurant Rating Prediction")
    
    st.header("Enter Restaurant Details")
    with st.form("prediction_form"):
        online_order = st.selectbox('Online Order', ['Yes', 'No'])
        book_table = st.selectbox('Book Table', ['Yes', 'No'])
        votes = st.number_input('Votes', min_value=0, value=0)
        approx_cost = st.number_input('Approx Cost for Two', min_value=0.0, value=0.0)
        listed_type = st.selectbox('Listed In (Type)', [
            'Buffet', 'Cafes', 'Delivery', 'Desserts', 
            'Dine-out', 'Drinks & nightlife', 'Pubs and bars'
        ])
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_df = pd.DataFrame({
                'online_order': [online_order],
                'book_table': [book_table],
                'votes': [votes],
                'approx_cost(for two people)': [approx_cost],
                'listed_in(type)': [listed_type]
            })
            input_pre = preprocessor.transform(input_df)
            
            # ML Model Prediction
            ml_pred = best_ml_model.predict(input_pre)[0]
            classes = {0: 'Low (<3.5)', 1: 'Medium (3.5-4.0)', 2: 'High (>4.0)'}
            st.subheader("Best ML Model Prediction")
            st.success(f"Predicted Rating Class: {classes[ml_pred]}")
            
            # NN Model Prediction
            nn_pred = nn_model.predict(input_pre).argmax(axis=1)[0]
            st.subheader("Neural Network Prediction")
            st.success(f"Predicted Rating Class: {classes[nn_pred]}")
