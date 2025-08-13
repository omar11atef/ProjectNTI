# === appTest.py ===
import streamlit as st
import pandas as pd
import joblib
import gdown
import tensorflow as tf

# ==============================
# روابط Google Drive
# ==============================
files = {
    "preprocessor.pkl": "1ZyiR3ZiGNXzDihWBuTTIK8PnTa-C0Ew_",
    "best_ml_model.pkl": "1Hrj1_EKfwqozCTM0FaVi82Qg1nnyfThn",
    "nn_model.h5": "13eoCKB9sk3JqPq0qIynO05E_m1y5ebTU",
    "zomato_sample.csv": "1U3CMhKvQ2_lOaFKVpFQVDxrr6hV4UE_Z"
}

# ==============================
# تحميل الملفات من Google Drive
# ==============================
def download_file(file_name, file_id):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

for fname, fid in files.items():
    download_file(fname, fid)

# ==============================
# تحميل البيانات
# ==============================
df = pd.read_csv("zomato_sample.csv")

# ==============================
# تحميل النماذج و الـ preprocessor
# ==============================
preprocessor = joblib.load("preprocessor.pkl")
best_model = joblib.load("best_ml_model.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")

# ==============================
# واجهة Streamlit
# ==============================
st.title("Zomato Restaurant Prediction App 🍽️")

st.write("### أدخل بيانات المطعم:")
online_order = st.selectbox("هل يوجد طلب أونلاين؟", ["Yes", "No"])
book_table = st.selectbox("هل يوجد حجز طاولة؟", ["Yes", "No"])
votes = st.number_input("عدد التقييمات", min_value=0)
location = st.text_input("الموقع")
rest_type = st.text_input("نوع المطعم")
cuisines = st.text_input("المأكولات المقدمة")
cost = st.number_input("تكلفة لشخصين", min_value=0)

if st.button("توقع"):
    input_data = pd.DataFrame({
        "online_order": [online_order],
        "book_table": [book_table],
        "votes": [votes],
        "location": [location],
        "rest_type": [rest_type],
        "cuisines": [cuisines],
        "approx_cost(for_two_people)": [cost]
    })

    # تجهيز البيانات
    processed_data = preprocessor.transform(input_data)

    # توقع من أفضل نموذج ML
    ml_pred = best_model.predict(processed_data)

    # توقع من NN model
    nn_pred = nn_model.predict(processed_data)

    st.write("### النتيجة من نموذج ML:", ml_pred)
    st.write("### النتيجة من نموذج NN:", nn_pred)

st.write("### عينة من البيانات:")
st.dataframe(df.head())
