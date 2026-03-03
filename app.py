import streamlit as st
import joblib
import pandas as pd

heart_model = joblib.load("heart_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")

st.title("🏥 Healthcare Disease Prediction System")
st.write("Enter patient health details below")

st.subheader("🧍Basic Information")
name=st.text_input("Name")
age = st.number_input("Age", 0, 120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])

st.subheader("❤️ Heart Related Details")
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar (0/1)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 250)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("CA (0-4)", [0,1,2,3,4])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

st.subheader("🩸 Diabetes Related Details")
if sex==0:
    pregnancies = st.number_input("Number of Pregnancies",0,20)
else:
    pregnancies=0

glucose = st.number_input("Glucose Level (Normal: 70-140)",50,300)

blood_pressure = st.number_input("Blood Pressure (Normal: 60-140)",60,200)

skin_thickness = st.number_input("Skin Thickness (10-50)",10,100)

insulin = st.number_input("Insulin Level (15-276)",15,900)

bmi = st.number_input("BMI (15-40)",15.0,70.0)

dpf = st.number_input("Diabetes Pedigree Function (0.0 - 2.5)",0.0,5.0)


if st.button("🔍 Predict Diseases"):

    heart_data = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,
                                thalach,exang,oldpeak,slope,ca,thal]],
                              columns=["age","sex","cp","trestbps","chol",
                                       "fbs","restecg","thalach","exang",
                                       "oldpeak","slope","ca","thal"])

    heart_pred = heart_model.predict(heart_data)[0]
    heart_prob = heart_model.predict_proba(heart_data)[0][1]

    diabetes_data = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi, dpf, age]],
                                 columns=["Pregnancies","Glucose","BloodPressure",
                                          "SkinThickness","Insulin","BMI",
                                          "DiabetesPedigreeFunction","Age"])

    diabetes_pred = diabetes_model.predict(diabetes_data)[0]
    diabetes_prob = diabetes_model.predict_proba(diabetes_data)[0][1]

    st.subheader(f"📋 Prediction Results for {name}")

    
    if heart_pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({heart_prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({heart_prob*100:.2f}%)")

    if diabetes_pred == 1:
        st.error(f"⚠️ High Risk of Diabetes ({diabetes_prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({diabetes_prob*100:.2f}%)")


    if heart_pred == 1 and diabetes_pred == 1:
        st.warning("🚨 Patient at risk of BOTH Heart Disease and Diabetes")