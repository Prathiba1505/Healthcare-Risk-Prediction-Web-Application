from flask import Flask, request, jsonify
import joblib
import pandas as pd
import mysql.connector

app = Flask(__name__)

heart_model = joblib.load("heart_model.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Prathiba.07",
    database="healthcare_db"
)

cursor = db.cursor()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        name=data["name"]
        heart_features = [
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]

        heart_df = pd.DataFrame([heart_features], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

        heart_pred = int(heart_model.predict(heart_df)[0])
        heart_prob = float(heart_model.predict_proba(heart_df)[0][1])

        pregnancies = data["pregnancies"] if data["sex"] == 0 else 0
        diabetes_features = [
            pregnancies, data["glucose"], data["blood_pressure"], 
            data["skin_thickness"], data["insulin"], data["bmi"],
            data["dpf"], data["age"]
        ]

        diabetes_df = pd.DataFrame([diabetes_features], columns=[
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"
        ])

        diabetes_pred = int(diabetes_model.predict(diabetes_df)[0])
        diabetes_prob = float(diabetes_model.predict_proba(diabetes_df)[0][1])

        def risk_level(prob):
            if prob < 0.4: return "Low risk"
            elif prob < 0.6: return "Moderate risk"
            elif prob < 0.8: return "High risk"
            else: return "Very high risk"

        result = {
            "heart_prediction": heart_pred,
            "heart_probability": round(heart_prob*100,2),
            "heart_risk": risk_level(heart_prob),
            "diabetes_prediction": diabetes_pred,
            "diabetes_probability": round(diabetes_prob*100,2),
            "diabetes_risk": risk_level(diabetes_prob)
        }

        sql = """
        INSERT INTO patient_predictions (
        name,
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal,
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf,
            heart_pred, heart_prob, diabetes_pred, diabetes_prob
        ) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            name,
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"],
            pregnancies, data["glucose"], data["blood_pressure"],
            data["skin_thickness"], data["insulin"], data["bmi"], data["dpf"],
            heart_pred, heart_prob, diabetes_pred, diabetes_prob
        )

        cursor.execute(sql, values)
        db.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)