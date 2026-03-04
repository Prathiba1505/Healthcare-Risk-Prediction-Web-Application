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

        required_fields = [
            "name","age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope",
            "ca","thal","pregnancies","glucose",
            "blood_pressure","skin_thickness",
            "insulin","bmi","dpf"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400
            
        if "age" in data and data["age"] < 0:
            return jsonify({"error": "Age cannot be negative"}), 400

        if "chol" in data and data["chol"] <= 0:
            return jsonify({"error": "Cholesterol must be positive"}), 400

        if "bmi" in data and data["bmi"] <= 0:
            return jsonify({"error": "BMI must be positive"}), 400

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


    
@app.route("/patients", methods=["GET"])
def get_all_patients():
    try:
        page = int(request.args.get("page"))
        per_page = int(request.args.get("per_page"))
        offset = (page - 1) * per_page

        cursor.execute("SELECT COUNT(*) FROM patient_predictions")
        total_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT * FROM patient_predictions LIMIT {per_page} OFFSET {offset}")
        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No patients found"}), 404

        columns = [col[0] for col in cursor.description]

        results = [dict(zip(columns, row)) for row in rows]
        response = {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": (total_count + per_page - 1) // per_page,  # ceil division
            "patients": results
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


    
@app.route("/patients/<int:id>", methods=["GET"])
def get_patient_by_id(id):
    try:
        cursor.execute("SELECT * FROM patient_predictions WHERE id = %s", (id,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({"message": "Patient not found"}), 404

        columns = [col[0] for col in cursor.description]
        result = dict(zip(columns, row))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/patients/search", methods=["GET"])
def search_patient():
    try:
        name = request.args.get("name")

        if not name:
            return jsonify({"error": "Name query parameter is required"}), 400
        sql = "SELECT * FROM patient_predictions WHERE name LIKE %s"
        cursor.execute(sql, (f"%{name}%",))
        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No patients found"}), 404
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/dashboard", methods=["GET"])
def dashboard():
    try:
        cursor.execute("SELECT COUNT(*) FROM patient_predictions")
        total_patients = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM patient_predictions WHERE heart_prob > 0.8")
        heart_high_risk = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM patient_predictions WHERE diabetes_prob > 0.8")
        diabetes_high_risk = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(age) FROM patient_predictions")
        avg_age = cursor.fetchone()[0]

        return jsonify({
            "total_patients": total_patients,
            "heart_high_risk_count": heart_high_risk,
            "diabetes_high_risk_count": diabetes_high_risk,
            "average_age": round(avg_age, 2) if avg_age else 0
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/patients/<int:id>", methods=["PATCH"])
def update_patient(id):
    try:
        data = request.json

        if "age" in data and data["age"] < 0:
            return jsonify({"error": "Age cannot be negative"}), 400

        if "chol" in data and data["chol"] <= 0:
            return jsonify({"error": "Cholesterol must be positive"}), 400

        if "bmi" in data and data["bmi"] <= 0:
            return jsonify({"error": "BMI must be positive"}), 400
        cursor.execute("SELECT * FROM patient_predictions WHERE id = %s", (id,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({"message": "Patient not found"}), 404

        columns = [col[0] for col in cursor.description]
        old_data = dict(zip(columns, row))

        updated_data = {}
        for key in old_data:
            updated_data[key] = data.get(key, old_data[key])

        heart_features = [
            updated_data["age"], updated_data["sex"], updated_data["cp"],
            updated_data["trestbps"], updated_data["chol"], updated_data["fbs"],
            updated_data["restecg"], updated_data["thalach"],
            updated_data["exang"], updated_data["oldpeak"],
            updated_data["slope"], updated_data["ca"], updated_data["thal"]
        ]

        heart_df = pd.DataFrame([heart_features], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

        heart_pred = int(heart_model.predict(heart_df)[0])
        heart_prob = float(heart_model.predict_proba(heart_df)[0][1])

        pregnancies = updated_data["pregnancies"] if updated_data["sex"] == 0 else 0

        diabetes_features = [
            pregnancies, updated_data["glucose"], updated_data["blood_pressure"],
            updated_data["skin_thickness"], updated_data["insulin"],
            updated_data["bmi"], updated_data["dpf"], updated_data["age"]
        ]

        diabetes_df = pd.DataFrame([diabetes_features], columns=[
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"
        ])

        diabetes_pred = int(diabetes_model.predict(diabetes_df)[0])
        diabetes_prob = float(diabetes_model.predict_proba(diabetes_df)[0][1])

        sql = """
        UPDATE patient_predictions SET
            name=%s, age=%s, sex=%s, cp=%s, trestbps=%s,
            chol=%s, fbs=%s, restecg=%s, thalach=%s,
            exang=%s, oldpeak=%s, slope=%s, ca=%s, thal=%s,
            pregnancies=%s, glucose=%s, blood_pressure=%s,
            skin_thickness=%s, insulin=%s, bmi=%s, dpf=%s,
            heart_pred=%s, heart_prob=%s,
            diabetes_pred=%s, diabetes_prob=%s
        WHERE id=%s
        """

        values = (
            updated_data["name"], updated_data["age"], updated_data["sex"],
            updated_data["cp"], updated_data["trestbps"],
            updated_data["chol"], updated_data["fbs"],
            updated_data["restecg"], updated_data["thalach"],
            updated_data["exang"], updated_data["oldpeak"],
            updated_data["slope"], updated_data["ca"], updated_data["thal"],
            updated_data["pregnancies"], updated_data["glucose"],
            updated_data["blood_pressure"], updated_data["skin_thickness"],
            updated_data["insulin"], updated_data["bmi"], updated_data["dpf"],
            heart_pred, heart_prob, diabetes_pred, diabetes_prob,
            id
        )

        cursor.execute(sql, values)
        db.commit()

        def risk_level(prob):
            if prob < 0.4: return "Low risk"
            elif prob < 0.6: return "Moderate risk"
            elif prob < 0.8: return "High risk"
            else: return "Very high risk"

        return jsonify({
            "message":"Patient updated & predictions recalculated",
            "heart_prediction": heart_pred,
            "heart_probability": round(heart_prob * 100, 2),
            "heart_risk": risk_level(heart_prob),
            "diabetes_prediction": diabetes_pred,
            "diabetes_probability": round(diabetes_prob * 100, 2),
            "diabetes_risk": risk_level(diabetes_prob)
})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/patients/<int:id>", methods=["DELETE"])
def delete_patient(id):
    try:
        cursor.execute("SELECT * FROM patient_predictions WHERE id = %s", (id,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({"message": "Patient not found"}), 404

        cursor.execute("DELETE FROM patient_predictions WHERE id = %s", (id,))
        db.commit()

        return jsonify({"message": "Patient deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)