import pandas as pd
from flask import Flask, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import pipeline
from joblib import load

# ========== Custom Encoder ==========
class ManualFreqEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.freq_maps = {}
        
    def fit(self, X, y=None):
        for col in self.cols:
            self.freq_maps[col] = X[col].value_counts() / len(X)
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)  # unseen -> 0
        return X

# ========== Flask App ==========
app = Flask(__name__)

# ✅ Load the saved model once at startup
math_model = load("dt_pipeline_math.pkl")
por_model = load("rf_pipeline_por.pkl")
sentiment_model = pipeline("sentiment-analysis", model="sentiment_model")


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/math-risk', methods=["GET", "POST"])
def math_risk():
    prediction = None
    if request.method == "POST":
        G1 = float(request.form.get("G1"))
        G2 = float(request.form.get("G2"))

        features = {
            "sex": float(request.form.get("sex")),
            "address": float(request.form.get("address")),
            "Medu": float(request.form.get("Medu")),
            "Fedu": float(request.form.get("Fedu")),
            "Mjob": request.form.get("Mjob"),
            "reason": request.form.get("reason"),
            "studytime": float(request.form.get("studytime")),
            "schoolsup": float(request.form.get("schoolsup")),
            "paid": float(request.form.get("paid")),
            "higher": float(request.form.get("higher")),
            "internet": float(request.form.get("internet")),
            "famrel": float(request.form.get("famrel")),
            "freetime": float(request.form.get("freetime")),
            "goout": float(request.form.get("goout")),
            "Dalc": float(request.form.get("Dalc")),
            "Walc": float(request.form.get("Walc")),
            "health": float(request.form.get("health")),
            "G1": G1,
            "G2": G2,
            "Mjob_at_home": float(request.form.get("Mjob_at_home")),
            "Mjob_health": float(request.form.get("Mjob_health")),
            "Mjob_teacher": float(request.form.get("Mjob_teacher")),
            "Fjob_teacher": float(request.form.get("Fjob_teacher")),
            "reason_course": float(request.form.get("reason_course")),
            "reason_other": float(request.form.get("reason_other")),
            "reason_reputation": float(request.form.get("reason_reputation")),
            "G1_G2_avg": (G1 + G2) / 2   # ⬅️ الجديد
        }

        # ⬅️ نحوله DataFrame بنفس الأعمدة اللي اتدرب عليها
        input_df = pd.DataFrame([features])

        # 3️⃣ Make prediction
        prediction_num = math_model.predict(input_df)[0]

        # 4️⃣ Map prediction to label
        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        prediction = risk_map.get(prediction_num, "Unknown")

    return render_template("math_risk.html", prediction=prediction)

@app.route('/por-risk', methods=["GET", "POST"])
def por_risk():
    prediction = None
    if request.method == "POST":
        G1 = float(request.form.get("G1"))
        G2 = float(request.form.get("G2"))

        features = {
            "sex": float(request.form.get("sex")),
            "address": float(request.form.get("address")),
            "Medu": float(request.form.get("Medu")),
            "Fedu": float(request.form.get("Fedu")),
            "Mjob": request.form.get("Mjob"),
            "reason": request.form.get("reason"),
            "studytime": float(request.form.get("studytime")),
            "schoolsup": float(request.form.get("schoolsup")),
            "paid": float(request.form.get("paid")),
            "higher": float(request.form.get("higher")),
            "health": float(request.form.get("health")),
            "G1": G1,
            "G2": G2,
            "Mjob_at_home": float(request.form.get("Mjob_at_home")),
            "Mjob_health": float(request.form.get("Mjob_health")),
            "Mjob_teacher": float(request.form.get("Mjob_teacher")),
            "Fjob_teacher": float(request.form.get("Fjob_teacher")),
            "G1_G2_avg": (G1 + G2) / 2
        }

        # ⬅️ DataFrame
        input_df = pd.DataFrame([features])

        # 3️⃣ Predict
        prediction_num = por_model.predict(input_df)[0]

        # 4️⃣ Map result
        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        prediction = risk_map.get(prediction_num, "Unknown")

    return render_template("por_risk.html", prediction=prediction)

@app.route('/feedback-analysis', methods=["GET", "POST"])
def feedback():
    prediction = None
    if request.method == "POST":
        user_text = request.form.get("feedback_text")

        if user_text:
            result = sentiment_model(user_text)[0]
            label = result['label']
            score = result['score']
            prediction = f"{label} Acc:({score:.2f})"

    return render_template("feedback.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
