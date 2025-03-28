from flask import Flask, request, jsonify
import joblib
import shap
import numpy as np 
import pandas as pd
import os


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "LogisticRegression_model.joblib")
model = joblib.load(model_path)

df_path = os.path.join(BASE_DIR, "..", "data", "cleaned", "df_sample.csv")
df = pd.read_csv(df_path, sep=";", encoding="utf-8")

clf = model.named_steps["clf"]
feature_names = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore").columns.tolist()

coefficients = clf.coef_.flatten()
coeff_df = pd.DataFrame({
  "feature": feature_names,
  "coefficients": coefficients,
  "abs_coefficients": np.abs(coefficients)
})
coeff_df = coeff_df.sort_values(by="abs_coefficients", ascending=False)
coeff_df_top = coeff_df.head(10)[["feature", "coefficients"]].to_dict(orient="records")


@app.route("/")
def home():
  return "Bienvenue sur l'API."


@app.route("/predict", methods=["POST"])
def predict():
  # Récupération des données envoyées en JSON
  data = request.get_json()
  sk_id = data["SK_ID_CURR"]
  
  sample = df[df["SK_ID_CURR"] == sk_id].copy()
  if sample.empty:
    return jsonify({"erreur": "SK_ID_CURR introuvable."}), 404
  
  sample.drop(columns=["SK_ID_CURR", "TARGET"], inplace=True)
  
  prediction = model.predict(sample)
  prediction_proba = model.predict_proba(sample)[:, 1]
  
  return jsonify({
    "prediction" : prediction.tolist(),
    "proba_classe_1" : float(prediction_proba[0]),
    "top_20_coefficients" : coeff_df_top
    })

if __name__ == "__main__":
  app.run(debug=True, port=5000)