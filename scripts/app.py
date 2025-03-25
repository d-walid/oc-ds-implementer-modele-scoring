from flask import Flask, request, jsonify
import joblib
import shap
import numpy as np 
import pandas as pd


app = Flask(__name__)

model = joblib.load("../models/LogisticRegression_model.joblib")

@app.route("/")
def home():
  return "Bienvenue sur l'API."


@app.route("/predict", methods=["POST"])
def predict():
  # Récupération des données envoyées en JSON
  data = request.get_json()

  