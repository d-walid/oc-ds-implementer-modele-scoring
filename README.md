# 🧠 Implémentation d'un Modèle de Scoring Crédit

Projet de data science visant à prédire la probabilité de défaut de paiement d’un client à partir de ses données personnelles, financières et historiques de crédit.  
Données issues de la compétition Kaggle : [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview).

## 📁 Structure du projet

```
oc-ds-implementer-modele-scoring/
├── data/                      # Données brutes et traitées (non incluses dans ce repo)
├── notebooks/                # Analyse exploratoire, modélisation et pipeline
│   ├── EDA.ipynb             # Analyse exploratoire des données
│   ├── ML.ipynb              # Entraînement et évaluation des modèles
│   └── Pipeline.ipynb        # Création du pipeline final de production
├── models/                   # Modèles enregistrés (format .joblib)
│   ├── LGBMClassifier_model.joblib        # Modèle final utilisé
│   ├── LogisticRegression_model.joblib
│   └── RandomForestClassifier_model.joblib
├── monitoring/               # Monitoring des performances et du drift
│   ├── data_drift.ipynb
│   ├── data_drift_predictions.ipynb
│   ├── data_drift_report.html
│   └── classification_report.html
├── scripts/                  # Scripts principaux
│   ├── app.py                # API Flask pour servir le modèle
│   ├── streamlit_app.py      # Interface Streamlit pour tester le scoring par identifiant client
│   └── versions_lib.py       # Gestion des versions des librairies
├── tests/                    # Tests unitaires
│   └── test_api.py
├── requirements.txt          # Dépendances Python
├── Procfile                  # Pour déploiement (Heroku par ex.)
└── .python-version           # Version de Python utilisée
```

## 🚀 Fonctionnalités

- 🔍 **Exploration des données**
- 🏗️ **Création d’un pipeline de features et modèle final (LGBMClassifier)**
- 📊 **Évaluation avec une fonction de coût personnalisée**
- 🧪 **Tests API**
- 🌐 **API Flask** pour exposer le modèle [👉 Accès ici](https://ocr-model-scoring-d21bdbd88983.herokuapp.com/)
- 🖥️ **Interface Streamlit** permettant d’entrer un identifiant client et afficher sa probabilité de défaut
- 📉 **Monitoring de drift** et suivi des performances du modèle

---

## ⚙️ Installation & Lancement local

### 1. Cloner le repo

```bash
git clone https://github.com/d-walid/oc-ds-implementer-modele-scoring.git
cd oc-ds-implementer-modele-scoring
```

### 2. Créer un environnement virtuel et installer les dépendances

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient toutes les librairies nécessaires (Flask, Streamlit, pandas, joblib, etc.), généré automatiquement depuis l’environnement utilisé pour entraîner et déployer le modèle.

---

## 🚀 Lancer les applications

### 🌐 API Flask

L’API permet d’interroger le modèle avec un identifiant client. Pour la lancer :

```bash
python scripts/app.py
```

Accessible localement via [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 🖥️ Interface Streamlit

L’interface utilisateur Streamlit permet de tester le scoring de clients en saisissant un ID :

```bash
streamlit run scripts/streamlit_app.py
```

Accessible généralement via [http://localhost:8501](http://localhost:8501)
