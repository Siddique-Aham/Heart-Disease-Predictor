
# 🫀 Heart Disease Predictor

A lightweight end-to-end ML project to predict the likelihood of heart disease from clinical inputs. It includes a scikit-learn model, a Flask web app with an HTML form, and a simple `predict` utility for programmatic inference. Licensed under MIT. ([GitHub][1])

---

## ✨ Features

* **End-to-end pipeline:** preprocessing + ML model training → saved artifact → Flask inference app.
* **Web UI:** HTML form in `templates/` to collect patient attributes and display prediction. ([GitHub][1])
* **Scripted inference:** `predict.py` to run predictions from the CLI / import in Python. ([GitHub][1])
* **Config-driven:** `config.yaml` for feature order / thresholds (kept simple). ([GitHub][1])
* **Model artifacts:** stored in `models/` for reproducible serving. ([GitHub][1])
* **MIT License** for open use. ([GitHub][1])

> Note: Repo languages show \~54% HTML and \~46% Python due to the Flask templates + Python backend. ([GitHub][1])

---

## 🗂️ Project Structure

```
SMART-HEALTH-PREDICTOR/
├── __pycache__/
│
├── data/
│   ├── processed/
│   │   ├── diabetes_processed.joblib
│   │   └── heart_processed.joblib
│   │
│   └── raw/
│       └── heart.csv
│
├── models/
│   └── heart_rf.pkl
│
├── src/
│   ├── __pycache__/
│   │   └── predict.cpython-310.pyc
│   ├── preprocess.py
│   └── train_models.py
│
├── templates/
│   └── index.html
│
├── venv/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   └── pyvenv.cfg
│
├── app.py
├── config.yaml
├── predict.py
└── requirements.txt


## 📦 Tech Stack

* **Python**, **Flask**, **scikit-learn**, **pandas**, **numpy**
* **PyYAML/ruamel** (for `config.yaml`), **joblib** or **pickle** for artifacts
* **HTML/CSS** templates for the UI

(Exact versions: see `requirements.txt` in the repo.) ([GitHub][1])

---

## 📊 Data & Target

Typical heart disease datasets (e.g., UCI Cleveland) include features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, oldpeak, slope, ca, thal, etc. The target is usually **binary** (disease vs no disease). For reference about the UCI dataset, see: ([UCI Machine Learning Repository][2])

> If you’re using a different CSV, update `config.yaml` feature order accordingly.

---

## 🧠 Modeling (overview)

* **Preprocessing:** handle missing values, scale numerical columns (e.g., StandardScaler), encode categoricals (if present).
* **Model:** a classic baseline like **Logistic Regression** or **Random Forest** (fast, interpretable, strong baseline for tabular health data).
* **Artifacts:** save preprocessor + model to `models/` and load them in `app.py` / `predict.py`.

> The exact algorithm can vary; the repo is structured to allow swapping models without changing the web code.

---

## 🚀 Quickstart

### 1) Clone & set up

```bash
git clone https://github.com/Siddique-Aham/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor

# (recommended) create venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

### 2) Configure (optional)

Open `config.yaml` and confirm:

* **feature\_order**: list in the exact order your model expects
* **model\_path** and **preprocessor\_path**: artifact locations under `models/`

### 3) Run the web app

```bash
python app.py
# or, if app.py uses Flask app factory:
# flask --app app run
```

Then open the printed local URL (usually `http://127.0.0.1:5000/`) and submit the form.


## 📈 Metrics & Validation (recommendations)

* Report **Accuracy**, **Precision/Recall**, **F1**, **ROC-AUC**.
* Use **StratifiedKFold** cross-validation.
* Calibrate probabilities if needed (e.g., **CalibratedClassifierCV**).
* Log a simple **confusion matrix** image in `models/` and embed it in the README.

(Background on widely used datasets & approaches: UCI Heart Disease dataset description.) ([UCI Machine Learning Repository][2])

---

## 🛡️ Responsible Use

This tool is **for educational/assistance purposes**, not a medical device. Do not rely on it for diagnosis or treatment decisions. Always consult qualified healthcare professionals.

---

## 🧱 Troubleshooting

* **Mismatch in input order:** Ensure web form → `predict.py` → `feature_order` are consistent.
* **Artifact not found:** Regenerate and confirm the path in `config.yaml`.
* **Version conflicts:** Recreate the virtualenv; install exact `requirements.txt`. ([GitHub][1])

---

## 📄 License

MIT © 2025 Siddique Aham. See `LICENSE`. ([GitHub][1])

---





