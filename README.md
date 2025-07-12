# 🩺 Diabetes Prediction Web App

A **machine learning-powered Streamlit web app** that predicts whether a person is likely to have **diabetes** based on key health parameters. Built using the **PIMA Indian Diabetes Dataset**, this app allows users to interact with an easy-to-use interface and get real-time predictions from a trained model.

---

## Features

- Predicts diabetes likelihood using 8 input medical attributes
- Clean and modern **dark-themed UI** built with Streamlit
- Powered by a **Random Forest Classifier** trained on preprocessed health data
- Displays prediction result and confidence level
- Easy to run locally or deploy online

---


## Project Structure

```
diabetes-prediction-app/
├── /Results
├── app.py                # Main Streamlit app
├── diabetes.csv          
├── diabetes_model.pkl    # Trained ML model
├── requirements.txt              
├── scaler.pkl            # Fitted StandardScaler
└── README.md             # Project overview
```

---

## Dataset Used

- **Name**: PIMA Indian Diabetes Dataset
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age
  - Outcome (0 = No diabetes, 1 = Diabetes)

---

## ⚙️ How to Run Locally

1. **Clone the Repository**

```bash
git clone https://github.com/asthanaaviral/diabetes-prediction-app.git
cd diabetes-prediction-app
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run app.py
```

4. **Open in Browser** Go to: [http://localhost:8501](http://localhost:8501)

---

## Model Details

- Algorithm: `RandomForestClassifier`
- Accuracy: \~76% on test set
- Scaler: `StandardScaler`
- Stored using: `joblib`

---

## Example Inputs

### Not Likely to Have Diabetes

```
Pregnancies: 1
Glucose: 85
Blood Pressure: 70
Skin Thickness: 20
Insulin: 80
BMI: 24.0
DPF: 0.3
Age: 25
```

### Likely to Have Diabetes

```
Pregnancies: 5
Glucose: 165
Blood Pressure: 80
Skin Thickness: 35
Insulin: 200
BMI: 38.0
DPF: 1.0
Age: 50
```

---

## Note
This project is built for educational/demo purposes and is freely available for use.

---

## Author

**Aviral Asthana**
