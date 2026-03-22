# Adult-Income-Classification

# 🧠 Adult Income Prediction using Machine Learning Pipeline

## 📌 Project Overview

This project focuses on predicting whether an individual earns more than **50K/year** using the Adult Income dataset.
It implements a **complete end-to-end Machine Learning pipeline** including preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 🚀 Key Features

* ✅ Data Cleaning & Preprocessing
* ✅ Handling Missing Values using Simple Imputer
* ✅ Encoding Categorical Variables using OneHotEncoder
* ✅ Feature Scaling using StandardScaler
* ✅ ML Pipeline using `Pipeline` & `ColumnTransformer`
* ✅ Models Used:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
* ✅ Hyperparameter Tuning:

  * GridSearchCV
  * RandomizedSearchCV
* ✅ Model Evaluation:

  * Accuracy Score
  * F1 Score
* ✅ Model Saving using Pickle (Deployment Ready)

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Pickle

---

## 📂 Project Structure

```
├── data/
│   └── adult.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
├── models/
│   └── model.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

1. Load Dataset
2. Handle Missing Values
3. Encode Categorical Features
4. Scale Numerical Features
5. Build ML Pipeline
6. Train Multiple Models
7. Perform Hyperparameter Tuning
8. Evaluate Performance
9. Save Best Model

---

## 🧪 Model Training Example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVC(kernel='rbf', C=10, gamma=0.01))
])
```

---

## 📊 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~85%     |
| Random Forest       | ~88%     |
| SVM                 | ~86%     |

---

## 💾 Model Saving

```python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/adult-income-ml.git
cd adult-income-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

```bash
python src/train.py
```

### 4. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Future Improvements

* 🔹 Add Deep Learning Models
* 🔹 Deploy using Docker
* 🔹 Add CI/CD Pipeline
* 🔹 Improve Feature Engineering


---

## 👨‍💻 Author

**Praveen Reddy**
Aspiring Data Scientist 🚀
