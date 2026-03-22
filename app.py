import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Adult Income Classification — End-to-End Pipeline")
st.markdown("Predict whether an individual's annual income exceeds **\$50K/yr**")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — Upload & Settings
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload adult.csv", type=["csv"])
model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    ["Decision Tree", "Logistic Regression", "Naive Bayes", "KNN"]
)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
run_btn = st.sidebar.button("🚀 Run Pipeline", use_container_width=True)

# ─────────────────────────────────────────────
# HELPER — build pipeline
# ─────────────────────────────────────────────
def build_pipeline(model_name, num_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            criterion="gini", max_depth=6,
            min_samples_split=5, min_samples_leaf=1, random_state=42
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(penalty="l2", max_iter=500)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:  # KNN
        model = KNeighborsClassifier(n_neighbors=9, weights="distance", p=1)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    return pipeline

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload your **adult.csv** file from the sidebar to begin.")
    st.markdown("""
    ### Expected CSV Columns
    | Column | Type |
    |---|---|
    | age | int |
    | workclass | object |
    | fnlwgt | int |
    | education | object |
    | education.num | int |
    | marital.status | object |
    | occupation | object |
    | relationship | object |
    | race | object |
    | sex | object |
    | capital.gain | int |
    | capital.loss | int |
    | hours.per.week | int |
    | native.country | object |
    | income | object (target: <=50K / >50K) |
    """)
    st.stop()

# ── Load Data ──────────────────────────────
df = pd.read_csv(uploaded_file)
df.replace("?", np.nan, inplace=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", "📈 EDA", "⚙️ Preprocessing", "🤖 Model & Results", "🔮 Predict"
])

# ─────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data Types & Null Count")
    info_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Null Count": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Income Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df["income"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "salmon"], edgecolor="black")
        ax.set_title("Income Distribution")
        ax.set_xlabel("Income")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df["age"], bins=20, color="steelblue", edgecolor="black")
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Age vs Income**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df.boxplot(column="age", by="income", ax=ax)
        ax.set_title("Age vs Income")
        plt.suptitle("")
        ax.set_xlabel("Income")
        ax.set_ylabel("Age")
        plt.tight_layout()
        st.pyplot(fig)

    with col4:
        st.markdown("**Sex vs Income**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x="sex", hue="income", data=df, ax=ax)
        ax.set_title("Sex vs Income")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap (Numerical Features)**")
    num_cols_eda = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    available = [c for c in num_cols_eda if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(df[available].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────
# TAB 3 — PREPROCESSING
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Preprocessing Steps")

    df_clean = df.copy()

    # Drop duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    st.success(f"✅ Removed **{before - after}** duplicate rows → {after} rows remaining")

    # Fill missing categoricals
    for col in ["workclass", "occupation", "native.country"]:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    st.success("✅ Filled missing categorical values with **mode**")

    # Encode target
    df_clean["income"] = df_clean["income"].map({"<=50K": 0, ">50K": 1})
    st.success("✅ Encoded target: **<=50K → 0**, **>50K → 1**")

    # IQR clipping on fnlwgt
    if "fnlwgt" in df_clean.columns:
        for _ in range(4):
            Q1 = df_clean["fnlwgt"].quantile(0.25)
            Q3 = df_clean["fnlwgt"].quantile(0.75)
            IQR = Q3 - Q1
            df_clean["fnlwgt"] = df_clean["fnlwgt"].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        st.success("✅ Applied **IQR clipping** on `fnlwgt` column")

    st.subheader("After Preprocessing")
    col1, col2 = st.columns(2)
    col1.metric("Remaining Rows", df_clean.shape[0])
    col2.metric("Remaining Null Values", int(df_clean.isnull().sum().sum()))
    st.dataframe(df_clean.head(10), use_container_width=True)

    # Store in session state
    st.session_state["df_clean"] = df_clean

# ─────────────────────────────────────────────
# TAB 4 — MODEL & RESULTS
# ─────────────────────────────────────────────
with tab4:
    st.subheader(f"Model: {model_choice} — Pipeline")

    if not run_btn:
        st.info("👈 Click **Run Pipeline** in the sidebar to train the model.")
    else:
        df_model = st.session_state.get("df_clean", None)
        if df_model is None:
            st.error("Please visit the Preprocessing tab first.")
        else:
            X = df_model.drop("income", axis=1)
            y = df_model["income"]

            num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            with st.spinner("Training pipeline..."):
                pipeline = build_pipeline(model_choice, num_cols, cat_cols)
                pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average="micro")

            st.success("✅ Pipeline trained successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Train Accuracy", f"{train_acc:.4f}")
            col2.metric("🎯 Test Accuracy", f"{test_acc:.4f}")
            col3.metric("📊 F1 Score", f"{f1:.4f}")

            st.subheader("Classification Report")
            report = classification_report(
                y_test, y_test_pred,
                target_names=["<=50K", ">50K"],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

            # Save model
            model_bytes = pickle.dumps(pipeline)
            st.download_button(
                label="💾 Download Trained Model (.pkl)",
                data=model_bytes,
                file_name="adult_income_model.pkl",
                mime="application/octet-stream"
            )

            # Store pipeline for prediction tab
            st.session_state["pipeline"] = pipeline
            st.session_state["num_cols"] = num_cols
            st.session_state["cat_cols"] = cat_cols

# ─────────────────────────────────────────────
# TAB 5 — PREDICT
# ─────────────────────────────────────────────
with tab5:
    st.subheader("🔮 Predict Income for New Individual")

    pipeline = st.session_state.get("pipeline", None)

    if pipeline is None:
        st.info("Please train the model in the **Model & Results** tab first.")
    else:
        st.markdown("Fill in the details below:")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 17, 90, 35)
            fnlwgt = st.number_input("fnlwgt", 10000, 1500000, 189778)
            education_num = st.number_input("Education Num", 1, 16, 10)
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        with col2:
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
            hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
            workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc",
                                                    "Federal-gov", "Local-gov", "State-gov",
                                                    "Without-pay", "Never-worked"])
            education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad",
                                                    "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                                                    "7th-8th", "12th", "Masters", "1st-4th",
                                                    "10th", "Doctorate", "5th-6th", "Preschool"])
        with col3:
            marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced",
                                                              "Never-married", "Separated",
                                                              "Widowed", "Married-spouse-absent",
                                                              "Married-AF-spouse"])
            occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service",
                                                      "Sales", "Exec-managerial", "Prof-specialty",
                                                      "Handlers-cleaners", "Machine-op-inspct",
                                                      "Adm-clerical", "Farming-fishing",
                                                      "Transport-moving", "Priv-house-serv",
                                                      "Protective-serv", "Armed-Forces"])
            relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband",
                                                          "Not-in-family", "Other-relative", "Unmarried"])
            race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
                                          "Other", "Black"])
            sex = st.selectbox("Sex", ["Male", "Female"])
            native_country = st.selectbox("Native Country", ["United-States", "Cuba", "Jamaica",
                                                              "India", "Mexico", "South", "Japan",
                                                              "Canada", "Germany", "Philippines",
                                                              "Other"])

        if st.button("🔮 Predict", use_container_width=True):
            input_data = pd.DataFrame([{
                "age": age,
                "workclass": workclass,
                "fnlwgt": fnlwgt,
                "education": education,
                "education.num": education_num,
                "marital.status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "capital.gain": capital_gain,
                "capital.loss": capital_loss,
                "hours.per.week": hours_per_week,
                "native.country": native_country
            }])

            prediction = pipeline.predict(input_data)[0]
            proba = pipeline.predict_proba(input_data)[0] if hasattr(pipeline.named_steps["model"], "predict_proba") else None

            if prediction == 1:
                st.success("✅ Predicted Income: **> 50K per year**")
            else:
                st.warning("📉 Predicted Income: **<= 50K per year**")

            if proba is not None:
                st.markdown(f"**Confidence:** <=50K: `{proba[0]:.2%}` | >50K: `{proba[1]:.2%}`")