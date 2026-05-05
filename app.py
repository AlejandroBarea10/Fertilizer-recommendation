"""
Fertilizer Recommendation System
=================================
AI for Business Analytics — Purdue University
Author: Alejandro Barea
Framework: Problem – Data – Insights – Deployment (PDID)

This Streamlit application deploys a machine learning model that recommends
the optimal fertilizer based on soil characteristics, nutrient levels,
and environmental conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid #2E7D32;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1B5E20;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.2rem;
    }
    .insight-box {
        background-color: #1a3a1a;
        border-left: 5px solid #4CAF50;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #E0E0E0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #E8F5E9, #A5D6A7);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #2E7D32;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1.1rem;
        color: #555;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1B5E20;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 40%, #388E3C 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
    div[data-testid="stSidebar"] label {
        color: white !important;
    }
    div[data-testid="stSidebar"] .stRadio label span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (cached for performance)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the fertilizer recommendation dataset."""
    df = pd.read_csv("fertilizer_recommendation.csv", sep=";")
    return df

@st.cache_resource
def load_model_artifacts():
    """Load the trained model and all preprocessing artifacts."""
    artifacts = {}
    try:
        artifacts["model"] = joblib.load("fertilizer_model.pkl")
        artifacts["label_encoders"] = joblib.load("label_encoders.pkl")
        artifacts["target_encoder"] = joblib.load("target_encoder.pkl")
        artifacts["scaler"] = joblib.load("scaler.pkl")
        artifacts["feature_columns"] = joblib.load("feature_columns.pkl")
        artifacts["uses_scaling"] = joblib.load("uses_scaling.pkl")
        artifacts["loaded"] = True
    except FileNotFoundError:
        artifacts["loaded"] = False
    return artifacts

@st.cache_data
def compute_model_results(_df):
    """
    Reproduce the full model training pipeline to show performance metrics.
    This ensures the app displays real results consistent with the notebook.
    """
    df = _df.copy()
    target = "Recommended_Fertilizer"

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_features = [c for c in cat_cols if c != target]
    num_features = num_cols.copy()
    feature_cols = cat_features + num_features

    # Encode
    data = df.copy()
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    le_target = LabelEncoder()
    data[target] = le_target.fit_transform(data[target])

    X = data[feature_cols]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    class_names = list(le_target.classes_)

    # Train models
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, max_depth=15, min_samples_split=5),
            X_train, X_test
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, min_samples_split=5, n_jobs=-1),
            X_train, X_test
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
            X_train_scaled, X_test_scaled
        ),
    }

    results = {}
    for name, (model, Xtr, Xte) in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results, class_names, feature_cols, cat_features, num_features


# ──────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌾 Navigation")
    st.markdown("---")
    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "📊 Data Explorer",
            "🔬 Model Performance",
            "🌱 Predict Fertilizer",
            "📈 Feature Importance",
            "💼 Value for Decision Makers",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This application was built as the final deliverable for the "
        "**AI for Business Analytics** course at Purdue University."
    )
    st.markdown("**Author:** Alejandro Barea")
    st.markdown("**Framework:** PDID")


# ──────────────────────────────────────────────────────────────
# LOAD DATA & ARTIFACTS
# ──────────────────────────────────────────────────────────────
df = load_data()
artifacts = load_model_artifacts()
model_results, class_names, feature_cols, cat_features, num_features = compute_model_results(df)


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">🌾 Fertilizer Recommendation System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">A Machine Learning Solution for Precision Agriculture</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Quick dataset stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">{:,}</div>'
            '<div class="metric-label">Observations</div></div>'.format(len(df)),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">{}</div>'
            '<div class="metric-label">Features</div></div>'.format(df.shape[1] - 1),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">{}</div>'
            '<div class="metric-label">Fertilizer Classes</div></div>'.format(
                df["Recommended_Fertilizer"].nunique()
            ),
            unsafe_allow_html=True,
        )
    with col4:
        best_model_name = max(model_results, key=lambda k: model_results[k]["f1"])
        best_f1 = model_results[best_model_name]["f1"]
        st.markdown(
            '<div class="metric-card"><div class="metric-value">{:.1%}</div>'
            '<div class="metric-label">Best F1-Score</div></div>'.format(best_f1),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # PDID Framework
    st.markdown("## The PDID Framework")
    st.markdown(
        "This project follows the **Problem – Data – Insights – Deployment** framework "
        "to deliver a complete analytical solution."
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### 🎯 Problem")
        st.markdown(
            "Fertilizer selection is critical in agriculture. Wrong choices lead to poor yields, "
            "soil degradation, and financial losses. Traditional recommendations rely on agronomist "
            "expertise, which is time-consuming and not always accessible. We frame this as a "
            "**multi-class classification problem**: given soil, nutrient, and environmental data, "
            "predict the optimal fertilizer."
        )
        st.markdown("### 🔍 Insights")
        st.markdown(
            "Three models were trained and compared (Decision Tree, Random Forest, KNN). "
            "Feature importance analysis revealed that nutrient levels, soil properties, and "
            "crop context are the strongest predictors. All research questions were answered "
            "with quantitative evidence."
        )
    with p2:
        st.markdown("### 📂 Data")
        st.markdown(
            "The dataset contains **10,000 agricultural observations** with 19 features "
            "covering soil characteristics (pH, moisture, organic carbon), nutrient levels "
            "(N, P, K), environmental conditions (temperature, humidity, rainfall), and "
            "agricultural context (crop type, season, region). No missing values were found."
        )
        st.markdown("### 🚀 Deployment")
        st.markdown(
            "The best-performing model was saved and integrated into this Streamlit application. "
            "Users can input field conditions and receive an instant fertilizer recommendation, "
            "making data-driven agriculture accessible to anyone."
        )

    st.markdown("---")
    st.markdown("## Research Questions")
    st.markdown(
        "1. **Which soil and environmental variables influence fertilizer recommendation?**\n"
        "2. **Can machine learning models accurately predict the optimal fertilizer?**\n"
        "3. **Which features are the most important drivers of fertilizer recommendation?**"
    )


# ══════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<p class="main-header">📊 Data Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explore the dataset and discover patterns in fertilizer recommendations</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Dataset Overview", "📊 Distributions", "🔗 Relationships", "🗺️ Correlation"]
    )

    # ── Tab 1: Dataset Overview ─────────────────────────────
    with tab1:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True, height=300)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Summary Statistics")
            st.dataframe(df.describe().T.round(2), use_container_width=True)
        with c2:
            st.markdown("### Data Types & Missing Values")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-Null": df.notnull().sum().values,
                "Missing": df.isnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, height=400)

    # ── Tab 2: Distributions ────────────────────────────────
    with tab2:
        st.markdown("### Target Variable Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        order = df["Recommended_Fertilizer"].value_counts().index
        sns.countplot(y="Recommended_Fertilizer", data=df, order=order, palette="Set2", ax=axes[0])
        axes[0].set_title("Fertilizer Frequency")
        axes[0].set_xlabel("Count")
        df["Recommended_Fertilizer"].value_counts().plot.pie(
            autopct="%1.1f%%", ax=axes[1],
            colors=sns.color_palette("Set2", df["Recommended_Fertilizer"].nunique()),
        )
        axes[1].set_ylabel("")
        axes[1].set_title("Fertilizer Proportion")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Numerical Feature Distributions")
        sel_num = st.multiselect(
            "Select variables to plot:",
            num_features,
            default=["Nitrogen_Level", "Phosphorus_Level", "Potassium_Level", "Soil_pH"],
        )
        if sel_num:
            ncols = min(len(sel_num), 4)
            nrows = (len(sel_num) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
            if nrows * ncols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            for i, col in enumerate(sel_num):
                sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("Set2")[i % 8])
                axes[i].set_title(col, fontsize=11)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 3: Relationships ────────────────────────────────
    with tab3:
        st.markdown("### Nutrient Levels by Fertilizer Type")
        nutrients = ["Nitrogen_Level", "Phosphorus_Level", "Potassium_Level"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, nut in enumerate(nutrients):
            sns.boxplot(x="Recommended_Fertilizer", y=nut, data=df, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{nut}")
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Categorical Variable Analysis")
        sel_cat = st.selectbox("Select a categorical variable:", cat_features)
        fig, ax = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df[sel_cat], df["Recommended_Fertilizer"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title(f"Fertilizer Distribution by {sel_cat}")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Fertilizer", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: Correlation ──────────────────────────────────
    with tab4:
        st.markdown("### Correlation Heatmap — Numerical Features")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[num_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.5, square=True, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="insight-box">'
            "<strong>Key Insight:</strong> The numerical features show low multicollinearity, "
            "meaning each variable contributes independent information to the model. "
            "This is favorable for training reliable classifiers."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Model Performance":
    st.markdown('<p class="main-header">🔬 Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Comparing Decision Tree, Random Forest, and K-Nearest Neighbors</p>',
        unsafe_allow_html=True,
    )

    # Summary table
    st.markdown("### Performance Comparison")
    comp_data = []
    for name, res in model_results.items():
        comp_data.append({
            "Model": name,
            "Accuracy": f'{res["accuracy"]:.4f}',
            "Precision": f'{res["precision"]:.4f}',
            "Recall": f'{res["recall"]:.4f}',
            "F1-Score": f'{res["f1"]:.4f}',
        })
    comp_df = pd.DataFrame(comp_data).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    comp_numeric = pd.DataFrame([
        {"Model": n, "Accuracy": r["accuracy"], "Precision": r["precision"],
         "Recall": r["recall"], "F1-Score": r["f1"]}
        for n, r in model_results.items()
    ])
    comp_melt = comp_numeric.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Metric", y="Score", hue="Model", data=comp_melt, palette="Set2", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=14)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Best model highlight
    best_name = max(model_results, key=lambda k: model_results[k]["f1"])
    best_res = model_results[best_name]
    st.markdown("---")
    st.success(
        f"**Best Model: {best_name}** — selected based on highest F1-Score ({best_res['f1']:.4f}), "
        f"which balances precision and recall across all 7 fertilizer classes."
    )

    # Confusion matrices
    st.markdown("---")
    st.markdown("### Confusion Matrices")
    selected_model = st.selectbox("Select a model to view its confusion matrix:", list(model_results.keys()))
    res = model_results[selected_model]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(res["y_test"], res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {selected_model}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification report
    st.markdown(f"### Classification Report — {selected_model}")
    report = classification_report(res["y_test"], res["y_pred"], target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        "<strong>Technical Note:</strong> Tree-based models (Decision Tree, Random Forest) were trained "
        "on unscaled data, while KNN used StandardScaler-transformed features since it is "
        "distance-based. The scaler was fitted only on training data to prevent data leakage."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT FERTILIZER
# ══════════════════════════════════════════════════════════════
elif page == "🌱 Predict Fertilizer":
    st.markdown('<p class="main-header">🌱 Fertilizer Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter your field conditions to receive a fertilizer recommendation</p>',
        unsafe_allow_html=True,
    )

    if not artifacts["loaded"]:
        st.error(
            "**Model files not found.** Please run the Jupyter Notebook first to generate "
            "the `.pkl` files (fertilizer_model.pkl, label_encoders.pkl, etc.)."
        )
        st.stop()

    # Input form
    st.markdown("### Field Conditions")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🌍 Soil & Environment**")
        soil_type = st.selectbox("Soil Type", ["Clay", "Loamy", "Sandy", "Silt"])
        soil_ph = st.slider("Soil pH", 45, 849, 600, help="Soil acidity/alkalinity level")
        soil_moisture = st.slider("Soil Moisture", 100, 5999, 3200, help="Soil moisture content")
        organic_carbon = st.slider("Organic Carbon", 2, 149, 75)
        electrical_cond = st.slider("Electrical Conductivity", 1, 299, 140)

    with col_b:
        st.markdown("**🧪 Nutrient Levels**")
        nitrogen = st.slider("Nitrogen Level (N)", 20, 159, 85)
        phosphorus = st.slider("Phosphorus Level (P)", 10, 89, 45)
        potassium = st.slider("Potassium Level (K)", 10, 119, 60)
        st.markdown("**🌡️ Weather**")
        temperature = st.slider("Temperature", 100, 3999, 2300)
        humidity = st.slider("Humidity", 301, 8999, 5500)
        rainfall = st.slider("Rainfall", 2102, 299998, 144000)

    with col_c:
        st.markdown("**🌾 Agricultural Context**")
        crop_type = st.selectbox("Crop Type", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"])
        growth_stage = st.selectbox("Crop Growth Stage", ["Flowering", "Harvest", "Sowing", "Vegetative"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        irrigation = st.selectbox("Irrigation Type", ["Canal", "Drip", "Rainfed", "Sprinkler"])
        previous_crop = st.selectbox("Previous Crop", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"])
        region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])
        fert_last = st.slider("Fertilizer Used Last Season", 504, 29998, 16000)
        yield_last = st.slider("Yield Last Season", 10, 799, 400)

    st.markdown("---")

    # Predict button
    if st.button("🔍 Get Fertilizer Recommendation", type="primary", use_container_width=True):
        new_sample = {
            "Soil_Type": soil_type,
            "Crop_Type": crop_type,
            "Crop_Growth_Stage": growth_stage,
            "Season": season,
            "Irrigation_Type": irrigation,
            "Previous_Crop": previous_crop,
            "Region": region,
            "Soil_pH": soil_ph,
            "Soil_Moisture": soil_moisture,
            "Organic_Carbon": organic_carbon,
            "Electrical_Conductivity": electrical_cond,
            "Nitrogen_Level": nitrogen,
            "Phosphorus_Level": phosphorus,
            "Potassium_Level": potassium,
            "Temperature": temperature,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "Fertilizer_Used_Last_Season": fert_last,
            "Yield_Last_Season": yield_last,
        }

        # Encode
        feat_cols = artifacts["feature_columns"]
        le_dict = artifacts["label_encoders"]
        sample_encoded = {}
        for col in feat_cols:
            if col in le_dict:
                sample_encoded[col] = le_dict[col].transform([new_sample[col]])[0]
            else:
                sample_encoded[col] = new_sample[col]

        sample_df = pd.DataFrame([sample_encoded], columns=feat_cols)

        if artifacts["uses_scaling"]:
            sample_df = pd.DataFrame(
                artifacts["scaler"].transform(sample_df), columns=feat_cols
            )

        prediction = artifacts["model"].predict(sample_df)
        predicted_fertilizer = artifacts["target_encoder"].inverse_transform(prediction)[0]

        # Fertilizer descriptions
        fert_info = {
            "Urea": "A nitrogen-rich fertilizer (46% N) ideal for promoting vegetative growth. Best applied during early growth stages.",
            "DAP": "Di-Ammonium Phosphate provides both nitrogen and phosphorus. Excellent for root development and early plant establishment.",
            "MOP": "Muriate of Potash (Potassium Chloride) supplies potassium for disease resistance, water regulation, and fruit quality.",
            "NPK": "A balanced fertilizer providing Nitrogen, Phosphorus, and Potassium in specific ratios for general crop nutrition.",
            "SSP": "Single Super Phosphate provides phosphorus and sulfur. Suitable for soils deficient in both nutrients.",
            "Compost": "Organic matter that improves soil structure, water retention, and microbial activity. Ideal for long-term soil health.",
            "Zinc Sulphate": "A micronutrient fertilizer addressing zinc deficiency, critical for enzyme function and growth hormone production.",
        }

        st.markdown(
            '<div class="prediction-box">'
            '<div class="prediction-label">Recommended Fertilizer</div>'
            f'<div class="prediction-value">🌿 {predicted_fertilizer}</div>'
            f'<div class="prediction-label">{fert_info.get(predicted_fertilizer, "")}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Show input summary
        with st.expander("📋 View Input Summary"):
            input_df = pd.DataFrame([new_sample]).T
            input_df.columns = ["Value"]
            st.dataframe(input_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.markdown('<p class="main-header">📈 Feature Importance & Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Understanding what drives fertilizer recommendations</p>',
        unsafe_allow_html=True,
    )

    # Extract feature importance from Random Forest
    rf_model = model_results["Random Forest"]["model"]
    dt_model = model_results["Decision Tree"]["model"]

    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Top features visualization
    st.markdown("### Random Forest Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(feat_imp))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis", ax=ax)
    ax.set_title("Feature Importance — Random Forest", fontsize=14)
    ax.set_xlabel("Relative Importance")
    for i, v in enumerate(feat_imp["Importance"]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Comparison DT vs RF
    st.markdown("### Decision Tree vs Random Forest Comparison")
    dt_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Decision Tree": dt_model.feature_importances_,
        "Random Forest": rf_model.feature_importances_,
    }).sort_values("Random Forest", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    x_pos = np.arange(len(dt_imp))
    width = 0.35
    ax.barh(x_pos - width / 2, dt_imp["Decision Tree"], width, label="Decision Tree", color="#66c2a5")
    ax.barh(x_pos + width / 2, dt_imp["Random Forest"], width, label="Random Forest", color="#fc8d62")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(dt_imp["Feature"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance: Decision Tree vs Random Forest", fontsize=13)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Research question answers
    st.markdown("---")
    st.markdown("### Answering the Research Questions")

    st.markdown(
        '<div class="insight-box">'
        "<strong>RQ1: Which soil and environmental variables influence fertilizer recommendation?</strong><br>"
        "The feature importance analysis shows that virtually all input variables contribute "
        "to the recommendation. Nutrient levels (N, P, K), soil properties (pH, moisture, organic carbon), "
        "and crop-related variables consistently rank among the most influential features."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="insight-box">'
        "<strong>RQ2: Can machine learning models accurately predict the optimal fertilizer?</strong><br>"
        "Yes. All three models achieved meaningful predictive performance. The best model "
        f"({max(model_results, key=lambda k: model_results[k]['f1'])}) achieved an F1-Score of "
        f"{max(r['f1'] for r in model_results.values()):.4f}, demonstrating that machine learning "
        "can effectively capture the complex relationships in agricultural data."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="insight-box">'
        "<strong>RQ3: Which features are the most important drivers?</strong><br>"
        "The top 5 most important features according to the Random Forest model are: "
        + ", ".join(f"**{row['Feature']}** ({row['Importance']:.3f})" for _, row in feat_imp.head(5).iterrows())
        + ". These align with agronomic knowledge — fertilizer choice is driven by what the soil "
        "contains, what the plant needs, and the environmental context."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE: VALUE FOR DECISION MAKERS
# ══════════════════════════════════════════════════════════════
elif page == "💼 Value for Decision Makers":
    st.markdown('<p class="main-header">💼 Value for Decision Makers</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">How this solution delivers tangible business value</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### The Problem This Solution Addresses")
    st.markdown(
        "Selecting the right fertilizer is a complex decision influenced by soil conditions, "
        "nutrient levels, weather, crop type, and agricultural practices. Incorrect fertilizer "
        "choices result in wasted resources, reduced yields, environmental damage, and economic losses. "
        "Many farmers, especially smallholders, lack access to personalized agronomic advice."
    )

    st.markdown("---")
    st.markdown("### Tangible Benefits")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown("#### 🎯 Better Decisions")
        st.markdown(
            "Data-driven recommendations replace guesswork, leading to optimized "
            "fertilizer selection tailored to specific field conditions."
        )
    with b2:
        st.markdown("#### ⏱️ Time Savings")
        st.markdown(
            "Instant recommendations eliminate the need for lengthy manual analysis "
            "or waiting for agronomist consultations."
        )
    with b3:
        st.markdown("#### 💰 Cost Reduction")
        st.markdown(
            "Precise recommendations prevent over-application of expensive fertilizers, "
            "reducing input costs while maintaining optimal yields."
        )
    with b4:
        st.markdown("#### 🌍 Sustainability")
        st.markdown(
            "Avoiding over-fertilization reduces nutrient runoff into waterways, "
            "contributing to environmental protection and sustainable farming."
        )

    st.markdown("---")
    st.markdown("### How a Non-Technical User Can Use This Tool")
    st.markdown(
        "1. **Navigate to the Predict Fertilizer page** using the sidebar.\n"
        "2. **Enter your field data** — soil type, nutrient test results, crop, weather conditions, and region.\n"
        "3. **Click the prediction button** to receive an instant fertilizer recommendation.\n"
        "4. **Review the result** — the system explains what the recommended fertilizer does and why it fits your conditions.\n\n"
        "No coding, no statistical knowledge, no agronomic expertise required. "
        "The system translates complex data patterns into a simple, actionable recommendation."
    )

    st.markdown("---")
    st.markdown("### Who Benefits?")

    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("#### 👨‍🌾 Farmers")
        st.markdown(
            "Access personalized fertilizer advice without depending on scarce expert consultations."
        )
    with u2:
        st.markdown("#### 🏢 Agricultural Companies")
        st.markdown(
            "Companies like Fertinagro Biotech can integrate this system into advisory services "
            "and precision agriculture products."
        )
    with u3:
        st.markdown("#### 🏛️ Policy Makers")
        st.markdown(
            "Data-backed recommendations support sustainable agriculture policies "
            "and resource allocation at regional level."
        )

    st.markdown("---")
    st.markdown("### Conclusions & Key Learnings")

    st.markdown("#### Technical Learnings")
    st.markdown(
        "Building this project provided hands-on experience with the full machine learning pipeline: "
        "data exploration, preprocessing (encoding, scaling, train-test splitting with stratification), "
        "training multiple classifiers, evaluating with proper metrics, and deploying via Streamlit. "
        "Preventing data leakage (fitting the scaler only on training data) and ensuring consistent "
        "preprocessing for deployment were critical technical lessons."
    )

    st.markdown("#### Analytical Insights")
    st.markdown(
        "Fertilizer recommendation is inherently multivariate — no single feature dominates. "
        "The Random Forest model captured complex interactions between features better than simpler models. "
        "Feature importance analysis confirmed that recommendations align with agronomic knowledge, "
        "reinforcing the model's credibility."
    )

    st.markdown("#### Challenges & Solutions")
    st.markdown(
        "Key challenges included ensuring the preprocessing pipeline (encoders, scaler) was saved "
        "and reusable for deployment, handling the semicolon-separated CSV format, and designing an "
        "intuitive interface for non-technical users. Each was addressed through careful engineering "
        "and testing of the full prediction pipeline."
    )

    st.markdown("#### Personal Growth")
    st.markdown(
        "This project reinforced the importance of thinking beyond model accuracy — the real value "
        "lies in making analytical solutions accessible and useful for decision makers. Following "
        "the PDID framework ensured a structured, end-to-end approach that connects business "
        "problems to deployed solutions."
    )


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "Fertilizer Recommendation System — AI for Business Analytics — Purdue University — "
    "Alejandro Barea — PDID Framework — 2026"
    "</div>",
    unsafe_allow_html=True,
)
