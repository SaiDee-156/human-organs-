import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

# --- MANDATORY IMPORTS (ML Models) ---
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --- Configuration ---
DEFAULT_PKL = r'my_cleaned_organ_data.xlsx'
RANDOM_STATE = 42

# --- HARDCODED DEFAULT VALUES (Used as a fallback for sliders) ---
DEFAULT_USER_VALUES = {
    "Age": 51.0, "Height_cm": 175.0, "Weight_kg": 100.0, "Heart_Rate": 75.0, "Lung_Capacity_L": 4.25,
    "Liver_ALT": 52.0, "Kidney_eGFR": 75.0, "Brain_Score": 74.0, "Glucose": 140.0, "Thyroid_TSH": 5.25,
    "Stomach_pH": 3.0, "Bone_Density": 1.05, "BP_Systolic": 135.0, "BP_Diastolic": 90.0, "Muscle_Mass_kg": 40.0,
    "Spleen_Size_cm": 11.50, "Bladder_Capacity_ml": 500.0, "Vision_Acuity": 0.55, "Skin_Elasticity": 5.0,
    "Liver_Function_Score": 50.0, "Kidney_Creatinine": 2.50, "Pancreas_Insulin": 17.50, "Reaction_Time_ms": 325.0,
}

# Define parameter ranges for the sliders
FEATURE_PARAMS = {
    "Age": (18, 100, 1, 1), "Height_cm": (100, 220, 1, 1), "Weight_kg": (30, 200, 1, 1), "Heart_Rate": (30, 150, 1, 1),
    "Lung_Capacity_L": (1.0, 8.0, 0.1, 2), "Liver_ALT": (5, 200, 1, 1), "Kidney_eGFR": (10, 150, 1, 1), "Brain_Score": (0, 100, 1, 1),
    "Glucose": (50, 400, 1, 1), "Thyroid_TSH": (0.1, 20.0, 0.1, 2), "Stomach_pH": (1.0, 7.0, 0.1, 2), "Bone_Density": (0.5, 2.0, 0.01, 2),
    "BP_Systolic": (80, 220, 1, 1), "BP_Diastolic": (50, 140, 1, 1), "Muscle_Mass_kg": (10.0, 80.0, 0.1, 2), "Spleen_Size_cm": (5.0, 20.0, 0.1, 2),
    "Bladder_Capacity_ml": (100, 1000, 1, 1), "Vision_Acuity": (0.0, 1.5, 0.01, 2), "Skin_Elasticity": (1, 10, 1, 1),
    "Liver_Function_Score": (0, 100, 1, 1), "Kidney_Creatinine": (0.1, 10.0, 0.1, 2), "Pancreas_Insulin": (1.0, 50.0, 0.1, 2),
    "Reaction_Time_ms": (100, 1000, 1, 1),
}

# --- Risk Criteria Map (Used for both target creation and live assessment) ---
RISK_MAP = {
    "Age": lambda x: x > 65, "Heart_Rate": lambda x: (x < 60) | (x > 100),
    "Lung_Capacity_L": lambda x: x < 3.0, "Liver_ALT": lambda x: x > 40,
    "Kidney_eGFR": lambda x: x < 60, "Brain_Score": lambda x: x < 70,
    "Glucose": lambda x: x > 126, "Thyroid_TSH": lambda x: x > 5.0,
    "Stomach_pH": lambda x: x > 3.0, "Bone_Density": lambda x: x < 0.8,
    "BP_Systolic": lambda x: x > 140, "BP_Diastolic": lambda x: x > 90,
    "Spleen_Size_cm": lambda x: x > 12.0, "Bladder_Capacity_ml": lambda x: x < 300,
    "Vision_Acuity": lambda x: x < 0.5, "Kidney_Creatinine": lambda x: x > 1.3,
    "Reaction_Time_ms": lambda x: x > 350, "Liver_Function_Score": lambda x: x < 50,
}

# --- HEALTH SUGGESTIONS MAP ---
HEALTH_SUGGESTIONS = {
    "Age": "Focus on muscle-strengthening exercises (2x/week) and cognitive training to maintain brain health.",
    "Height_cm": None, 
    "Weight_kg": "Consult a nutritionist to establish a balanced diet. Aim for consistent, moderate weight reduction through diet and increased activity.",
    "Heart_Rate": "If high or low, seek medical review. Otherwise, consistent aerobic exercise (3-5 times/week) improves heart efficiency and stabilizes heart rate.",
    "Lung_Capacity_L": "Practice deep-breathing exercises and sustained cardiovascular activity (swimming or running) to increase lung capacity.",
    "Liver_ALT": "Reduce alcohol consumption and avoid processed foods high in sugar and unhealthy fats. Increase water and vegetable intake.",
    "Kidney_eGFR": "Maintain excellent hydration. Limit sodium and processed protein intake. Avoid self-medicating with over-the-counter painkillers (NSAIDs).",
    "Brain_Score": "Engage in complex mental tasks (learning a new language, puzzles), prioritize 7-9 hours of sleep, and reduce chronic stress.",
    "Glucose": "Limit simple carbohydrates and sugary drinks. Incorporate fiber-rich foods (beans, whole grains) and consider moderate daily walking.",
    "Thyroid_TSH": "Consult an endocrinologist. Ensure adequate intake of iodine and selenium, but only under professional guidance.",
    "Stomach_pH": "Avoid eating right before bed. Limit highly acidic foods and drinks (coffee, soda). Eat smaller, more frequent meals.",
    "Bone_Density": "Ensure adequate Calcium and Vitamin D intake. Engage in weight-bearing exercises (walking, weightlifting) 3-4 times a week.",
    "BP_Systolic": "Adopt the DASH diet (low sodium, high potassium). Incorporate daily brisk walking (30 min) and stress reduction techniques.",
    "BP_Diastolic": "Adopt the DASH diet (low sodium, high potassium). Incorporate daily brisk walking (30 min) and stress reduction techniques.",
    "Muscle_Mass_kg": "Increase protein intake. Engage in resistance training (weights or bodyweight exercises) 3 times per week.",
    "Spleen_Size_cm": "Requires medical review. Ensure you are not fighting an underlying infection. Focus on immune system support.",
    "Bladder_Capacity_ml": "Avoid excessive caffeine and alcohol, which irritate the bladder. Practice timed voiding techniques.",
    "Vision_Acuity": "Consult an eye specialist for an updated prescription. Ensure regular screen breaks (20-20-20 rule).",
    "Skin_Elasticity": "Stay well-hydrated. Use a daily moisturizer with SPF. Increase Vitamin C and E intake.",
    "Liver_Function_Score": "See Liver ALT suggestion. Consistent light-to-moderate exercise supports overall liver function.",
    "Kidney_Creatinine": "Limit red meat and high-protein supplements, as they can temporarily increase creatinine. Ensure full hydration.",
    "Pancreas_Insulin": "Manage blood sugar through diet (low-glycemic index foods) and regular, intense physical activity.",
    "Reaction_Time_ms": "Ensure sufficient sleep quality and duration. Practice cognitive speed games and focused meditation.",
}

# --- Core Logic Functions ---

@st.cache_data
def load_data_from_path(path):
    """Loads data based on file extension (xlsx or pkl)."""
    if path.lower().endswith(('.pkl', '.pickle')):
        return pd.read_pickle(path)
    elif path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return None

@st.cache_data
def create_target(df_in):
    """Calculates the 'high_risk' target based on 3 or more flagged risk criteria."""
    df_local = df_in.copy()
    conds = []

    for col, criteria in RISK_MAP.items():
        if col in df_local.columns:
            conds.append(criteria(df_local[col]))

    if not conds:
        df_local["high_risk"] = 0
    else:
        risk_count = pd.concat(conds, axis=1).sum(axis=1)
        # High risk if 3 or more criteria are met
        df_local["high_risk"] = (risk_count >= 3).astype(int)
    return df_local

def assess_user_risk_by_rule(df_in):
    """Calculates the rule-based 'high_risk' target for a single input row (df_in must be 1 row)."""
    conds = []

    for col, criteria in RISK_MAP.items():
        if col in df_in.columns:
            conds.append(criteria(df_in[col].iloc[0])) 

    if not conds:
        return 0
    else:
        risk_count = sum(conds)
        return 1 if risk_count >= 3 else 0

def generate_health_suggestions(df_in):
    """
    Compares the single-row user input against RISK_MAP 
    and returns actionable suggestions for flagged metrics.
    """
    suggestions = []
    
    for col, criteria in RISK_MAP.items():
        if col in df_in.columns and HEALTH_SUGGESTIONS.get(col):
            # Check if the user's value crosses the risk threshold
            user_value = df_in[col].iloc[0]
            is_at_risk = criteria(user_value)
            
            if is_at_risk:
                # Metric is at risk, provide the suggestion
                suggestion = HEALTH_SUGGESTIONS[col]
                
                suggestions.append({
                    "Feature": col,
                    "Value": user_value,
                    "Suggestion": suggestion
                })

    return suggestions

# --- Training Logic Function ---
@st.cache_resource(show_spinner="Training all 3 models...")
def train_all_models(X_train, X_test, y_train, y_test, X, y, feature_columns):
    """Trains XGBoost, Random Forest, and Logistic Regression models."""
    
    models = {}
    results = {}
    stratify_available = len(np.unique(y)) > 1
    
    # 1. XGBoost Classifier
    if XGBOOST_AVAILABLE:
        model_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss"
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model_xgb.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        cv_scores = cross_val_score(model_xgb, X, y, cv=5, scoring="accuracy") if stratify_available else np.array([accuracy])
        
        models['XGBoost'] = model_xgb
        results['XGBoost'] = {
            "accuracy": accuracy, "cv_mean": float(np.mean(cv_scores)),
            "y_test": y_test, "y_pred": y_pred
        }
    
    # 2. Random Forest Classifier
    model_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    accuracy_rf = float(accuracy_score(y_test, y_pred_rf))
    cv_scores_rf = cross_val_score(model_rf, X, y, cv=5, scoring="accuracy") if stratify_available else np.array([accuracy_rf])

    models['Random Forest'] = model_rf
    results['Random Forest'] = {
        "accuracy": accuracy_rf, "cv_mean": float(np.mean(cv_scores_rf)),
        "y_test": y_test, "y_pred": y_pred_rf
    }
    
    # 3. Logistic Regression
    model_lr = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=1000, C=100) 
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    accuracy_lr = float(accuracy_score(y_test, y_pred_lr))
    cv_scores_lr = cross_val_score(model_lr, X, y, cv=5, scoring="accuracy") if stratify_available else np.array([accuracy_lr])
    
    models['Logistic Regression'] = model_lr
    results['Logistic Regression'] = {
        "accuracy": accuracy_lr, "cv_mean": float(np.mean(cv_scores_lr)),
        "y_test": y_test, "y_pred": y_pred_lr
    }

    return models, results

# --- Plotting Helper ---
def st_plot_figure(fig):
    """Displays a Matplotlib figure in Streamlit using a byte buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

# --- Plotting Functions ---

def plot_univariate(df_local, features):
    if not features: return
    cols = 3
    rows_for_grid = 2 
    features_to_plot = features[:3]
    
    fig = plt.figure(figsize=(4 * cols, 3 * rows_for_grid))
    
    for i, feat in enumerate(features_to_plot):
        ax1 = fig.add_subplot(rows_for_grid, cols, i + 1)
        sns.histplot(df_local[feat], kde=True, ax=ax1)
        ax1.set_title(f"Distribution: {feat}", fontsize=10)
    
    for i, feat in enumerate(features_to_plot):
        ax2 = fig.add_subplot(rows_for_grid, cols, i + 1 + cols) 
        sns.boxplot(y=df_local[feat], ax=ax2)
        ax2.set_title(f"Boxplot: {feat}", fontsize=10)
        
    plt.tight_layout()
    st_plot_figure(fig)

def plot_correlation(df_local, features):
    if not features: return

    if len(features) > 15:
        cols = features[:15] + ["high_risk"]
    else:
        cols = features + ["high_risk"]
        
    corr = df_local[cols].corr()
    
    fig, ax = plt.subplots(figsize=(max(14, len(cols) * 1.0), max(12, len(cols) * 0.8)))
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={'shrink': 0.7}, annot_kws={"size": 8})
    ax.set_title("Correlation Matrix (incl. high_risk) - Limited Features", fontsize=16) 
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()
    st_plot_figure(fig)

def plot_target_pie(y_series):
    counts = y_series.value_counts().reindex([0,1], fill_value=0)
    labels = [f"Low Risk (0): {counts.loc[0]}", f"High Risk (1): {counts.loc[1]}"]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90, explode=(0.03,0))
    ax.set_title("Target Distribution (Rule-Based)")
    st_plot_figure(fig)

def plot_organ_risks(df_local):
    risks = {}
    
    risk_map_labels = {
        "Age (>65)": "Age", "Heart Rate (Out of Range)": "Heart_Rate",
        "Lung Capacity (<3.0L)": "Lung_Capacity_L", "Liver ALT (>40 U/L)": "Liver_ALT",
        "Kidney eGFR (<60)": "Kidney_eGFR", "Brain Score (<70)": "Brain_Score",
        "Glucose (>126 mg/dL)": "Glucose", "Thyroid TSH (>5.0)": "Thyroid_TSH",
        "Stomach pH (>3.0)": "Stomach_pH", "Bone Density (<0.8)": "Bone_Density",
        "BP Systolic (>140)": "BP_Systolic", "BP Diastolic (>90)": "BP_Diastolic",
        "Spleen Size (>12.0cm)": "Spleen_Size_cm", "Bladder Capacity (<300ml)": "Bladder_Capacity_ml",
        "Vision Acuity (<0.5)": "Vision_Acuity", "Kidney Creatinine (>1.3)": "Kidney_Creatinine",
        "Reaction Time (>350ms)": "Reaction_Time_ms", "Liver Function Score (<50)": "Liver_Function_Score",
    }
    
    for label, col in risk_map_labels.items():
        if col in df_local.columns and col in RISK_MAP:
            criteria = RISK_MAP[col]
            risks[label] = criteria(df_local[col]).sum()

    labels = list(risks.keys())
    values = list(risks.values())
    
    if not values:
        st.info("No risk factor data available to plot.")
        return

    fig, axes = plt.subplots(1,2, figsize=(14,6))
    axes[0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Organ/Health Risk Distribution (counts)")
    axes[1].barh(labels, values)
    axes[1].set_title("Organ/Health Risk Counts")
    plt.tight_layout()
    st_plot_figure(fig)

# --- Sidebar Input Function ---
def get_user_input(median_values):
    st.sidebar.write("Adjust the **23 health metrics** below:")

    user_data = {}
    for i, (feature, (min_v, max_v, step_v, decimals)) in enumerate(FEATURE_PARAMS.items()):
        
        default_val = median_values.get(feature)
        if default_val is None:
            default_val = DEFAULT_USER_VALUES.get(feature, (min_v + max_v) / 2)

        # FIXED: Ensure default_val is not a list or array
        if isinstance(default_val, (list, np.ndarray)):
            if len(default_val) > 0:
                default_val = float(default_val[0])
            else:
                default_val = (min_v + max_v) / 2
        elif isinstance(default_val, pd.Series):
            default_val = float(default_val.iloc[0])
        
        is_integer_slider = isinstance(min_v, int) and isinstance(max_v, int)
        
        # Ensure default_val is a float/int
        try:
            default_val = float(default_val)
            if is_integer_slider:
                default_val = int(default_val)
        except:
            default_val = (min_v + max_v) / 2
        
        # Ensure default_val is within bounds
        default_val = max(min_v, min(max_v, default_val))
        
        # Label formatting 
        label = f"{i+1}. {feature}"
        if feature == "Age": label += " (Years)"
        elif feature in ["Height_cm", "Spleen_Size_cm"]: label += " (cm)"
        elif feature in ["Weight_kg", "Muscle_Mass_kg"]: label += " (kg)"
        elif feature == "Heart_Rate": label += " (bpm)"
        elif feature == "Lung_Capacity_L": label += " (L)"
        elif feature == "Liver_ALT": label += " (U/L)"
        elif feature == "Kidney_eGFR": label += " (mL/min/1.73m²)"
        elif feature == "Glucose": label += " (mg/dL)"
        elif feature == "Thyroid_TSH": label += " (mIU/L)"
        elif feature == "Stomach_pH": label += " (pH)"
        elif feature == "Bone_Density": label += " (T-score/BMD)"
        elif feature in ["BP_Systolic", "BP_Diastolic"]: label += " (mmHg)"
        elif feature == "Bladder_Capacity_ml": label += " (ml)"
        elif feature == "Vision_Acuity": label += " (0.1 - 1.0)"
        elif feature == "Kidney_Creatinine": label += " (mg/dL)"
        elif feature == "Pancreas_Insulin": label += " (mU/L)"
        elif feature == "Reaction_Time_ms": label += " (ms)"
        elif feature == "Liver_Function_Score": label += " (0-100)"
        elif feature == "Brain_Score": label += " (0-100)"
        elif feature == "Skin_Elasticity": label += " (1-10)"

        # Create the slider (FIXED: ensure all values are simple numbers)
        user_data[feature] = st.sidebar.slider(
            label,
            min_value=float(min_v) if not is_integer_slider else int(min_v),
            max_value=float(max_v) if not is_integer_slider else int(max_v),
            value=float(default_val) if not is_integer_slider else int(default_val),
            step=float(step_v) if not is_integer_slider else int(step_v),
            format=f"%.{decimals}f"
        )

    return pd.DataFrame([user_data])

# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide")
    st.title("🩺 23-Feature Organ Health Risk — Streamlit App (3-Model Comparison)")
    
    MODEL_DIR = "organ_health_model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    st.sidebar.header("Data & Model Settings")
    st.sidebar.info("App Version: V3.2 (3 ML Models + Health Suggestions)") 

    # --- Data Loading ---
    df = None
    load_error = None
    uploaded_file = st.sidebar.file_uploader("Upload a DataFrame (pkl or excel)", type=["pkl", "pickle", "xlsx"])
    use_default_path = st.sidebar.checkbox(f"Try default local path ({DEFAULT_PKL})", value=True) 
    
    if uploaded_file is not None:
        try:
            df = load_data_from_path(uploaded_file)
            st.sidebar.success("Loaded uploaded dataset.")
        except Exception as e:
            load_error = f"Failed to read uploaded file: {e}"

    elif use_default_path and os.path.exists(DEFAULT_PKL):
        try:
            df = load_data_from_path(DEFAULT_PKL)
            st.sidebar.success(f"Loaded dataset from default: {DEFAULT_PKL}")
        except Exception as e:
            load_error = f"Failed to load default file: {e}. Check file format."
    elif use_default_path and not os.path.exists(DEFAULT_PKL):
        load_error = f"Default file not found at: {DEFAULT_PKL}"
        
    if df is None:
        if load_error:
             st.sidebar.error(load_error)
        st.sidebar.warning("Using SYNTHETIC DATA.")
        # --- SYNTHETIC DATA GENERATION ---
        N_SAMPLES = 1000 
        np.random.seed(RANDOM_STATE)
        data = {
            col: np.random.uniform(min_v, max_v, N_SAMPLES).round(decimals) 
            for col, (min_v, max_v, step, decimals) in FEATURE_PARAMS.items()
        }
        data["Person_ID"] = np.arange(N_SAMPLES)
        df = pd.DataFrame(data)
        # --------------------------------------------------------

    # --- Target Creation and Data Cleaning ---
    df = create_target(df)
    df_clean = df.dropna().reset_index(drop=True)
    non_feature_cols = ["Person_ID", "high_risk"]
    feature_columns = [c for c in df_clean.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df_clean[c])]
    
    # Calculate median values for slider defaults
    median_values = {}
    for col in feature_columns:
        try:
            median_val = df_clean[col].median()
            # Ensure it's a scalar, not a list/array
            if isinstance(median_val, (list, np.ndarray, pd.Series)):
                median_val = float(median_val[0]) if len(median_val) > 0 else DEFAULT_USER_VALUES.get(col, 50)
            median_values[col] = float(median_val)
        except Exception:
            median_values[col] = DEFAULT_USER_VALUES.get(col, 50)

    # --- Model Training (3 Models) ---
    models = {}
    results = {}
    
    st.header("1. Model Training & Comparison")
    
    if not XGBOOST_AVAILABLE:
        st.error("ML libraries (xgboost, scikit-learn) not found. Cannot train models. Please install them.")
    elif len(feature_columns) < 2:
        st.warning("Not enough numeric features detected to train on. Training skipped.")
    else:
        # Split data once for all models
        X = df_clean[feature_columns]
        y = df_clean["high_risk"]
        stratify_arg = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_arg)

        # Train models using the cached function
        models, results = train_all_models(X_train, X_test, y_train, y_test, X, y, feature_columns)

        st.success(f"Training complete. {len(models)} Models are ready for prediction.")

        # Display performance comparison table
        st.subheader("Model Performance Summary")
        performance_data = {}
        for name, res in results.items():
            performance_data[name] = {
                "Test Accuracy": f"{res['accuracy']:.4f}",
                "5-Fold CV Mean": f"{res['cv_mean']:.4f}",
            }
        
        performance_df = pd.DataFrame.from_dict(performance_data, orient='index')
        st.dataframe(performance_df)

    # --- Sidebar Input & Live Prediction ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Patient Risk Assessment")

    # Add model selection to the sidebar
    model_names = list(models.keys())
    selected_model_name = st.sidebar.selectbox(
        "Select ML Model for Prediction:",
        options=model_names if model_names else ["No Model Available"],
        index=0 if "XGBoost" in model_names else 0
    )
    
    user_input_df = get_user_input(median_values)

    st.header("2. Live Prediction based on Filters")
    st.write("---")
    
    col_model, col_rule = st.columns(2)
    
    # 2.1 Model Prediction
    with col_model:
        st.subheader(f"Prediction ({selected_model_name})")
        if selected_model_name in models:
            model = models[selected_model_name]
            input_data = user_input_df[feature_columns]
            
            # Predict probability
            try:
                prediction_proba = model.predict_proba(input_data)[0]
                risk_proba = prediction_proba[1] 
                prediction = np.argmax(prediction_proba)

                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK**")
                else:
                    st.success(f"✅ **LOW RISK**")
                    
                st.metric(
                    label="Probability of High Risk (1)", 
                    value=f"{risk_proba * 100:.2f}%", 
                    delta_color="off"
                )
            except Exception:
                st.warning("Model does not support `predict_proba`. Showing binary prediction.")
                prediction = model.predict(input_data)[0]
                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK**")
                else:
                    st.success(f"✅ **LOW RISK**")
                st.metric(label="Binary Prediction", value=f"Risk {prediction}", delta_color="off")
        else:
            st.info("No ML Model available.")

    # 2.2 Rule-Based Prediction (with Suggestions)
    with col_rule:
        st.subheader("Rule-Based Assessment")
        rule_prediction = assess_user_risk_by_rule(user_input_df)
        
        # Calculate specific health suggestions
        detailed_suggestions = generate_health_suggestions(user_input_df)

        if rule_prediction == 1:
            st.error("🚨 **HIGH RISK**")
            st.write("This means **3 or more** vital signs crossed critical thresholds based on the dataset's target definition rules.")
        else:
            st.success("👍 **LOW RISK**")
            st.write("This means **fewer than 3** vital signs crossed critical thresholds.")
        
        st.metric(
            label="Rule-Based Result", 
            value=f"Risk {rule_prediction}", 
            delta_color="off"
        )
        
        # --- Display Suggestions ---
        if detailed_suggestions:
            st.markdown("##### 💡 Targeted Health Suggestions")
            
            suggestions_df = pd.DataFrame(detailed_suggestions)[['Feature', 'Value', 'Suggestion']]
            
            # Custom formatting for the table
            suggestions_df['Value'] = suggestions_df['Value'].apply(lambda x: f"{x:.2f}")

            st.dataframe(
                suggestions_df.style.set_properties(**{'font-size': '10pt', 'background-color': '#f0f2f6'}, subset=['Suggestion']),
                hide_index=True
            )
        else:
            st.info("All key metrics are within defined healthy ranges (Rule-Based).")
        
    st.markdown("---")
    st.markdown(f"**Input Vitals (23 Features):**")
    st.dataframe(user_input_df.T.rename(columns={0: "Value"}))
        
    # --- EDA Section ---
    st.header("3. Exploratory Data Analysis (EDA)")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Show Target Distribution"):
            plot_target_pie(df_clean["high_risk"])
    with col_b:
        with st.expander("Show Raw Data Sample"):
            st.dataframe(df.head(100))
            
    with st.expander("Show Feature Distributions (Histograms & Boxplots)"):
        plot_univariate(df_clean, feature_columns)

    with st.expander("Show Organ Risk Factor Breakdown"):
        plot_organ_risks(df_clean)

    with st.expander("Show Correlation Heatmap"):
        plot_correlation(df_clean, feature_columns)

    # --- Model Diagnostics Section ---
    st.header("4. Model Diagnostics")
    
    if models:
        selected_diag_model_name = st.selectbox(
            "Select Model for Diagnostics:",
            options=model_names,
            index=0
        )
    else:
        selected_diag_model_name = None

    if selected_diag_model_name in models and selected_diag_model_name in results:
        selected_model = models[selected_diag_model_name]
        train_results = results[selected_diag_model_name]
        
        # Feature Importance Plot (available for XGBoost and Random Forest)
        if hasattr(selected_model, 'feature_importances_'):
            st.subheader(f"Feature Importance ({selected_diag_model_name})")
            def plot_feature_importance(model, features, top_n=10):
                imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                imp = imp.sort_values("importance", ascending=False).head(top_n)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.barplot(x="importance", y="feature", data=imp, ax=ax)
                ax.set_title(f"Top {len(imp)} Feature Importances")
                plt.tight_layout()
                st_plot_figure(fig)
            
            plot_feature_importance(selected_model, feature_columns, top_n=min(10, len(feature_columns)))
        elif selected_diag_model_name == 'Logistic Regression' and hasattr(selected_model, 'coef_'):
            # Coefficients for Logistic Regression
            st.subheader(f"Feature Coefficients (Weight) - {selected_diag_model_name}")
            coefs = pd.DataFrame({
                "feature": feature_columns, 
                "coefficient": selected_model.coef_[0]
            })
            coefs["abs_coef"] = np.abs(coefs["coefficient"])
            coefs = coefs.sort_values("abs_coef", ascending=False).head(min(10, len(feature_columns)))
            
            fig, ax = plt.subplots(figsize=(8,6))
            sns.barplot(x="coefficient", y="feature", data=coefs, ax=ax, palette='coolwarm')
            ax.set_title("Top 10 Feature Coefficients (Impact on Risk)")
            plt.tight_layout()
            st_plot_figure(fig)
        else:
             st.info(f"{selected_diag_model_name} does not have a standard feature importance attribute to display.")
        
        st.subheader("Confusion Matrix and Classification Report")
        try:
            y_test = train_results.get("y_test")
            y_pred = train_results.get("y_pred")
            if y_test is not None and y_pred is not None:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
                ax.set_title(f"Confusion Matrix ({selected_diag_model_name})")
                st_plot_figure(fig)

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
        except Exception as e:
            st.warning(f"Could not render model visualizations for {selected_diag_model_name}: {e}")
    else:
        st.info("Model diagnostics skipped because the models were not trained.")

    # --- Save & Download Assets ---
    st.header("5. Save & Download Assets")
    now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Offer download for the selected prediction model
    if selected_model_name in models:
        selected_model_obj = models[selected_model_name]
        model_path = os.path.join(MODEL_DIR, f"{selected_model_name.replace(' ', '_')}_model_{now_stamp}.pkl")
        joblib.dump(selected_model_obj, model_path)
        with open(model_path, "rb") as f:
            st.download_button(label=f"Download trained {selected_model_name} model (.pkl)", data=f, file_name=os.path.basename(model_path))

    model_info = {
        "feature_columns": feature_columns,
        "models_trained": list(models.keys()),
        "selected_model": selected_model_name,
        "version": "3.2.1",
        "description": "23-Feature Organ Health Risk Prediction (Multi-Model)",
        "model_results_summary": {k: {'accuracy': v['accuracy'], 'cv_mean': v['cv_mean']} for k, v in results.items()},
        "generated_at": now_stamp
    }
    meta_path = os.path.join(MODEL_DIR, f"model_info_{now_stamp}.json")
    with open(meta_path, "w") as f:
        json.dump(model_info, f, indent=2)
    with open(meta_path, "rb") as f:
        # FIXED: Changed os.pathasename to os.path.basename
        st.download_button(label="Download model metadata (.json)", data=f, file_name=os.path.basename(meta_path))

    st.markdown("---")
    st.write("🔚 Use the **23 controls in the sidebar** to change settings and view the results in the **Live Prediction** section.")


if __name__ == "__main__":
    main()