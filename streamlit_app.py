import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration with custom theme
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .positive-result {
        background-color: #ffcdd2;
        border-left: 5px solid #f44336;
    }
    .negative-result {
        background-color: #c8e6c9;
        border-left: 5px solid #4caf50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .feature-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .feature-bar {
        background-color: #e3f2fd;
        border-radius: 5px;
        margin-bottom: 8px;
        overflow: hidden;
    }
    .feature-fill {
        background-color: #1e88e5;
        color: white;
        padding: 8px;
        text-align: left;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and information
with st.sidebar:
    st.markdown("# 💖 Heart Disease Predictor")
    st.markdown("## Navigation")
    page = st.radio("", ["Prediction", "Model Metrics", "Data Overview"])
    
    st.markdown("---")
    st.markdown("## About")
    st.info("""
    This app predicts the likelihood of heart disease based on patient features.
    
    The model used is a Random Forest classifier trained on the Heart Disease UCI dataset.
    
    Made with ❤️ by Your Healthcare AI Team
    """)

# Function to load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    # Data Preprocessing
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train RandomForest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return df, X, y, X_train, X_test, y_train, y_test, rf_model

df, X, y, X_train, X_test, y_train, y_test, rf_model = load_data()

# Feature descriptions for tooltips
feature_descriptions = {
    "age": "Age of the patient in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3, with 0 being typical angina)",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"
}

# Prediction Page
if page == "Prediction":
    st.markdown("<h1 class='main-header'>💖 Heart Disease Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Enter patient clinical information below to predict heart disease risk. Hover over each feature for more information.</div>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    input_method = st.radio("Choose input method:", ["Form Input", "Quick Sample Cases"])
    
    user_input = {}
    
    if input_method == "Form Input":
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        # Demographics
        with col1:
            st.markdown("#### Demographics")
            user_input["age"] = st.number_input("Age", 20, 100, 55, help=feature_descriptions["age"])
            user_input["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=feature_descriptions["sex"])
        
        # Vitals
        with col2:
            st.markdown("#### Vital Signs")
            user_input["trestbps"] = st.number_input("Resting BP (mm Hg)", 90, 200, 130, help=feature_descriptions["trestbps"])
            user_input["chol"] = st.number_input("Cholesterol (mg/dl)", 100, 600, 250, help=feature_descriptions["chol"])
            user_input["fbs"] = st.selectbox("High Fasting Blood Sugar", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=feature_descriptions["fbs"])
            user_input["thalach"] = st.number_input("Max Heart Rate", 60, 220, 150, help=feature_descriptions["thalach"])
        
        # Cardiac Metrics
        with col3:
            st.markdown("#### Cardiac Metrics")
            user_input["cp"] = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x], help=feature_descriptions["cp"])
            user_input["restecg"] = st.selectbox("ECG Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x], help=feature_descriptions["restecg"])
            user_input["exang"] = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=feature_descriptions["exang"])
            user_input["oldpeak"] = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, help=feature_descriptions["oldpeak"])
            user_input["slope"] = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], help=feature_descriptions["slope"])
        
        # Additional Tests
        st.markdown("#### Additional Tests")
        col4, col5 = st.columns(2)
        with col4:
            user_input["ca"] = st.selectbox("Major Vessels Count", [0, 1, 2, 3, 4], help=feature_descriptions["ca"])
        
        with col5:
            # Mapping for Thalassemia values
            thal_mapping = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

            # Selectbox for Thalassemia with corrected indexing
            user_input["thal"] = st.selectbox(
                "Thalassemia", 
                options=[1, 2, 3], 
                format_func=lambda x: thal_mapping.get(x, "Unknown"), 
                help=feature_descriptions["thal"]
            )

        st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # Quick Sample Cases
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        
        sample_cases = {
            "Low Risk Case": {
                "age": 45, "sex": 0, "cp": 0, "trestbps": 120, "chol": 180, 
                "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, 
                "oldpeak": 0.5, "slope": 0, "ca": 0, "thal": 1
            },
            "Medium Risk Case": {
                "age": 58, "sex": 1, "cp": 1, "trestbps": 140, "chol": 240, 
                "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0, 
                "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
            },
            "High Risk Case": {
                "age": 65, "sex": 1, "cp": 3, "trestbps": 160, "chol": 300, 
                "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1, 
                "oldpeak": 2.5, "slope": 2, "ca": 3, "thal": 3
            }
        }
        
        selected_case = st.selectbox("Select a sample case:", list(sample_cases.keys()))
        user_input = sample_cases[selected_case]
        
        # Show the case details
        st.write("Case details:")
        col1, col2 = st.columns(2)
        for i, (key, value) in enumerate(user_input.items()):
            if i % 2 == 0:
                col1.write(f"**{key}**: {value}")
            else:
                col2.write(f"**{key}**: {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Make prediction button
    predict_button = st.button("Predict Heart Disease Risk", key="predict_button", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            # Convert input to DataFrame
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            prediction = rf_model.predict(input_df)[0]
            prediction_proba = rf_model.predict_proba(input_df)[0][1]
            
            # Display results
            st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
            
            # Create a stylized result box
            if prediction == 1:
                st.markdown(f"""
                <div class='result-box positive-result'>
                    <h3>⚠️ Heart Disease Detected</h3>
                    <p>The model predicts a <b>{prediction_proba:.1%}</b> probability of heart disease.</p>
                    <p><b>Recommendation:</b> Further cardiac evaluation is strongly recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box negative-result'>
                    <h3>✅ No Heart Disease Detected</h3>
                    <p>The model predicts a <b>{1-prediction_proba:.1%}</b> probability of no heart disease.</p>
                    <p><b>Recommendation:</b> Continue regular health check-ups.</p>
                </div>
                """, unsafe_allow_html=True)

            
            # Progress bar for risk visualization
            st.markdown("### Risk Level")
            st.progress(prediction_proba)
            st.write(f"Risk Score: {prediction_proba:.1%}")
            
            # Feature importance for this prediction
            st.markdown("<h3 class='sub-header'>Key Factors Influencing This Prediction</h3>", unsafe_allow_html=True)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(6)
            
            # Create a simple bar chart with HTML
            for idx, row in feature_importance.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                width = int(importance * 100)  # Scale to percentage
                st.markdown(f"""
                <div class="feature-bar">
                    <div class="feature-fill" style="width: {width}%">
                        {feature}: {importance:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Model Metrics Page
elif page == "Model Metrics":
    st.markdown("<h1 class='main-header'>Model Performance Metrics</h1>", unsafe_allow_html=True)
    
    # Calculate metrics
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix for display
    from collections import Counter
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_test)
    true_positive = sum((a == 1 and b == 1) for a, b in zip(y_test, y_pred))
    true_negative = sum((a == 0 and b == 0) for a, b in zip(y_test, y_pred))
    false_positive = sum((a == 0 and b == 1) for a, b in zip(y_test, y_pred))
    false_negative = sum((a == 1 and b == 0) for a, b in zip(y_test, y_pred))
    
    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display metrics in a nicer format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.2%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """.format(accuracy), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.2%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """.format(precision), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.2%}</div>
            <div class="metric-label">Recall</div>
        </div>
        """.format(recall), unsafe_allow_html=True)
    
    # Confusion Matrix
    st.markdown("<h2 class='sub-header'>Confusion Matrix</h2>", unsafe_allow_html=True)
    
    # Create a simple HTML table for confusion matrix
    st.markdown("""
    <table style="width:100%; border-collapse: collapse; text-align: center; margin: 20px 0;">
        <tr>
            <th colspan="2" rowspan="2" style="background-color: #f1f8e9; padding: 10px;"></th>
            <th colspan="2" style="background-color: #f1f8e9; padding: 10px;">Predicted</th>
        </tr>
        <tr>
            <th style="background-color: #f1f8e9; padding: 10px;">No Disease (0)</th>
            <th style="background-color: #f1f8e9; padding: 10px;">Heart Disease (1)</th>
        </tr>
        <tr>
            <th rowspan="2" style="background-color: #f1f8e9; padding: 10px;">Actual</th>
            <th style="background-color: #f1f8e9; padding: 10px;">No Disease (0)</th>
            <td style="background-color: #c8e6c9; padding: 15px; font-weight: bold;">{}</td>
            <td style="background-color: #ffcdd2; padding: 15px; font-weight: bold;">{}</td>
        </tr>
        <tr>
            <th style="background-color: #f1f8e9; padding: 10px;">Heart Disease (1)</th>
            <td style="background-color: #ffcdd2; padding: 15px; font-weight: bold;">{}</td>
            <td style="background-color: #c8e6c9; padding: 15px; font-weight: bold;">{}</td>
        </tr>
    </table>
    """.format(true_negative, false_positive, false_negative, true_positive), unsafe_allow_html=True)
    
    # Display explanation of confusion matrix
    st.markdown("""
    <div class="info-box">
        <h4>Confusion Matrix Explanation:</h4>
        <ul>
            <li><strong>True Negatives ({}):</strong> Correctly predicted as No Heart Disease</li>
            <li><strong>False Positives ({}):</strong> Incorrectly predicted as Heart Disease</li>
            <li><strong>False Negatives ({}):</strong> Incorrectly predicted as No Heart Disease</li>
            <li><strong>True Positives ({}):</strong> Correctly predicted as Heart Disease</li>
        </ul>
    </div>
    """.format(true_negative, false_positive, false_negative, true_positive), unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Create a simple bar chart with HTML
    for idx, row in feature_importance.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        width = int(importance * 100)  # Scale to percentage
        st.markdown(f"""
        <div class="feature-bar">
            <div class="feature-fill" style="width: {width}%">
                {feature}: {importance:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature descriptions
    st.markdown("<h3>Feature Descriptions</h3>", unsafe_allow_html=True)
    
    for feature, description in feature_descriptions.items():
        st.markdown(f"**{feature}**: {description}")

# Dataset Explorer Page
else:
    st.markdown("<h1 class='main-header'>Dataset Explorer</h1>", unsafe_allow_html=True)
    
    # Display dataset overview
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"**Total Samples:** {len(df)}")
        st.write(f"**Positive Cases:** {df['target'].sum()}")
        st.write(f"**Negative Cases:** {len(df) - df['target'].sum()}")
        st.write(f"**Number of Features:** {df.shape[1] - 1}")
    
    with col2:
        # Create a simple HTML chart for target distribution
        positive_count = df['target'].sum()
        negative_count = len(df) - positive_count
        positive_percent = positive_count / len(df) * 100
        negative_percent = negative_count / len(df) * 100
        
        st.markdown("""
        <h3>Target Distribution</h3>
        <div style="background-color: #e3f2fd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center;">
                <div style="width: {}%; background-color: #FF9999; padding: 10px; text-align: center; border-radius: 5px 0 0 5px;">
                    Heart Disease: {}%
                </div>
                <div style="width: {}%; background-color: #91C4F2; padding: 10px; text-align: center; border-radius: 0 5px 5px 0;">
                    No Disease: {}%
                </div>
            </div>
        </div>
        """.format(positive_percent, round(positive_percent, 1), negative_percent, round(negative_percent, 1)), unsafe_allow_html=True)
    
    # Data Exploration
    exploration_option = st.selectbox(
        "Choose exploration option:",
        ["Data Table", "Statistical Summary", "Feature Analysis"]
    )
    
    if exploration_option == "Data Table":
        st.markdown("<h3>Data Table</h3>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
    
    elif exploration_option == "Statistical Summary":
        st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        # Additional stats
        st.markdown("<h3>Additional Statistics</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum().to_dict())
        
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes.astype(str).to_dict())
    
    else:  # Feature Analysis
        st.markdown("<h3>Feature Analysis</h3>", unsafe_allow_html=True)
        
        # Select a feature to analyze
        selected_feature = st.selectbox("Select a feature to analyze:", X.columns)
        
        # Display basic statistics
        st.write(f"**Statistics for {selected_feature}:**")
        st.write(df[selected_feature].describe())
        
        # Create a simple histogram with HTML
        st.markdown("<h4>Distribution by Heart Disease Status</h4>", unsafe_allow_html=True)
        
        # Compute values for display
        positive_values = df[df['target'] == 1][selected_feature]
        negative_values = df[df['target'] == 0][selected_feature]
        
        # Create bins for histogram
        if df[selected_feature].nunique() <= 5:
            # For categorical features, use value counts
            pos_counts = positive_values.value_counts().sort_index()
            neg_counts = negative_values.value_counts().sort_index()
            
            # Create a simple HTML table for categorical values
            st.markdown("<div style='overflow-x: auto;'>", unsafe_allow_html=True)
            st.markdown("<table style='width: 100%; border-collapse: collapse;'>", unsafe_allow_html=True)
            st.markdown("<tr><th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>Value</th><th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>Heart Disease</th><th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>No Disease</th></tr>", unsafe_allow_html=True)
            
            for value in sorted(df[selected_feature].unique()):
                pos_count = pos_counts.get(value, 0)
                neg_count = neg_counts.get(value, 0)
                st.markdown(f"<tr><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{value}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{pos_count}</td><td style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>{neg_count}</td></tr>", unsafe_allow_html=True)
            
            st.markdown("</table>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # For continuous features, describe by target
            st.write("**By Heart Disease Status:**")
            st.write(df.groupby('target')[selected_feature].describe())
            
            # Show correlation with target
            correlation = df[[selected_feature, 'target']].corr().iloc[0, 1]
            st.write(f"**Correlation with Heart Disease:** {correlation:.3f}")
            
            # Show risk by quartiles
            st.markdown("<h4>Risk by Quartiles</h4>", unsafe_allow_html=True)
            
            quartiles = pd.qcut(df[selected_feature], 4, duplicates='drop')
            risk_by_quartile = df.groupby(quartiles)['target'].mean()
            
            for i, (group, risk) in enumerate(risk_by_quartile.items()):
                st.markdown(f"""
                <div class="feature-bar">
                    <div class="feature-fill" style="width: {risk * 100}%">
                        {group}: {risk:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 Heart Disease Prediction App | Developed with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)