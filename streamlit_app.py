import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import shap

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
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    .feature-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
    st.markdown("## Navigation")
    page = st.radio("", ["Prediction", "Model Insights", "Dataset Explorer"])
    
    st.markdown("---")
    st.markdown("## About")
    st.info("""
    This app predicts the likelihood of heart disease based on patient features.
    
    The model used is a Random Forest classifier trained on the Heart Disease UCI dataset.
    
    Made with ❤ by Your Healthcare AI Team
    """)

# Function to load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    # Data Preprocessing
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Normalize data for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    return df, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, rf_model, scaler

df, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, rf_model, scaler = load_data()

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

# Function to make a prediction
def predict_heart_disease(input_data, model, scaler):
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    # Get prediction probability
    prediction_proba = model.predict_proba(scaled_input)[0][1]
    return prediction, prediction_proba

# Prediction Page
if page == "Prediction":
    st.markdown("<h1 class='main-header'>💖 Heart Disease Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Enter patient clinical information below to predict heart disease risk. Hover over each feature for more information.</div>", unsafe_allow_html=True)
    
    # Create tabs for different input methods
    input_method = st.radio("Choose input method:", ["Form Input", "Slider Input", "Quick Sample Cases"])
    
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
            user_input["thal"] = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x], help=feature_descriptions["thal"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif input_method == "Slider Input":
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        
        # General features with sliders
        for col in X.columns:
            if col in ["sex", "fbs", "exang"]:
                user_input[col] = st.select_slider(
                    f"{col}", 
                    options=[0, 1], 
                    value=int(df[col].median()),
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help=feature_descriptions[col]
                )
            elif col in ["cp", "restecg", "slope", "ca", "thal"]:
                options = range(int(df[col].min()), int(df[col].max()) + 1)
                user_input[col] = st.select_slider(
                    f"{col}", 
                    options=options, 
                    value=int(df[col].median()),
                    help=feature_descriptions[col]
                )
            else:
                user_input[col] = st.slider(
                    f"{col}", 
                    float(df[col].min()), 
                    float(df[col].max()), 
                    float(df[col].median()),
                    help=feature_descriptions[col]
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
                col1.write(f"{key}: {value}")
            else:
                col2.write(f"{key}: {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Make prediction button
    predict_button = st.button("Predict Heart Disease Risk", key="predict_button", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            # Convert input to DataFrame
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            prediction, prediction_proba = predict_heart_disease(input_df, rf_model, scaler)
            
            # Display results
            st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a stylized result box
                if prediction == 1:
                    st.markdown(f"""
                    <div class='result-box positive-result'>
                        <h3>⚠ Heart Disease Detected</h3>
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
            
            with col2:
                # Gauge chart for probability visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Heart Disease Risk", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#19D3F3" if prediction_proba < 0.5 else "#FF4B4B"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#c8e6c9'},
                            {'range': [30, 70], 'color': '#fff9c4'},
                            {'range': [70, 100], 'color': '#ffcdd2'}
                        ],
                    }
                    ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for this prediction
            st.markdown("<h3 class='sub-header'>Key Factors Influencing This Prediction</h3>", unsafe_allow_html=True)
            
            # Get feature importances for this specific prediction
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(input_df)
            
            # Convert to DataFrame for easier display
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': np.abs(shap_values[1][0]),
                'Value': input_df.values[0]
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False).head(6)
            
            # Display in a more visual way
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 6 Features Influencing Prediction',
                color='Importance',
                color_continuous_scale='Blues',
                text='Value'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Model Insights Page
elif page == "Model Insights":
    st.markdown("<h1 class='main-header'>Model Insights & Performance</h1>", unsafe_allow_html=True)
    
    # Metrics
    st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Calculate metrics
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
        st.metric("Precision", f"{precision:.2%}")
    
    with col3:
        recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
        st.metric("Recall", f"{recall:.2%}")
    
    # Confusion Matrix
    st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # Feature Importance
    st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Display with Plotly
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Model',
        color='Importance',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Values
    st.markdown("<h2 class='sub-header'>SHAP Values - Feature Impact</h2>", unsafe_allow_html=True)
    
    # Use SHAP for more detailed feature importance
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Plot SHAP summary
        st.write("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns, show=False)
        st.pyplot(fig)

# Dataset Explorer Page
else:
    st.markdown("<h1 class='main-header'>Dataset Explorer</h1>", unsafe_allow_html=True)
    
    # Display dataset overview
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"*Total Samples:* {len(df)}")
        st.write(f"*Positive Cases:* {df['target'].sum()}")
        st.write(f"*Negative Cases:* {len(df) - df['target'].sum()}")
    
    with col2:
        # Pie chart for target distribution
        fig = px.pie(
            df, 
            names='target', 
            title='Distribution of Heart Disease Cases',
            color='target',
            color_discrete_map={0: '#91C4F2', 1: '#FF9999'},
            labels={0: 'No Disease', 1: 'Heart Disease'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Exploration
    exploration_option = st.selectbox(
        "Choose exploration option:",
        ["Data Table", "Correlation Analysis", "Feature Distributions"]
    )
    
    if exploration_option == "Data Table":
        st.markdown("<h3>Data Table</h3>", unsafe_allow_html=True)
        st.dataframe(df.style.background_gradient(cmap='Blues', subset=['age', 'trestbps', 'chol', 'thalach']), use_container_width=True)
        
        # Data description
        st.markdown("<h3>Statistical Summary</h3>", unsafe_allow_html=True)
        st.dataframe(df.describe().style.background_gradient(cmap='Greens'), use_container_width=True)
    
    elif exploration_option == "Correlation Analysis":
        st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plotly heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation with target
        st.markdown("<h3>Feature Correlation with Target</h3>", unsafe_allow_html=True)
        
        target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
        fig = px.bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation='h',
            title='Feature Correlation with Heart Disease',
            color=target_corr.values,
            color_continuous_scale='RdBu_r',
            labels={'x': 'Correlation', 'y': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Feature Distributions
        st.markdown("<h3>Feature Distributions</h3>", unsafe_allow_html=True)
        
        # Select features to plot
        features_to_plot = st.multiselect(
            "Select features to plot:",
            options=X.columns,
            default=['age', 'chol', 'thalach', 'oldpeak']
        )
        
        if features_to_plot:
            # Create distribution plots
            for feature in features_to_plot:
                st.markdown(f"<h4>{feature} Distribution by Heart Disease</h4>", unsafe_allow_html=True)
                
                # Choose plot type based on feature uniqueness
                if df[feature].nunique() <= 5:
                    # For categorical features
                    fig = px.histogram(
                        df,
                        x=feature,
                        color='target',
                        barmode='group',
                        color_discrete_map={0: '#91C4F2', 1: '#FF9999'},
                        labels={'target': 'Heart Disease'},
                        category_orders={feature: sorted(df[feature].unique())}
                    )
                else:
                    # For continuous features
                    fig = px.histogram(
                        df,
                        x=feature,
                        color='target',
                        marginal='box',
                        opacity=0.7,
                        color_discrete_map={0: '#91C4F2', 1: '#FF9999'},
                        labels={'target': 'Heart Disease'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of selected features
        if len(features_to_plot) >= 2:
            st.markdown("<h3>Relationship Between Selected Features</h3>", unsafe_allow_html=True)
            
            # Select two features for scatter plot
            x_feature = st.selectbox("X-axis feature:", options=features_to_plot, index=0)
            y_feature = st.selectbox("Y-axis feature:", options=features_to_plot, index=min(1, len(features_to_plot)-1))
            
            fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                color='target',
                color_discrete_map={0: '#91C4F2', 1: '#FF9999'},
                labels={'target': 'Heart Disease'},
                title=f"{x_feature} vs {y_feature} by Heart Disease Status",
                opacity=0.7,
                size_max=10,
                render_mode='webgl'
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 Heart Disease Prediction App | Developed with ❤ using Streamlit</p>
</div>
""", unsafe_allow_html=True)