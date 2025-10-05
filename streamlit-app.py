# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# --- Configuration and Model Loading ---
st.set_page_config(
    page_title="Smart Diabetes Risk Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and feature list
try:
    model_pipeline = joblib.load('best_diabetes_model.pkl')
    model_features = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Error: Model or feature files not found. Please run the notebook first to generate 'best_diabetes_model.pkl' and 'model_features.pkl'.")
    st.stop()
    
# streamlit_app.py (Near the other model loading)


    


# --- Model and Component Loading (CRITICAL GLOBAL DEFINITION) ---
# NOTE: Ensure all these .pkl files exist in the script directory.
try:
    import joblib
    
    # Load the SHAP model pipeline
    shap_pipeline = joblib.load('shap_diabetes_model.pkl')
    shap_components = joblib.load('shap_components.pkl')
    
    # Define the global components
    SHAP_EXPLAINER = shap_components['explainer']
    SHAP_FEATURE_NAMES = shap_components['feature_names']
    
    # FIX 1: Define PREPROCESSOR globally by extracting it from the loaded pipeline
    PREPROCESSOR = shap_pipeline.named_steps['preprocessor']
    
except FileNotFoundError as e:
    st.error(f"Error loading required files for SHAP analysis: {e}. Please check file paths.")
    st.stop()
except KeyError as e:
    st.error(f"Error extracting component: {e}. Check contents of 'shap_components.pkl'.")
    st.stop()

# -------------------------------------------------------------


# --- Utility Functions ---

def get_risk_category(probability):
    """Determines the risk category based on the predicted probability."""
    if probability < 0.35:
        return "Low Risk", "green"
    elif probability < 0.65:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def generate_personalized_tips(user_data, risk_category):
    """Generates educational tips based on high-risk inputs."""
    tips = set()
    
    if risk_category == "Low Risk":
        tips.add("Keep up the great work! Maintain your healthy lifestyle for long-term health.")
        return list(tips)

    # Lifestyle Factor Checks
    if user_data['BMI'].iloc[0] >= 25:
        tips.add("**Focus on weight management:** Aim for 30 minutes of moderate activity (like brisk walking) most days of the week.")
    if user_data['PhysicalActivity'].iloc[0] < 3:
        tips.add("**Increase activity:** Try to incorporate at least 150 minutes of moderate exercise into your week.")
    if user_data['SleepQuality'].iloc[0] <= 6:
        tips.add("**Improve sleep hygiene:** Aim for 7-9 hours of quality sleep. Establish a consistent routine.")
    if user_data['DietQuality'].iloc[0] <= 4:
        tips.add("**Boost Diet Quality:** Focus on whole foods, lean proteins, and high-fiber vegetables. Limit processed sugars.")
    if user_data['Smoking'].iloc[0] == 1:
        tips.add("**Quit Smoking:** Smoking dramatically increases diabetes risk. Seek support to quit immediately.")

    # Medical History/Clinical Checks
    if user_data['FamilyHistoryDiabetes'].iloc[0] == 1:
        tips.add("**Monitor Blood Sugar:** Since you have a family history, schedule regular check-ups with your doctor for blood sugar screening.")
    if user_data['FastingBloodSugar'].iloc[0] > 100 or user_data['HbA1c'].iloc[0] > 5.6:
        tips.add("**Consult your Physician:** Your current blood sugar readings are elevated. A medical professional should evaluate these immediately.")
    if user_data['Hypertension'].iloc[0] == 1:
        tips.add("**Manage Blood Pressure:** Keep your blood pressure in check. High BP is a major risk factor for diabetes and heart disease.")
    
    if not tips:
        tips.add("Small, consistent changes lead to big results. Review your input data to find an area for improvement!")
        
    return list(tips)[:4] # Limit to 4 key tips

# --- User Input Form (Mapping the Dataset Fields to User-Friendly Questions) ---

# streamlit_app.py

# ... (Previous imports and utility functions) ...

# --- User Input Form (Mapping the Dataset Fields to User-Friendly Questions) ---

# streamlit_app.py

# ... (Previous imports and utility functions) ...

# --- User Input Form (Mapping the Dataset Fields to User-Friendly Questions) ---

def user_input_features():
    # --- 1. Demographic & Socioeconomic (SIDEBAR) ---
    st.sidebar.header('1. Demographic & Socioeconomic')
    
    age = st.sidebar.slider('Age (Years)', 20, 90, 45)
    gender = st.sidebar.selectbox('Gender', options=[('Male', 0), ('Female', 1)], format_func=lambda x: x[0])[1]
    
    ethnicity_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}
    ethnicity = ethnicity_map[st.sidebar.selectbox('Ethnicity', options=list(ethnicity_map.keys()), index=0)]
    
    ses_map = {'Low': 0, 'Middle': 1, 'High': 2}
    ses = ses_map[st.sidebar.selectbox('Socioeconomic Status', options=list(ses_map.keys()), index=1)] 
    
    edu_map = {'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Higher Degree': 3}
    education = edu_map[st.sidebar.selectbox('Highest Education Level', options=list(edu_map.keys()), index=2)]
    
    
    # --- 2. Lifestyle Factors (SIDEBAR) ---
    st.sidebar.header('2. Lifestyle Factors')
    
    bmi = st.sidebar.slider('Body Mass Index (BMI)', 15.0, 40.0, 25.0)
    smoking = st.sidebar.selectbox('Do you currently smoke?', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    alcohol = st.sidebar.slider('Weekly Alcohol Consumption (Units, 0-20)', 0, 20, 0)
    activity = st.sidebar.slider('Weekly Physical Activity (Hours, 0-10)', 0.0, 10.0, 3.5)
    diet = st.sidebar.slider('Diet Quality (0: Poor to 10: Excellent)', 0, 10, 5)
    sleep = st.sidebar.slider('Sleep Quality (4: Very Poor to 10: Excellent)', 4, 10, 7)
    
    # --- 3. Medical History & Medications (SIDEBAR) ---
    st.sidebar.header('3. Medical History & Meds')
    
    col_hist1, col_hist2, col_hist3 = st.sidebar.columns(3)
    fam_hist = col_hist1.selectbox('Family History of Diabetes', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    prev_prediabetes = col_hist2.selectbox('Previous Pre-diabetes Diagnosis', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    htn = col_hist3.selectbox('Hypertension (High BP)', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]

    col_hist4, col_hist5, col_hist6 = st.sidebar.columns(3)
    gdm = col_hist4.selectbox('History of Gestational Diabetes', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    pcos = col_hist5.selectbox('PCOS', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    
    col_med1, col_med2, col_med3 = st.sidebar.columns(3)
    antihyp = col_med1.selectbox('Antihypertensive Meds', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    statins = col_med2.selectbox('Statins', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    antidiab = col_med3.selectbox('Antidiabetic Meds', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]


    # --- 4. Clinical Measurements (COLLAPSIBLE BAR IN MAIN BODY) ---
    
    # Define default values for optional inputs (Mid-range for continuity)
    DEFAULT_SYS_BP = 120
    DEFAULT_DIAS_BP = 80
    DEFAULT_FASTING_SUGAR = 95
    DEFAULT_HBA1C = 5.5
    DEFAULT_SCR = 1.0
    DEFAULT_BUN = 15
    DEFAULT_CHOL_TOTAL = 200
    DEFAULT_CHOL_LDL = 120
    DEFAULT_CHOL_HDL = 45
    DEFAULT_CHOL_TRIG = 150

    
    with st.expander("🔬 4. Clinical Measurements (Optional - Click to Enter Lab Results)", expanded=False):
        st.markdown("Enter your most recent lab values. **If left blank, average values will be used.**")
        
        # Initialize variables with defaults
        systolic_bp = DEFAULT_SYS_BP
        diastolic_bp = DEFAULT_DIAS_BP
        fasting_sugar = DEFAULT_FASTING_SUGAR
        hba1c = DEFAULT_HBA1C
        serum_creat = DEFAULT_SCR
        bun_levels = DEFAULT_BUN
        chol_total = DEFAULT_CHOL_TOTAL
        chol_ldl = DEFAULT_CHOL_LDL
        chol_hdl = DEFAULT_CHOL_HDL
        chol_trig = DEFAULT_CHOL_TRIG
        
        # Use three columns for an organized layout
        col_clin1, col_clin2, col_clin3 = st.columns(3)

        with col_clin1:
            st.subheader("Blood & Kidney")
            systolic_bp = st.slider('Systolic BP (mmHg, 90-180)', 90, 180, DEFAULT_SYS_BP)
            diastolic_bp = st.slider('Diastolic BP (mmHg, 60-120)', 60, 120, DEFAULT_DIAS_BP)
            serum_creat = st.slider('Serum Creatinine (mg/dL, 0.5-5.0)', 0.5, 5.0, DEFAULT_SCR, step=0.1)
            bun_levels = st.slider('BUN Levels (mg/dL, 5-50)', 5, 50, DEFAULT_BUN)

        with col_clin2:
            st.subheader("Glucose/Sugar")
            fasting_sugar = st.slider('Fasting Blood Sugar (mg/dL, 70-200)', 70, 200, DEFAULT_FASTING_SUGAR)
            hba1c = st.slider('HbA1c (%, 4.0-10.0)', 4.0, 10.0, DEFAULT_HBA1C, step=0.1)

        with col_clin3:
            st.subheader("Cholesterol & Fats")
            chol_total = st.slider('Total Cholesterol (mg/dL, 150-300)', 150, 300, DEFAULT_CHOL_TOTAL)
            chol_ldl = st.slider('LDL Cholesterol (mg/dL, 50-200)', 50, 200, DEFAULT_CHOL_LDL)
            chol_hdl = st.slider('HDL Cholesterol (mg/dL, 20-100)', 20, 100, DEFAULT_CHOL_HDL)
            chol_trig = st.slider('Triglycerides (mg/dL, 50-400)', 50, 400, DEFAULT_CHOL_TRIG)
        
    
    # --- 5. Symptoms, QOL & Environment (SIDEBAR) ---
    st.sidebar.header('5. Symptoms & Exposures')
    
    col_sym1, col_sym2, col_sym3 = st.sidebar.columns(3)
    freq_urin = col_sym1.selectbox('Frequent Urination', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    excess_thirst = col_sym2.selectbox('Excessive Thirst', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    weight_loss = col_sym3.selectbox('Unexplained Weight Loss', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    
    col_sym4, col_sym5, col_sym6 = st.sidebar.columns(3)
    blur_vision = col_sym4.selectbox('Blurred Vision', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    slow_sores = col_sym5.selectbox('Slow Healing Sores', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    tingling = col_sym6.selectbox('Tingling in Hands/Feet', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    
    fatigue = st.sidebar.slider('Fatigue Level (0: None to 10: Extreme)', 0, 10, 5)
    qol = st.sidebar.slider('Quality of Life Score (0-100)', 0, 100, 75)
    
    
    col_env1, col_env2, col_env3 = st.sidebar.columns(3)
    heavy_metals = col_env1.selectbox('Heavy Metals Exposure', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    occup_chemicals = col_env2.selectbox('Occupational Chemical Exposure', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])[1]
    water_quality = col_env3.selectbox('Water Quality (0=Good, 1=Poor)', options=[('Good', 0), ('Poor', 1)], format_func=lambda x: x[0])[1]

    # --- 6. Health Behaviors (SIDEBAR) ---
    st.sidebar.header('6. Health Behaviors')
    
    med_freq = st.sidebar.slider('Medical Checkups Frequency (per year)', 0, 4, 1)
    med_adherence = st.sidebar.slider('Medication Adherence Score (0-10)', 0, 10, 8)
    health_literacy = st.sidebar.slider('Health Literacy Score (0-10)', 0, 10, 5)


    # --- Data Dictionary Assembly ---
    # NOTE: The clinical variables (e.g., systolic_bp) hold either the default value 
    # (if the expander was not opened/changed) or the user-set value.
    data = {
        'Age': age, 'Gender': gender, 'Ethnicity': ethnicity, 'SocioeconomicStatus': ses, 'EducationLevel': education,
        'BMI': bmi, 'Smoking': smoking, 'AlcoholConsumption': alcohol, 'PhysicalActivity': activity, 'DietQuality': diet,
        'SleepQuality': sleep, 'FamilyHistoryDiabetes': fam_hist, 'GestationalDiabetes': gdm, 'PolycysticOvarySyndrome': pcos, 
        'PreviousPreDiabetes': prev_prediabetes, 'Hypertension': htn,
        
        # Clinical Measurements
        'SystolicBP': systolic_bp,
        'DiastolicBP': diastolic_bp,
        'FastingBloodSugar': fasting_sugar,
        'HbA1c': hba1c,
        'SerumCreatinine': serum_creat,
        'BUNLevels': bun_levels,
        'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig,
        
        # Medications
        'AntihypertensiveMedications': antihyp, 'Statins': statins, 'AntidiabeticMedications': antidiab,
        
        # Symptoms and QoL
        'FrequentUrination': freq_urin, 'ExcessiveThirst': excess_thirst, 'UnexplainedWeightLoss': weight_loss,
        'FatigueLevels': fatigue, 'BlurredVision': blur_vision, 'SlowHealingSores': slow_sores,
        'TinglingHandsFeet': tingling, 'QualityOfLifeScore': qol,
        
        # Environmental and Occupational Exposures
        'HeavyMetalsExposure': heavy_metals, 'OccupationalExposureChemicals': occup_chemicals, 'WaterQuality': water_quality,
        
        # Health Behaviors
        'MedicalCheckupsFrequency': med_freq, 'MedicationAdherence': med_adherence, 'HealthLiteracy': health_literacy,
    }

    # Create a DataFrame ensuring the columns are in the exact order the model expects
    features = pd.DataFrame([data])
    return features[model_features]

    # streamlit_app.py (Add this function)
def generate_shap_explanation(input_df, model_pipeline, feature_names, top_n=3):
    """
    Calculates SHAP values and generates highly personalized, NLP-style explanations 
    for the top positive and negative contributors based on the user's specific input values.
    """
        # NOTE: Assuming PREPROCESSOR, SHAP_EXPLAINER, np, pd, and plt are globally defined/imported.
    
    try:
        # 1. Transform the single input data point
        X_transformed_single = PREPROCESSOR.transform(input_df) 
        
        # 2. Get SHAP values
        raw_shap_values_output = SHAP_EXPLAINER.shap_values(X_transformed_single)
        
    except Exception as e:
        print(f"FATAL SHAP ERROR: {e}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, 'FATAL SHAP ERROR: An internal error occurred during SHAP calculation (Code 101).', 
                horizontalalignment='center', verticalalignment='center', fontsize=10, color='red')
        ax.axis('off')
        # Return the figure object containing the error message
        return fig
    
    # 3. CRITICAL FIX: Align feature names and handle array structure
    try:
        # Get the true, transformed feature names from the ColumnTransformer
        feature_names = PREPROCESSOR.get_feature_names_out()
        expected_len = len(feature_names) 

        if isinstance(raw_shap_values_output, list) and len(raw_shap_values_output) == 2:
            shap_values = np.array(raw_shap_values_output[1][0]).flatten()
        else:
            shap_values_flat = np.array(raw_shap_values_output).flatten()
            if len(shap_values_flat) == 2 * expected_len:
                shap_values = shap_values_flat[expected_len:] 
            elif len(shap_values_flat) == expected_len:
                shap_values = shap_values_flat
            else:
                raise ValueError(f"SHAP array length ({len(shap_values_flat)}) mismatch feature count ({expected_len}).")

        if len(shap_values) != expected_len:
            raise ValueError(f"Final SHAP array length ({len(shap_values)}) does not match feature name count ({expected_len}).")
            
    except Exception as e:
        print(f"FATAL FEATURE ALIGNMENT ERROR: {e}")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, f"FATAL FEATURE ALIGNMENT ERROR: {e}", 
                horizontalalignment='center', verticalalignment='center', fontsize=10, color='red')
        ax.axis('off')
        return fig


    # 4. Create a DataFrame
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': shap_values,
    })
    
    # 5. --- CRITICAL FILTERING STEP ---
    # Define features to exclude from the visualization because they are non-actionable proxies
    demographic_features_to_exclude = ['Gender', 'Ethnicity', 'SocioeconomicStatus', 'SES', 'Age','EducationLevel', 'FamilyHistoryDiabetes', 'GestationalDiabetes', 'MedicalCheckupsFrequency','HealthLiteracy', 'FrequentUrination','AntidiabeticMedications','TinglingHandsFeet'] 

    # Create a filter that keeps features NOT containing any of the exclusion terms
    # We check if *any* of the key terms are in the transformed feature name (e.g., 'onehotencoder__Ethnicity_1')
    filter_mask = ~shap_df['Feature'].str.contains('|'.join(demographic_features_to_exclude), case=False, na=False)

    # Apply the filter
    filtered_df = shap_df[filter_mask].copy()

    # Get positive contributors (risk-increasing) from the filtered data
    positive_contributors = filtered_df[filtered_df['SHAP_Value'] > 0].sort_values(
        by='SHAP_Value', ascending=False
    )
    
    # Helper to map OHE features back to readable names
    def get_display_name(feature):
        # Clean up OHE names
        if '__' in feature:
            feature = feature.split('__')[-1]
        
        # Map specific OHE categories back to context (adjust to your real maps if needed)
        if 'Ethnicity_' in feature: 
            val_map = {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'}
            try:
                category = int(feature.split('_')[-1])
                return f"Ethnicity ({val_map.get(category, f'Cat {category}')})"
            except:
                pass
        
        return feature.replace('Levels', '').replace('Score', '').replace('scaler__', '').replace('passthrough__', '')
    
    
    # --- START OF PIE CHART GENERATION ---
    
    if positive_contributors.empty:
        # If no positive contributors (after filtering), return an empty plot with a message.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, 'No significant modifiable risk factors detected.', 
                horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax.set_title('Top Modifiable Risk Factors')
        ax.axis('off') 
        return fig

    # Get the top 6 positive risk factors
    top_pos_contributors = positive_contributors.head(6).copy()
    
    # Calculate the total positive SHAP contribution for normalization
    total_positive_shap = top_pos_contributors['SHAP_Value'].sum()

    # Calculate percentage contribution for the pie chart label
    top_pos_contributors['Percentage'] = (top_pos_contributors['SHAP_Value'] / total_positive_shap) * 100
    
    # Create labels: Feature Name (Percentage)
    labels = top_pos_contributors.apply(
        lambda row: f"{get_display_name(row['Feature'])} ({row['Percentage']:.1f}%)", 
        axis=1
    ).tolist()
    
    # Values for the pie chart are the SHAP values 
    sizes = top_pos_contributors['SHAP_Value'].values
    
    # Create the plot
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight') 
    
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Define colors for the slices (using reds/oranges to signify risk)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(sizes)))
    
    wedges, texts = ax.pie(
        sizes, 
        colors=colors,
        startangle=90, 
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )

    # Add legend with detailed labels
    ax.legend(wedges, labels,
              title="Top Modifiable Risk Factors",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    # Set title dynamically
    num_factors = len(top_pos_contributors)
    ax.set_title(f'Top {num_factors} Actionable Factors Increasing Risk', fontsize=16, pad=20)
    ax.axis('equal') 

    return fig







# --- Main Application Logic ---

st.title('🩺 Smart Diabetes Risk Screener')
st.markdown("""
    **Goal:** Predict your risk of developing diabetes based on a comprehensive set of health, lifestyle, and clinical factors.
    Adjust the inputs in the sidebar to see your real-time risk assessment.
""")

input_df = user_input_features()

# Display the user inputs (optional, can be commented out)
# st.subheader('Review Your Inputs')
# st.write(input_df)

if st.button('Calculate Risk', type="primary"):
    with st.spinner('Analyzing your data...'):
        
        # --- ISOLATION STEP 1: Check PREPROCESSOR ---
        # try:
        #     # NOTE: This assumes PREPROCESSOR is defined globally
        #     X_transformed_single = PREPROCESSOR.transform(input_df)
        #     st.write("DEBUG 1: Data successfully transformed.")
            
        #     # --- ISOLATION STEP 2: Check SHAP EXPLAINER ---
        #     try:
        #         # NOTE: This assumes SHAP_EXPLAINER is defined globally
        #         shap_values = SHAP_EXPLAINER.shap_values(X_transformed_single)
        #         st.write("DEBUG 2: SHAP values successfully generated (raw).")
                
        #         # Now, call the function with the transformed data if possible (or rewrite the function to accept it)
        #         # For now, let's just make the simple call, as the crash is likely earlier.
        #         explanation_points = generate_shap_explanation(
        #             input_df, # Keep original data for NLP context
        #             None, 
        #             SHAP_FEATURE_NAMES, 
        #             top_n=3
        #         )
        #         st.write("DEBUG 3: Full explanation function returned data.")
                
        #     except Exception as e:
        #         st.error(f"FATAL ERROR during SHAP calculation (DEBUG 2): {e}")
        #         explanation_points = ["Error: Could not calculate SHAP values."]

        # except Exception as e:
        #     st.error(f"FATAL ERROR during Data Transformation (DEBUG 1): {e}")
        #     explanation_points = ["Error: Could not transform input data."]
        
        # 1. Prediction
        # The model pipeline handles both scaling and one-hot encoding
        prediction_proba = model_pipeline.predict_proba(input_df)[:, 1][0]
        risk_percentage = round(prediction_proba * 100)
        
        # 2. Interpretation
        risk_category, color_code = get_risk_category(prediction_proba)
        
        # 3. Personalized Tips
        tips = generate_personalized_tips(input_df, risk_category)

        # --- Visual Output ---
        st.header('Prediction Result')
        
        col_risk, col_gauge, col_tips = st.columns([1.5, 1, 2.5])

        with col_risk:
            st.markdown(f"### Your Risk Assessment: **{risk_category}**")
            st.markdown(f"## **{risk_percentage}%**")
            
            # Simple color-coded card
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {color_code}; color: white; text-align: center;">
                **{risk_category}**
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <p style='font-size: small; margin-top: 10px;'>
            This prediction is based on a robust Machine Learning model and should not replace professional medical advice.
            </p>
            """, unsafe_allow_html=True)

        with col_gauge:
            # Simple gauge meter simulation using an emoji and text
            emoji = ""
            if color_code == "green":
                emoji = "✅"
            elif color_code == "orange":
                emoji = "⚠️"
            else:
                emoji = "🚨"

            st.metric(label="Risk Probability", value=f"{risk_percentage}%")
            st.progress(risk_percentage / 100)
            
        # -----------------------------------------------------------------------
        # --- NEW SHAP EXPLANATION LOGIC BLOCK (Inserted Here) ---
        # -----------------------------------------------------------------------
        if risk_percentage >= 30:
            st.markdown("---")
            st.header('💡 Top Risk Drivers Visualized:')
            
            # We are now generating the explanation IMMEDIATELY because risk >= 30
            
            with st.spinner('Calculating personalized factor impact...'):
                
                # Generate the custom NLP-style explanation points
                explanation_points = generate_shap_explanation(
                    input_df, 
                    None, 
                    SHAP_FEATURE_NAMES, 
                    top_n=3
                )
                
            # ------------------------------------------------------------------
            # --- START OF IMMEDIATE DEBUG DISPLAY CODE ---
            # We are reusing the simple st.text display that we know works in principle
            # ------------------------------------------------------------------
            

            # --- START OF MODIFIED DISPLAY LOGIC ---

            # Check the type of output to determine how to display it
            if isinstance(explanation_points, plt.Figure):
                # Success: Display the Matplotlib Figure (Pie Chart)
                st.pyplot(explanation_points, use_container_width=True)

                st.markdown(
                    """
                    <div style='background-color:#fff7e6; padding: 10px; border-radius: 8px; border-left: 5px solid #ffaa00;'>
                    **How to read this chart:** The chart above shows the **relative importance** of the 
                    top factors pushing your risk score up (based on SHAP values). 
                    The size of each slice represents the magnitude of its impact on the prediction.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # If it's a list and starts with "FATAL", display the error.
            elif isinstance(explanation_points, list) and explanation_points and explanation_points[0].startswith("FATAL"):
                st.error(explanation_points[0])
            
            # Catch any unexpected output
            else:
                st.error("The explanation function returned an unexpected result and could not generate the chart.")

             # --- END OF MODIFIED DISPLAY LOGIC ---
            
            # ------------------------------------------------------------------
            # --- END OF IMMEDIATE DEBUG DISPLAY CODE ---
            # ------------------------------------------------------------------

            
            # Optional: Add tips section back if needed
            with col_tips:
                st.markdown("### Personalized Health & Lifestyle Tips")
                for tip in tips:
                    st.info(f"💡 {tip}")


                            
                # -----------------------------------------------------------------------
                # The generate_shap_explanation function definition follows below (as provided by the user)
                # -----------------------------------------------------------------------

