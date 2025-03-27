
Streamlit application to analyze how lifestyle habits (sleep, screen time, activity, etc.)
affect mental and physical health (stress, happiness). The data comes from two CSV files
(mental_health.csv and smartwatch_health.csv), potentially AI-generated or enriched.

Author: Florian Boulay


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

###############################################################################
# STREAMLIT CONFIG - MUST BE FIRST
###############################################################################
st.set_page_config(page_title="Health & Lifestyle Insights App", layout="wide")

###############################################################################
# DATA LOADING AND PREPARATION
###############################################################################

# Add error handling for file loading
@st.cache_data
def load_data():
    try:
        df_mental = pd.read_csv("mental_health.csv")
        df_smartwatch = pd.read_csv("smartwatch_health.csv")
        return df_mental, df_smartwatch, None
    except FileNotFoundError as e:
        error_message = f"Error loading data: {e}. Please ensure CSV files are in the correct location."
        return None, None, error_message
    except Exception as e:
        error_message = f"Unexpected error loading data: {e}"
        return None, None, error_message

df_mental, df_smartwatch, load_error = load_data()

st.title("📊 Health & Lifestyle Insights: Data-Driven Well-being Dashboard")
st.markdown(
    "**This application analyzes how various lifestyle habits—sleep, screen time, physical activity, etc.—"
    "influence stress and happiness.**\n"
    "It offers correlation exploration, average comparisons, and happiness prediction via a Random Forest model."
)

# Display error if data loading failed
if load_error:
    st.error(load_error)
    st.stop()

# Focus on df_mental for this app
df = df_mental.copy()

# Get actual columns from the dataframe
available_columns = df.columns.tolist()

# Define expected columns with fallbacks
def get_column(expected_columns, default=None):
    """Return the first column from expected_columns that exists in the dataframe"""
    for col in expected_columns:
        if col in available_columns:
            return col
    return default

# Map expected columns to actual columns
stress_col = get_column(["stress_level", "stress"])
happiness_col = get_column(["happiness_score", "happiness"])
sleep_col = get_column(["sleep_duration_hrs", "sleep_hours", "sleep"])
screen_col = get_column(["screen_time_hours", "screen_time", "screen"])
work_col = get_column(["work_hours_per_week", "work_hours", "work"])
activity_col = get_column(["activity_level", "activity"])
social_col = get_column(["social_interaction_score", "social_score", "social"])
age_col = get_column(["age"])

# Identify available numeric columns
numeric_cols = [col for col in [stress_col, happiness_col, sleep_col, screen_col, 
                               work_col, social_col, age_col] if col is not None]

# Sidebar Navigation
section = st.sidebar.radio(
    "Navigate",
    [
        "1. Overview",
        "2. Explore Correlations",
        "3. Personalized Recommendation",
        "4. Compare Yourself to the Average",
        "5. Predict Your Happiness"
    ]
)

###############################################################################
# SECTION 1: OVERVIEW
###############################################################################
if section == "1. Overview":
    st.header("📌 Project Overview")
    st.markdown(
        "This project highlights the importance of factors such as **sleep**, **screen time**, "
        "**physical activity**, and **work hours** on **stress** and **happiness**.\n\n"
        "After an initial data cleaning process (using SQL) and discovering missing values "
        "and structural limitations, an **AI-based approach** was used to generate or refine synthetic data, "
        "preserving the original structure and distributions.\n"
        "This ensures a reliable and scalable analysis."
    )
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Available Columns")
    st.info("The following columns are available in the dataset:\n" + 
            "\n".join([f"- {col}" for col in available_columns]))

###############################################################################
# SECTION 2: EXPLORE CORRELATIONS
###############################################################################
elif section == "2. Explore Correlations":
    st.header("🔍 Explore Lifestyle Correlations")

    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to display correlations.")
    else:
        st.subheader("Correlation Matrix")
        corr_data = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("Custom Scatter Plot")
        x_var = st.selectbox("X-Axis", numeric_cols, index=0)
        y_options = [col for col in numeric_cols if col != x_var]
        if not y_options:
            st.warning("Need at least two different numeric columns for a scatter plot.")
        else:
            y_var = st.selectbox("Y-Axis", y_options, index=0)
            fig2 = px.scatter(df, x=x_var, y=y_var, title=f"{y_var} vs {x_var}")
            st.plotly_chart(fig2, use_container_width=True)

###############################################################################
# SECTION 3: PERSONALIZED RECOMMENDATION
###############################################################################
elif section == "3. Personalized Recommendation":
    st.header("🌟 Personalized Well-being Tips")

    st.markdown("Provide some basic lifestyle inputs to get data-driven advice.")

    sleep = st.slider("Average hours of sleep (per night)", 3, 10, 6)
    screen = st.slider("Average daily screen time (hours)", 1, 16, 6)
    work = st.slider("Work hours per week", 0, 80, 40)
    
    activity_options = ["Sedentary", "Moderate", "Active", "Highly Active"]
    activity = st.selectbox("Activity Level", activity_options)
    
    stress = st.slider("Your stress level (1-10)", 1, 10, 5)

    st.subheader("🔎 Recommendations")

    if sleep < 6:
        st.warning("Increasing sleep can significantly reduce stress.")
    if screen > 8:
        st.warning("High screen time can affect mood. Try to reduce it if possible.")
    if work > 50:
        st.warning("You work a lot. A better work-life balance may help your well-being.")
    if stress > 7:
        st.info("High stress detected. Consider mindfulness or relaxation techniques.")
    if activity == "Sedentary":
        st.info("Moving to at least moderate activity can improve both mood and stress levels.")

###############################################################################
# SECTION 4: COMPARE YOURSELF TO THE AVERAGE
###############################################################################
elif section == "4. Compare Yourself to the Average":
    st.header("📊 Compare Yourself to the Overall Dataset")
    st.markdown("Enter your own data to see how you rank compared to the sample.")

    user_sleep = st.slider("Your Sleep (hrs)", 3, 10, 7)
    user_screen = st.slider("Your Screen Time (hrs/day)", 1, 16, 5)
    user_stress = st.slider("Your Stress (1-10)", 1, 10, 5)

    col1, col2, col3 = st.columns(3)

    if sleep_col in df.columns:
        avg_sleep = df[sleep_col].mean()
        with col1:
            st.metric(
                label="Your vs. Avg Sleep (hrs)",
                value=f"{user_sleep}",
                delta=f"{round(user_sleep - avg_sleep,2)} vs Avg"
            )
    else:
        with col1:
            st.info("Sleep data not available in dataset")
            
    if screen_col in df.columns:
        avg_screen = df[screen_col].mean()
        with col2:
            st.metric(
                label="Your vs. Avg Screen Time (hrs)",
                value=f"{user_screen}",
                delta=f"{round(user_screen - avg_screen,2)} vs Avg"
            )
    else:
        with col2:
            st.info("Screen time data not available in dataset")
            
    if stress_col in df.columns:
        avg_stress = df[stress_col].mean()
        with col3:
            st.metric(
                label="Your vs. Avg Stress",
                value=f"{user_stress}",
                delta=f"{round(user_stress - avg_stress,2)} vs Avg"
            )
    else:
        with col3:
            st.info("Stress data not available in dataset")

    if stress_col in df.columns:
        st.subheader("Overall Stress Distribution")
        fig3 = px.box(df, y=stress_col, points="all", title="Stress Distribution")
        st.plotly_chart(fig3)
        st.markdown(
            "This boxplot shows how stress is distributed across the dataset. "
            "Each point represents an individual. Compare your own stress level to this distribution."
        )
    else:
        st.warning("Cannot display stress distribution: stress data not available in the dataset.")

###############################################################################
# SECTION 5: PREDICT YOUR HAPPINESS
###############################################################################
elif section == "5. Predict Your Happiness":
    st.header("🤖 Happiness Prediction (Random Forest)")
    
    if happiness_col is None:
        st.error("Cannot predict happiness: happiness score column is missing from the dataset.")
        st.stop()
        
    st.markdown("Enter your lifestyle habits to get an estimated happiness score (0-15).")

    # Create input sliders based on available data
    input_values = {}
    
    if sleep_col:
        input_values[sleep_col] = st.slider("Sleep Duration (hrs)", 3.0, 10.0, 6.0)
    
    if stress_col:
        input_values[stress_col] = st.slider("Stress Level (1-10)", 1, 10, 5)
    
    if screen_col:
        input_values[screen_col] = st.slider("Screen Time (hrs/day)", 1.0, 16.0, 6.0)
    
    if work_col:
        input_values[work_col] = st.slider("Work Hours per Week", 0, 80, 40)
    
    if social_col:
        input_values[social_col] = st.slider("Social Interaction Score (0-10)", 0.0, 10.0, 5.0)
    
    if age_col:
        input_values[age_col] = st.slider("Age", 16, 80, 30)

    # Define features for the model
    features = list(input_values.keys())
    
    if not features:
        st.error("No suitable features available for prediction.")
        st.stop()
        
    # Prepare data for the model
    df_model = df.dropna(subset=[happiness_col] + features).copy()
    
    if len(df_model) <= 10:
        st.warning("Not enough data to train the model properly. The dataset has too many missing values.")
        st.stop()
    
    X = df_model[features]
    y = df_model[happiness_col]

    # Train the model
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Create input dataframe for prediction
        user_input = {feature: [value] for feature, value in input_values.items()}
        df_user = pd.DataFrame(user_input)
        
        # Make prediction
        prediction = model.predict(df_user)[0]
        
        # Calculate possible range by looking at the actual data
        min_happiness = df_model[happiness_col].min()
        max_happiness = df_model[happiness_col].max()
        
        st.success(f"Estimated Happiness Score: {round(prediction, 2)} (Range in data: {min_happiness} to {max_happiness})")
        
        # Show feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        fig = px.bar(feature_importance, x='Feature', y='Importance', 
                     title="Which factors affect happiness the most?")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error building prediction model: {e}")