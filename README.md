# 📊 Integrated Analysis of Mental and Physical Health: Links Between Lifestyle Habits and Biometric Measures

## 📌 Project Overview

In today's fast-paced world, mental and physical health are deeply connected to lifestyle choices. This project investigates how habits such as physical activity, sleep, and screen time influence stress and happiness levels. By leveraging smartwatch biometric data and self-reported mental health information, we aim to uncover key behavioral patterns that impact well-being.

To achieve this, I conducted an in-depth data analysis using statistical methods and machine learning models. The project initially started with real-world data that I cleaned using SQL. However, due to missing values and structural limitations, I decided to generate a synthetic dataset using Artificial Intelligence (AI). The synthetic dataset was carefully designed to replicate the original structure, distributions, and relationships found in the cleaned dataset, allowing for a more precise and controlled analysis on Python.

## 📌 Research Questions & Objectives

- How do lifestyle habits impact stress, sleep, and health scores?
- Does physical activity influence stress levels?
- Which biometric features (heart rate, step count, sleep) are most correlated with stress?
- What are the most important factors for predicting happiness?

By addressing these questions, this study aims to identify behavioral changes that can improve mental and physical well-being.

## 📌 Data Sources & Structure

This study is based on two datasets, each containing key health and lifestyle variables:

🩺 Health & Biometric Dataset

This dataset includes biometric and activity data collected from wearable devices.

Variables:

- `user_id`: Unique identifier for each individual  
- `heart_rate_bpm`: Average heart rate (beats per minute)  
- `blood_oxygen_pct`: Blood oxygen saturation (%)  
- `step_count`: Total daily steps  
- `sleep_duration_hrs`: Total sleep duration in hours  
- `activity_level`: Physical activity level (categorical)  
- `stress_level`: Measured stress level  
- `stress_category`: Categorized stress level (e.g., Low, Moderate, High)  
- `health_score`: Composite score reflecting overall physical health  
- `health_category`: Categorized health status (e.g., Poor, Average, Good)  


🧠 Mental Health Dataset

This dataset contains self-reported mental health and lifestyle indicators.

Variables:

- `country`: Country of residence  
- `age`: Age of the individual  
- `gender`: Gender  
- `exercise_level`: Frequency or intensity of exercise  
- `diet_type`: Type of diet followed  
- `sleep_hours`: Average sleep duration per night  
- `stress_level`: Self-reported stress level  
- `stress_category`: Categorized stress level (Low, Moderate, High)  
- `mental_health_condition`: Reported mental health condition (e.g., Anxiety, Depression, None)  
- `work_hours_per_week`: Weekly work hours  
- `screen_time_hours`: Daily screen time (in hours)  
- `social_interaction_score`: Score reflecting social interaction or connectedness  
- `happiness_score`: Self-reported happiness score  
- `happiness_category`: Categorized happiness level (Low, Moderate, High)  
- `interaction_category`: Category of social interaction (e.g., Low, Medium, High)  


## 📌 Methodology & Approach

1. 📊 Data Cleaning & Preprocessing: The initial dataset underwent cleaning in SQL, fixing missing values and inconsistencies before switching to AI-generated data.
2. 📉 Exploratory Data Analysis (EDA): Visualized key trends and relationships using Python, Pandas Seaborn and Matplotlib.
3. 📈 Statistical Analysis: Performed correlation analysis, ANOVA tests, and Tukey post-hoc tests to understand significant factors.
4. 🤖 Machine Learning Model: Implemented a Random Forest Regressor to predict happiness based on lifestyle factors.
5. 📊 Data Visualization: Developed insightful charts to illustrate findings.

## 📌 Key Insights

✅ Stress levels are negatively correlated with sleep duration – those who sleep less tend to have higher stress. 
✅ Screen time is negatively associated with happiness – excessive screen time may reduce overall well-being. 
✅Physical activity has a complex relationship with stress – moderate activity levels correlate with lower stress, but extremes (too low or too high) may increase stress. 
✅ Heart rate is a significant indicator of stress – higher heart rates correlate with increased stress levels.
✅ Machine learning analysis identified stress level as the strongest predictor of happiness.

## 📌 Project Structure

- `Health_Analysis_Lifestyle.ipynb` → Jupyter Notebook with full analysis.
- `Health_Analysis_Lifestyle.py` → Executable Python script.
- `mental_health.xlsx` → Processed mental health dataset.
- `smartwatch_health.xlsx` → Processed biometric dataset.
- `app.py`→ Streamlit app.
- `README.md` → Project documentation (this file).

## 📌 How to Run the Project

1. Clone the repository:
    
    ```bash
    git clone https://github.com/YOUR-USERNAME/Health_Analysis_Lifestyle.git
    ```
    
2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Open Jupyter Notebook:
    
    ```bash
    jupyter notebook Health_Analysis_Lifestyle.ipynb
    ```
4. Run the streamlit app : 
    ```bash
    streamlit run app.py
⚠️ Make sure you are in the project folder before running the command.
    
  
5. Run all cells to reproduce the analysis.

 📩 Contact

For any questions, feel free to connect:

- 📧 Email: florian.boulay@hec.ca
- 🔗 LinkedIn: https://www.linkedin.com/in/florian-boulay-524298179/

🚀 If you found this project helpful, consider giving it a ⭐ on GitHub!

