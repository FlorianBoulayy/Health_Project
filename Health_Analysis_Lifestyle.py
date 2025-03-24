#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 📊 Integrated Analysis of Mental and Physical Health: Links Between Lifestyle Habits and Biometric Measures

## Understanding the Impact of Lifestyle Habits on Stress, Sleep, and Happiness  

### 📌 Project Overview 

In today fast-paced world, mental and physical health are deeply connected to lifestyle choices.

get_ipython().run_line_magic('pinfo', 'happiness')
get_ipython().run_line_magic('pinfo', 'being')

To answer these questions, I analyzed the relationship between lifestyle habits and health metrics, using smartwatch data and self-reported mental health information. Through statistical analysis, machine learning, and AI-generated data, I uncovered valuable insights into how stress, sleep, and daily activities shape overall well-being.
During the initial data cleaning process with SQL, I identified missing values, inconsistencies, and structural limitations in the original datasets. To ensure a robust and scalable analysis, I leveraged Artificial Intelligence (AI) to generate a synthetic dataset, carefully preserving the original structure, distributions, and relationships.
This approach allowed me to maintain data integrity while gaining deeper insights into health trends, making the analysis more controlled, precise, and actionable using Python and machine learning

### 🔍 Research Questions & Objectives 
✅ How does physical activity level impact stress, sleep, and health scores? 
✅ What is the relationship between screen time and happiness? 
✅ How do sleep patterns influence stress levels?  
✅ Which biometric features (heart rate, step count, sleep) are most correlated with stress? 
✅ What are the most important factors for predicting happiness? 

By addressing these questions, this study aims to identify behavioral changes that can improve mental and physical well-being.  


## 📂 Data Sources & Structure 
This study is based on two datasets, each containing key health and lifestyle variables:

### 1️⃣ Smartwatch Health Dataset 📲 
This dataset includes biometric and activity data collected from wearable devices.  
Variables:  
- User Information: `user_id` (Unique identifier)  
- Biometric Data: `heart_rate_bpm`, `blood_oxygen_pct`, `step_count`, `sleep_duration_hrs`  
- Activity Levels: `activity_level` (Sedentary, Moderate, Active, Highly Active)  
- Stress Metrics: `stress_level` (Scale from 1-10), `stress_category`  
- Health Score Metrics: `health_score` (Scale from 1-15), `health_category`  

### 2️⃣ Mental Health & Lifestyle Dataset   
This dataset contains self-reported mental health indicators and lifestyle habits.  
Variables:  
- User Demographics: `age`, `gender`, `country`  
- Lifestyle Factors: `diet_type`, `exercise_level`, `work_hours_per_week`, `screen_time_hours`  
- Mental Health Metrics: `stress_level`, `happiness_score`, `mental_health_condition`  
- Social & Behavioral Factors: `social_interaction_score`, `interaction_category`  
- Sleep Data: `sleep_hours`, `sleep_category`  

## 🛠️ Methodology & Workflow
This project follows a structured data science pipeline:

### 1️⃣ Data Preprocessing & Cleaning**  
✅ SQL-based cleaning to remove inconsistencies and missing values  
✅ AI-generated synthetic dataset to overcome data limitations  
✅ Normalization & formatting to ensure compatibility in Python  

### 2️⃣ Exploratory Data Analysis (EDA)
📊 Key Visualizations:  
- Distribution of stress, sleep, and activity levels  
- Correlations between biometric and mental health factors 
- Comparison of lifestyle habits across activity groups  

📌 Objective: Identify initial trends & anomalies  

### 3️⃣ Statistical Analysis
📈 Tests Performed: 
- ANOVA & Tukey’s HSD to test group differences  
- Correlation heatmaps to measure relationships between variables  
- Regression models to quantify predictive power  

📌 Objective: Understand statistical significance of factors impacting stress & happiness  

###4️⃣ Machine Learning Analysis
Random Forest Model to predict happiness score
- Feature importance analysis to determine key drivers of well-being  
- Performance evaluation using Mean Absolute Error (MAE) & R² Score 

📌 Objective: Identify which variables have the strongest predictive power  


🚀 Let’s begin the analysis! 


# In[1]:


import pandas as pd

df_smartwatch = pd.read_csv(r"C:\Users\flori\OneDrive\Documents\Health_Project\smartwatch_health.csv")
df_mental = pd.read_csv(r"C:\Users\flori\OneDrive\Documents\Health_Project\mental_health.csv")

print(df_smartwatch.head())  # Verify the first rows
print(df_mental.head())


# In[3]:


# Check first 5 rows of smartwatch dataset
print("Smartwatch Health Dataset - First 5 rows:")
print(df_smartwatch.head())

# Check first 5 rows of mental health dataset
print("\nMental Health Dataset - First 5 rows:")
print(df_mental.head())

# Check general info including data types and non-null counts
print("\nSmartwatch Health Dataset - Info:")
print(df_smartwatch.info())

print("\nMental Health Dataset - Info:")
print(df_mental.info())

# Check for missing values in both datasets
print("\nMissing values in Smartwatch dataset:")
print(df_smartwatch.isnull().sum())

print("\nMissing values in Mental Health dataset:")
print(df_mental.isnull().sum())

# Descriptive statistics to understand numerical data distribution
print("\nSmartwatch Health Dataset - Descriptive Statistics:")
print(df_smartwatch.describe())

print("\nMental Health Dataset - Descriptive Statistics:")
print(df_mental.describe())



# In[7]:


# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Set aesthetic parameters for consistent style
sns.set(style="whitegrid")

# Set figure size for clear visualization
plt.figure(figsize=(14, 6))

# Plot stress level distribution for smartwatch_health dataset
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
sns.countplot(x='stress_level', data=df_smartwatch, hue='stress_level', palette="Blues_d", legend=False)
plt.title('Stress Level Distribution - Smartwatch Health')
plt.xlabel('Stress Level')
plt.ylabel('Count')

# Plot stress level distribution for mental_health dataset
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
sns.countplot(x='stress_level', data=df_mental, hue='stress_level', palette="Greens_d", legend=False)
plt.title('Stress Level Distribution - Mental Health')
plt.xlabel('Stress Level')
plt.ylabel('Count')

# Display plots neatly
plt.tight_layout()
plt.show()


# In[ ]:


### Stress Level Distribution: Smartwatch vs. Mental Health

Above, we compare how stress levels are distributed in both datasets:
- Smartwatch Health (left): Data captured via wearable devices (heart rate, steps, blood oxygen etc... )
- Mental Health (right): Self-reported data such as sleep hours, age, gender, screen time, happiness etc...

This side-by-side view allows us to see if stress levels follow a similar pattern in device-based measurements versus self-reported measures. Large differences might indicate discrepancies in how users perceive stress compared to biometric indicators.


# In[11]:


# Step 2 Analyze correlations between numerical variables in both datasets

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Correlation heatmap for Smartwatch Health Dataset
plt.figure(figsize=(10, 7))
sns.heatmap(df_smartwatch.corr(numeric_only=True), annot=True, cmap='Blues', linewidths=0.5)
plt.title('Correlation Heatmap - Smartwatch Health Dataset')
plt.show()

# Correlation heatmap for Mental Health Dataset
plt.figure(figsize=(10, 7))
sns.heatmap(df_mental.corr(numeric_only=True), annot=True, cmap='Greens', linewidths=0.5)
plt.title('Correlation Heatmap - Mental Health Dataset')
plt.show()


# In[ ]:


### Comparing Correlations in Smartwatch vs. Mental Health Datasets

Here, we visualize two separate correlation heatmaps:

1. Smartwatch Dataset (TOP): Focused on biometric/activity metrics ( heart rate, blood oxygen, steps etc). We can see which physiological measures are related (e.g., whether higher step counts correlate with better health scores or lower stress).

2. Mental Health Dataset (BOTTOM): Involves self-reported factors (Hapiness score, sleep hours, screen time etc). This helps us identify if higher screen time correlates with lower happiness, or if more sleep is associated with less stress.


# In[13]:


# Merge both datasets on 'stress_level' to create a combined dataset for deeper analysis

# Merge datasets based on the common column 'stress_level'
combined_df = pd.merge(df_smartwatch, df_mental, on='stress_level', suffixes=('_smartwatch', '_mental'))

# Verify the merged dataset structure and first rows
print("Combined Dataset - First 5 rows:")
print(combined_df.head())

print("\nCombined Dataset - Info:")
print(combined_df.info())

print("\nCombined Dataset - Shape:", combined_df.shape)


# In[15]:


# Aggregate smartwatch_health dataset by stress_level
smartwatch_agg = df_smartwatch.groupby('stress_level', as_index=False).agg({
    'heart_rate_bpm': 'mean',
    'blood_oxygen_pct': 'mean',
    'step_count': 'mean',
    'sleep_duration_hrs': 'mean',
    'health_score': 'mean'
})

# Aggregate mental_health dataset by stress_level
mental_agg = df_mental.groupby('stress_level', as_index=False).agg({
    'age': 'mean',
    'sleep_hours': 'mean',
    'work_hours_per_week': 'mean',
    'screen_time_hours': 'mean',
    'social_interaction_score': 'mean',
    'happiness_score': 'mean'
})

# Join aggregated datasets on stress_level
aggregated_df = pd.merge(smartwatch_agg, mental_agg, on='stress_level', how='inner')

#  Inspect aggregated joined dataset
print("Aggregated and Joined Dataset (by Stress Level):")
print(aggregated_df.head(10))


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style for consistency and aesthetics
sns.set_style('whitegrid')

# Plot 1: Stress Level vs. Heart Rate

plt.figure(figsize=(10, 6))
sns.lineplot(data=aggregated_df, x='stress_level', y='heart_rate_bpm', marker='o', color='red')
plt.title('Average Heart Rate BPM by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Average Heart Rate (BPM)')
plt.show()


# In[19]:


# Plot 2: Stress Level vs. Sleep Duration and Sleep Hours

plt.figure(figsize=(10, 6))
sns.lineplot(data=aggregated_df, x='stress_level', y='sleep_duration_hrs', marker='o', label='Smartwatch Sleep Hours')
sns.lineplot(data=aggregated_df, x='stress_level', y='sleep_hours', marker='o', label='Reported Sleep Hours')
plt.title('Average Sleep Duration by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Hours of Sleep')
plt.legend()
plt.show()


# In[25]:


#Plot 3: Stress Level vs. Step Count 

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated_df, x='stress_level', y='step_count', hue='stress_level', palette='viridis', legend=False)
plt.title('Average Step Count by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Average Daily Steps')
plt.show()


# In[27]:


# Plot 4: Stress Level vs. Happiness Score
plt.figure(figsize=(10, 6))
sns.lineplot(data=aggregated_df, x='stress_level', y='happiness_score', marker='o', color='green')
plt.title('Average Happiness Score by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Average Happiness Score')
plt.show()


# In[31]:


# Plot 5: Stress Level vs. Screen Time Hours
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated_df, x='stress_level', y='screen_time_hours', hue= 'stress_level', palette='magma')
plt.title('Average Screen Time by Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Average Screen Time (Hours)')
plt.show()


# In[33]:


#Graph 1 (Heart Rate): Clearly visualize the direct correlation between stress and average heart rate.
#Graph 2 (Sleep Duration): Compare and validate sleep data recorded by the smartwatch with self-reported sleep hours.
#Graph 3 (Step Count): Analyze whether stress levels significantly impact average daily physical activity.
#Graph 4 (Happiness Score): Show the impact of stress on users' perceived happiness.
#Graph 5 (Screen Time): Identify whether high stress levels correspond to a significant increase in screen time usage.


# In[35]:


import scipy.stats as stats

# Compute Pearson correlation coefficients and p-values
pearson_corr, pearson_pval = stats.pearsonr(aggregated_df["stress_level"], aggregated_df["heart_rate_bpm"])
print(f"Pearson Correlation (Stress Level vs Heart Rate): {pearson_corr:.4f}, p-value: {pearson_pval:.4f}")

pearson_corr, pearson_pval = stats.pearsonr(aggregated_df["stress_level"], aggregated_df["happiness_score"])
print(f"Pearson Correlation (Stress Level vs Happiness Score): {pearson_corr:.4f}, p-value: {pearson_pval:.4f}")

pearson_corr, pearson_pval = stats.pearsonr(aggregated_df["stress_level"], aggregated_df["screen_time_hours"])
print(f"Pearson Correlation (Stress Level vs Screen Time): {pearson_corr:.4f}, p-value: {pearson_pval:.4f}")

# Spearman correlation for non-linear relationships
spearman_corr, spearman_pval = stats.spearmanr(aggregated_df["stress_level"], aggregated_df["sleep_duration_hrs"])
print(f"Spearman Correlation (Stress Level vs Sleep Duration): {spearman_corr:.4f}, p-value: {spearman_pval:.4f}")


# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set a modern style for the plots
sns.set_style("whitegrid")

# Define figure and axes for subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Stress Level vs Heart Rate
sns.regplot(x='stress_level', y='heart_rate_bpm', data=aggregated_df, ax=axes[0, 0], scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
axes[0, 0].set_title("Stress Level vs Heart Rate", fontsize=14)
axes[0, 0].set_xlabel("Stress Level")
axes[0, 0].set_ylabel("Average Heart Rate (BPM)")

# Plot 2: Stress Level vs Happiness Score 
sns.regplot(x='stress_level', y='happiness_score', data=aggregated_df, ax=axes[0, 1], scatter_kws={'alpha':0.5}, line_kws={"color": "blue"})
axes[0, 1].set_title("Stress Level vs Happiness Score", fontsize=14)
axes[0, 1].set_xlabel("Stress Level")
axes[0, 1].set_ylabel("Average Happiness Score")

# Plot 3: Stress Level vs Screen Time
sns.regplot(x='stress_level', y='screen_time_hours', data=aggregated_df, ax=axes[1, 0], scatter_kws={'alpha':0.5}, line_kws={"color": "green"})
axes[1, 0].set_title("Stress Level vs Screen Time", fontsize=14)
axes[1, 0].set_xlabel("Stress Level")
axes[1, 0].set_ylabel("Average Screen Time (Hours)")

# Plot 4: Stress Level vs Sleep Duration
sns.regplot(x='stress_level', y='sleep_duration_hrs', data=aggregated_df, ax=axes[1, 1], scatter_kws={'alpha':0.5}, line_kws={"color": "purple"})
axes[1, 1].set_title("Stress Level vs Sleep Duration", fontsize=14)
axes[1, 1].set_xlabel("Stress Level")
axes[1, 1].set_ylabel("Average Sleep Duration (Hours)")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[ ]:


### Four RegPlots: Stress Level vs. Key Variables

Plot 1: Stress Level vs. Heart Rate (Red)
- Shows whether higher stress correlates with elevated average heart rate.  
- Helps confirm if physiological arousal increases as stress goes up.

Plot 2: Stress Level vs. Happiness (Blue)
- Explores if heightened stress is associated with a drop in happiness.  
- Useful for identifying an inverse relationship between these factors.

Plot 3: Stress Level vs. Screen Time (Green)
- Checks if people experiencing more stress also spend more hours on screens.  
- Highlights potential coping or avoidance behaviors.

Plot 4: Stress Level vs. Sleep Duration (Purple)
- Examines whether increased stress reduces the average hours of sleep.  
- Helps see if poor sleep could be linked to high stress. 


# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set figure and GridSpec (2 rows, 3 columns)
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3)

# Define the five axes, skipping the bottom-right slot
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

# ---- Boxplot 1: Heart Rate vs Stress Level ----
sns.boxplot(
    x="stress_level",
    y="heart_rate_bpm",
    data=df_smartwatch,
    hue="stress_level",
    ax=ax1,
    palette="Reds"
)
ax1.set_title("Heart Rate by Stress Level")
ax1.set_xlabel("Stress Level")
ax1.set_ylabel("Heart Rate (BPM)")
ax1.legend().set_visible(False)

# ---- Boxplot 2: Step Count vs Stress Level ----
sns.boxplot(
    x="stress_level",
    y="step_count",
    data=df_smartwatch,
    hue="stress_level",
    ax=ax2,
    palette="Blues"
)
ax2.set_title("Step Count by Stress Level")
ax2.set_xlabel("Stress Level")
ax2.set_ylabel("Daily Steps")
ax2.legend().set_visible(False)

# ---- Boxplot 3: Sleep Duration vs Stress Level ----
sns.boxplot(
    x="stress_level",
    y="sleep_duration_hrs",
    data=df_smartwatch,
    hue="stress_level",
    ax=ax3,
    palette="Purples"
)
ax3.set_title("Sleep Duration by Stress Level")
ax3.set_xlabel("Stress Level")
ax3.set_ylabel("Sleep Duration (Hours)")
ax3.legend().set_visible(False)

# ---- Boxplot 4: Screen Time vs Stress Level ----
sns.boxplot(
    x="stress_level",
    y="screen_time_hours",
    data=df_mental,
    hue="stress_level",
    ax=ax4,
    palette="Greens"
)
ax4.set_title("Screen Time by Stress Level")
ax4.set_xlabel("Stress Level")
ax4.set_ylabel("Screen Time (Hours)")
ax4.legend().set_visible(False)

# ---- Boxplot 5: Happiness Score vs Stress Level ----
sns.boxplot(
    x="stress_level",
    y="happiness_score",
    data=df_mental,
    hue="stress_level",
    ax=ax5,
    palette="Oranges"
)
ax5.set_title("Happiness Score by Stress Level")
ax5.set_xlabel("Stress Level")
ax5.set_ylabel("Happiness Score")
ax5.legend().set_visible(False)

plt.tight_layout()
plt.show()



# In[ ]:


## Boxplots: Stress Level vs. Key Metrics

Five boxplots to explore how stress level relates to different health indicators from both the smartwatch and mental health datasets.

1. Heart Rate by Stress Level 
   - Displays how the heart rate (BPM)varies for each stress level.  
   - Useful for spotting if highly stressed individuals tend to have an elevated heart rate.

2. Step Count by Stress Level  
   - Shows daily step count grouped by stress level.  
   - Helps indicate whether low-stress individuals are more physically active, or if high-stress leads to fewer steps.

3. Sleep Duration by Stress Level  
   - Illustrates the hours of sleep recorded by the smartwatch at each stress level.  
   - Can reveal if high stress correlates with shorter sleep.

4. Screen Time by Stress Level
   - Comes from the mental health dataset, indicating how many hours are spent on screens per day.  
   - Examines whether screen usage rises with higher stress levels.

5. Happiness Score by Stress Level  
   - Also from the mental health data, showing self-reported happiness.  
   - Highlights a potential inverse relationship between stress and happiness.

Together, these boxplots provide a quick visual comparison of multiple health and lifestyle metrics as stress levels change.


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set figure size with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Stress Level & Health Score by Activity Level", fontsize=16, fontweight="bold")

# Boxplot for Stress Level by Activity Level
sns.boxplot(data=df_smartwatch, x='activity_level', y='stress_level', hue='activity_level', dodge=False, legend=False, palette='coolwarm', ax=axes[0])
axes[0].set_title("Stress Level Distribution by Activity Level", fontsize=14)
axes[0].set_xlabel("Activity Level")
axes[0].set_ylabel("Stress Level (1-10)")

# Boxplot for Health Score by Activity Level
sns.boxplot(data=df_smartwatch, x='activity_level', y='health_score', hue='activity_level', dodge=False, legend=False, palette='viridis', ax=axes[1])
axes[1].set_title("Health Score Distribution by Activity Level", fontsize=14)
axes[1].set_xlabel("Activity Level")
axes[1].set_ylabel("Health Score (5-15)")

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust title position

# Show the plot
plt.show()



# In[78]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set figure size with two subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Stress Level & Sleep Duration by Activity Level", fontsize=16, fontweight="bold")

# Boxplot for Stress Level by Activity Level
sns.boxplot(data=df_smartwatch, x='activity_level', y='stress_level', hue='activity_level', legend=False, palette='coolwarm', ax=axes[0])
axes[0].set_title("Stress Level Distribution by Activity Level", fontsize=14)
axes[0].set_xlabel("Activity Level")
axes[0].set_ylabel("Stress Level (1-10)")

# Boxplot for Sleep Duration by Activity Level
sns.boxplot(data=df_smartwatch, x='activity_level', y='sleep_duration_hrs', hue='activity_level', legend=False, palette='viridis', ax=axes[1])
axes[1].set_title("Sleep Duration Distribution by Activity Level", fontsize=14)
axes[1].set_xlabel("Activity Level")
axes[1].set_ylabel("Sleep Duration (Hours)")

# Adjust layout for better spacing
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust title position

# Show the plot
plt.show()



# In[ ]:


## Stress Level & Health Score by Activity Level

Here we compare stress_level and **health_score across different activity levels (Sedentary, Moderate, Highly Active, Active). This gives us a sense of whether:

- More active individuals experience lower stress on average.  
- Physical activity correlates with a higher overall health score.

By putting stress level on one plot and **health score on the other, we can see if the same activity level that lowers stress also increases health.

## Stress Level & Sleep Duration by Activity Level

These boxplots examine how stress levels and sleep duration vary with activity level. We can see if being more active is linked to:
- Lower stress levels, potentially indicating exercise helps manage stress.
- Longer sleep durations, suggesting that active individuals maintain healthier sleep habits.


# In[90]:


import pandas as pd
import scipy.stats as stats

# 🔹 Ensure column names are lowercase for consistency
df_smartwatch.rename(columns=lambda x: x.strip().lower(), inplace=True)
df_mental.rename(columns=lambda x: x.strip().lower(), inplace=True)

# 🔹 Define datasets and grouping variables
datasets = {
    'smartwatch': {'df': df_smartwatch, 'group_by': 'activity_level'},
    'mental_health': {'df': df_mental, 'group_by': 'exercise_level'}  # Alternative grouping
}

# 🔹 Define relevant metrics
metrics_dict = {
    'smartwatch': ['stress_level', 'health_score', 'sleep_duration_hrs'],
    'mental_health': ['happiness_score']
}

# Store ANOVA results
anova_results = {}

#  Loop through each dataset
for dataset_name, dataset_info in datasets.items():
    df = dataset_info['df']
    group_column = dataset_info['group_by']
    
    if group_column not in df.columns:
        print(f"⚠️ Skipping {dataset_name}: '{group_column}' column not found!")
        continue  # Skip if grouping column is missing
    
    for metric in metrics_dict.get(dataset_name, []):
        if metric in df.columns:
            groups = [df[df[group_column] == level][metric] 
                      for level in df[group_column].unique()]
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results[f"{dataset_name}_{metric}"] = {'F-Statistic': f_stat, 'P-Value': p_value}

#  Convert results into a DataFrame
anova_df = pd.DataFrame(anova_results).T

#  Display results
print("🔹 ANOVA Test Results:")
print(anova_df)

# Optionally, display first few rows for better readability
anova_df.head()



# In[ ]:


### Key Takeaways
1. Health Score is significantly different across activity levels. This suggests that more active groups may have higher (or lower) health scores compared to less active groups.  
2. Stress Level, Sleep Duration, and Happiness Score do not differ significantly by activity/exercise level, based on these p-values.  
   - For stress level and happiness score, we can infer that activity level alone might not explain their variations.  
   - For sleep duration, it appears there’s no strong statistical difference among different activity groups in this dataset.


# In[92]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Ensure 'activity_level' and 'health_score' are in your dataframe
print("\nColumns in df_smartwatch:", df_smartwatch.columns.tolist())

# Filter rows with valid health_score & activity_level
df_valid = df_smartwatch.dropna(subset=['activity_level', 'health_score'])

# Perform Tukey's test
tukey = pairwise_tukeyhsd(
    endog=df_valid['health_score'],            # Numeric data (health_score)
    groups=df_valid['activity_level'],         # Groups (activity_level)
    alpha=0.05                                 # Significance level
)

# Display the results
print("\nTukey’s HSD Test Results for Health Score by Activity Level:\n")
print(tukey.summary())


# In[ ]:


### Interpretation
- False under “Reject Ho?” means there is no statistically significant difference between those two activity levels in terms of health_score.
- True indicates a significant difference. For instance, the active group differs significantly from moderate and sedentary, but not from highly active.

### Key Findings
1. Active & Highly Active groups show no significant difference in health score (p > 0.05).  
2. Every other comparison is significant (p < 0.05), suggesting meaningful differences in health score across those pairs of activity levels.  

In short, being active or highly active yields similar health scores, while moderate and sedentary levels are significantly lower. This aligns with our ANOVA result indicating that activity level is a strong predictor of health score.


# In[ ]:





# In[150]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set the aesthetic style
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "0.96", "grid.color": "0.9"})
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.labelsize': 12
})

# Create the ridgeline plot
fig, ax = plt.subplots(figsize=(14, 10))

# Get unique activity levels and sort them
activity_levels = sorted(df_smartwatch['activity_level'].unique())

# Create a custom colormap
colors = sns.color_palette("viridis", len(activity_levels))

# Plot each activity level separately with offset
for i, activity in enumerate(activity_levels):
    subset = df_smartwatch[df_smartwatch['activity_level'] == activity]
    
    # Add vertical offset
    y_offset = i * 0.3
    
    # Check if subset has data
    if len(subset) > 0:
        # Create the KDE plot with density estimation
        sns.kdeplot(
            data=subset,
            x="stress_level",
            ax=ax,
            fill=True,
            alpha=0.7,
            color=colors[i],
            linewidth=2,
            label=f"{activity}"
        )
        
        # Manually get the current line just created
        if len(ax.lines) > 0:
            line = ax.lines[-1]
            x_data, y_data = line.get_data()
            
            # Clear the previous line and redraw with offset
            line.remove()
            
            # Draw the filled area with offset
            ax.fill_between(x_data, y_data + y_offset, y_offset, alpha=0.7, color=colors[i])
            ax.plot(x_data, y_data + y_offset, color=colors[i], linewidth=2)

# Add annotation
plt.annotate(
    "Higher peaks indicate more common stress levels\nfor that activity group",
    xy=(0.75, 0.15), xycoords="figure fraction",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="lightgray"),
    fontsize=10
)

# Title and labels
plt.title("Stress Level Distribution by Activity Level", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Stress Level (0-100)", fontsize=14)
plt.ylabel("Density (with vertical offset)", fontsize=14)

# Legend
plt.legend(
    title="Activity Level",
    title_fontsize=12,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    edgecolor="lightgray"
)

# Add grid lines
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Ensure proper spacing
plt.tight_layout()
plt.show()


# In[ ]:


### Ridgeline Plot: Stress Level Distribution by Activity Level

This plot shows kernel density estimates for stress levels (ranging 0–10 or 0–100) across different activity levels (Sedentary, Moderate, Highly Active, Active). Each “layer” is offset vertically to highlight where most data points cluster for each activity group.

- Purpose: Compare how stress is distributed among users with different activity levels.  
- Interpretation: Higher peaks indicate common stress levels for that group. Overlaps or differing peak positions can show if active users tend to report lower (or higher) stress compared to sedentary users.


# In[144]:


import seaborn as sns
import matplotlib.pyplot as plt

# Vérifier quelles colonnes existent
print("Colonnes disponibles dans df_smartwatch:", df_smartwatch.columns)

# Sélectionner uniquement les colonnes qui existent
valid_columns = [col for col in ['stress_level', 'heart_rate_bpm', 'sleep_duration_hrs', 'screen_time_hours', 'activity_level'] if col in df_smartwatch.columns]

# Create a COPY of the dataframe (this is the key fix)
df_subset = df_smartwatch[valid_columns].copy()

# Convertir activity_level en type catégoriel pour un meilleur affichage
if 'activity_level' in df_subset.columns:
    df_subset['activity_level'] = df_subset['activity_level'].astype(str)

# Création du FaceGrid avec scatter plots
g = sns.FacetGrid(df_subset, col="activity_level", hue="stress_level", palette="viridis", col_wrap=3, height=4, sharex=False, sharey=False)

# Vérifier quelles colonnes sont disponibles avant de les utiliser
if "sleep_duration_hrs" in df_subset.columns and "heart_rate_bpm" in df_subset.columns:
    g.map_dataframe(sns.scatterplot, x="sleep_duration_hrs", y="heart_rate_bpm", alpha=0.7)
else:
    print("⚠️ Certaines colonnes manquent, ajustez le code si nécessaire.")

# Customisation de l'affichage
g.set_axis_labels("Sleep Duration (Hours)", "Heart Rate (BPM)")
g.set_titles(col_template="Activity Level: {col_name}")
g.add_legend()
plt.show()


# In[ ]:


### Scatter Plots: Sleep Duration vs. Heart Rate Across Activity Levels

This visualization consists of four scatter plots, each corresponding to a different activity level (Sedentary, Moderate, Highly Active, Active). The x-axis represents sleep duration (in hours), and the y-axis represents heart rate (BPM). The color gradient indicates stress levels, with darker colors representing lower stress levels and lighter colors representing higher stress levels.

- Purpose: Explore the relationship between sleep duration, heart rate, and stress across different activity groups.  
- Observations:
  - Users with shorter sleep durations (<6 hours) generally tend to have higher heart rates.
  - Individuals with longer sleep durations (~7-8 hours) tend to have lower heart rates, often associated with lower stress levels.
  - Across all activity levels, users with higher stress levels (light colors) appear to have a wider range of heart rates, potentially indicating physiological stress responses.
  - There is no drastic difference between activity levels, though highly active individuals seem to maintain more stable heart rates.

These insights suggest that sleep duration may have a significant impact on heart rate and stress levels, regardless of activity level.


# In[110]:


df = df.copy()  # Ensure we're working on a separate copy, not modifying the original dataset

# Create a new column categorizing screen_time_hours without modifying original values
df["Screen_Time_Category"] = pd.cut(df["screen_time_hours"], bins=[0, 3, 6, 24], labels=["Low", "Medium", "High"])

# Check if the column is created properly
print(df[["screen_time_hours", "Screen_Time_Category"]].head())


# In[142]:


import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Normality and homogeneity test for screen time
shapiro_screen = stats.shapiro(df["stress_level"])
levene_screen = stats.levene(
    *[df[df["Screen_Time_Category"] == s]["stress_level"] for s in df["Screen_Time_Category"].unique()]
)

# ANOVA for screen time categories
anova_screen = ols("stress_level ~ C(Screen_Time_Category)", data=df).fit()
anova_screen_results = sm.stats.anova_lm(anova_screen, typ=2)

# Tukey test if ANOVA is significant - Fix for FutureWarning
# Use .iloc to access by position or .loc to access by label name
anova_p_value = anova_screen_results.iloc[0, anova_screen_results.columns.get_loc("PR(>F)")]
tukey_screen = pairwise_tukeyhsd(df["stress_level"], df["Screen_Time_Category"]) if anova_p_value < 0.05 else None

# Print results
print("\nANOVA - Screen Time:")
print(anova_screen_results)
if tukey_screen:
    print("\nTukey Test - Screen Time:")
    print(tukey_screen)


# In[ ]:


### ANOVA & Tukey Test: Screen Time and Stress Level

This analysis aims to evaluate whether different categories of screen time (Low, Medium, High) significantly impact stress levels. The analysis consists of:

1. ANOVA Test (Analysis of Variance): Determines if there is a statistically significant difference in stress levels between the screen time categories.
2. Tukey’s HSD Test: A post-hoc test used when ANOVA finds a significant difference, to pinpoint which specific groups differ.

#### **ANOVA Results**
- The F-statistic is 1813.39, with a p-value of 0.0 (very significant).
- This suggests that screen time categories significantly impact stress levels.

#### **Tukey's HSD Test Results**
- High vs. Low Screen Time: The mean difference is -5.36, with a p-value < 0.05, meaning individuals with high screen time experience significantly higher stress compared to those with low screen time.
- High vs. Medium Screen Time: The mean difference is **-3.65, also statistically significant.
- Medium vs. Low Screen Time: The mean difference is **1.7, indicating medium screen time is also associated with higher stress than low screen time.

#### **Key Takeaways**
- Higher screen time is strongly linked to increased stress levels.
- There is a clear, statistically significant difference between all screen time categories.
- This suggests that reducing screen time may be a potential strategy for managing stress.


# In[140]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import matplotlib.font_manager as fm

# Set plot style
sns.set_theme(style="whitegrid")

# Create subplots for Sleep Hours, Happiness, and Social Interaction
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Fix the boxplots by adding hue parameter and legend=False
# 📊 1. Sleep Hours by Diet Type
sns.boxplot(x="diet_type", y="sleep_hours", hue="diet_type", data=df, ax=axes[0], palette="coolwarm", legend=False)
axes[0].set_title("Sleep Hours by Diet Type")
# Fix the rotation by getting ticks first
axes[0].set_xticks(axes[0].get_xticks())
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# 📊 2. Happiness Score by Diet Type
sns.boxplot(x="diet_type", y="happiness_score", hue="diet_type", data=df, ax=axes[1], palette="coolwarm", legend=False)
axes[1].set_title("Happiness Score by Diet Type")
# Fix the rotation by getting ticks first
axes[1].set_xticks(axes[1].get_xticks())
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# 📊 3. Social Interaction Score by Diet Type
sns.boxplot(x="diet_type", y="social_interaction_score", hue="diet_type", data=df, ax=axes[2], palette="coolwarm", legend=False)
axes[2].set_title("Social Interaction Score by Diet Type")
# Fix the rotation by getting ticks first
axes[2].set_xticks(axes[2].get_xticks())
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 🔬 Statistical Analysis: ANOVA Tests
anova_sleep = stats.f_oneway(*[df[df["diet_type"] == d]["sleep_hours"] for d in df["diet_type"].unique()])
anova_happiness = stats.f_oneway(*[df[df["diet_type"] == d]["happiness_score"] for d in df["diet_type"].unique()])
anova_social = stats.f_oneway(*[df[df["diet_type"] == d]["social_interaction_score"] for d in df["diet_type"].unique()])

# Print results (removing emoji that cause font warnings)
print(f"ANOVA - Sleep Hours & Diet Type: p-value = {anova_sleep.pvalue:.4f}")
print(f"ANOVA - Happiness Score & Diet Type: p-value = {anova_happiness.pvalue:.4f}")
print(f"ANOVA - Social Interaction & Diet Type: p-value = {anova_social.pvalue:.4f}")


# In[ ]:


### ANOVA: Impact of Diet Type  

This analysis examines whether diet type influences:  
1. Sleep hours 
2. Happiness score 
3. Social interaction score  

#### **Results**  
- Sleep & Diet: p = 0.7849 → No significant effect  
- Happiness & Diet: p = 0.5998 → No significant effect  
- Social Interaction & Diet: p = 0.6833 → No significant effect  

#### Conclusion 
Diet type does not have a significant impact on these variables.  


# In[126]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Select relevant numeric columns for correlation analysis
correlation_columns = ["happiness_score", "stress_level", "sleep_hours", "screen_time_hours"]
correlation_matrix = df[correlation_columns].corr()

#  1. Correlation Heatmap (Key Takeaways in One Chart)
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Key Factors Affecting Happiness & Stress")
plt.show()

# 2. Scatterplots for Key Relationships
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Happiness vs Sleep (Is more sleep linked to higher happiness?)
sns.scatterplot(x=df["sleep_hours"], y=df["happiness_score"], alpha=0.5, ax=axes[0])
axes[0].set_title("Happiness vs Sleep Hours")
axes[0].set_xlabel("Sleep Hours")
axes[0].set_ylabel("Happiness Score")

# Happiness vs Screen Time (Does high screen time reduce happiness?)
sns.scatterplot(x=df["screen_time_hours"], y=df["happiness_score"], alpha=0.5, ax=axes[1])
axes[1].set_title("Happiness vs Screen Time")
axes[1].set_xlabel("Screen Time (Hours)")
axes[1].set_ylabel("Happiness Score")

plt.tight_layout()
plt.show()



# In[ ]:


### Correlation Analysis: Key Factors Affecting Happiness & Stress

#### 1. Correlation Heatmap
The heatmap presents correlations between happiness score, stress level, sleep hours, and screen time:
- Happiness & Stress Level: Strong negative correlation (-0.82)  
- Happiness & Sleep Hours: Strong positive correlation (0.78)  
- Happiness & Screen Time: Moderate negative correlation (-0.74)  

#### 2. Scatter Plots: Key Relationships
- Happiness vs. Sleep Hours: More sleep is associated with higher happiness.
- Happiness vs. Screen Time: More screen time correlates with lower happiness.

These results suggest that stress and screen time negatively impact happiness, while sleep has a positive effect.


# In[138]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  1. Select features (Now includes 6 variables)
features = ["stress_level", "screen_time_hours", "sleep_hours", 
            "work_hours_per_week", "social_interaction_score", "age"]
target = "happiness_score"

X = df[features]
y = df[target]

#  2. Split data into training & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  3. Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  4. Make predictions
y_pred = model.predict(X_test)

#  5. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ R² Score: {r2:.2f}")

#  6. Feature Importance Analysis
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# 📊 Plot Feature Importance (Fixed version to avoid FutureWarning)
plt.figure(figsize=(8,5))
# Option 1: Use the new syntax with explicit x, y parameters and hue=y
sns.barplot(x=feature_importance.values, y=feature_importance.index, hue=feature_importance.index, palette="coolwarm", legend=False)

# Alternative Option 2: Remove the palette parameter completely if custom colors aren't necessary
# sns.barplot(x=feature_importance.values, y=feature_importance.index)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance - What Affects Happiness?")
plt.show()


# In[ ]:


### Feature Importance: What Affects Happiness?

This analysis uses a Random Forest Model to determine the most influential factors impacting happiness score.

#### Model Performance
- Mean Absolute Error (MAE): 1.04→ Low error, meaning the model predictions are relatively accurate.
- R² Score: 0.76 → The model explains 76% of the variance in happiness.

#### **Key Factors Influencing Happiness**
The feature importance plot highlights the most impactful variables:
1. Stress Level (Strongest negative impact)
2. Social Interaction Score (Moderate positive impact)
3. Age (Slight influence)
4. Screen Time Hours (Negative impact)
5. Sleep Hours (Small positive impact)
6. Work Hours Per Week (Minimal impact)

These findings reinforce tha higher stress and excessive screen time reduce happiness, while more social interaction and sleep contribute positively.


# In[ ]:


## 📌 Final Summary: Integrated Analysis of Mental and Physical Health: Links Between Lifestyle Habits and Biometric Measures 

This project explores the complex relationships between stress, sleep, physical activity, screen time, and mental well-being through biometric smartwatch data and self-reported health surveys. By analyzing these variables, we aim to identify patterns that contribute to stress reduction and improved happiness levels.

### 🔎 Key Findings
1️⃣ Activity Level and Stress  
   - Sedentary individuals show higher stress levels compared to active ones.  
   - However, excessive activity does not necessarily equate to lower stress.  

2️⃣ Sleep and Stress 
   - Higher stress levels are associated with reduced sleep duration.  
   - Individuals with stress levels above 7 tend to sleep less than 5 hours, reinforcing the link between chronic stress and sleep deprivation.

3️⃣ Heart Rate and Stress  
   - A significant correlation exists between high stress levels and increased heart rate, demonstrating the physiological burden of stress.

4️⃣ Screen Time and Mental Health 
   - More screen time correlates with higher stress and lower happiness.  
   - Individuals spending 6+ hours per day on screens report notably lower happiness scores.

5️⃣ Feature Importance: What Drives Happiness?  
   - Stress is the most significant negative predictor of happiness.  
   - Sleep duration positively influences happiness, but to a smaller degree.  
   - Social interaction and age also impact overall well-being.  
   - Excessive screen time negatively affects happiness, reinforcing concerns about digital habits and well-being.

### 🎯 **Conclusions & Next Steps
- This study confirms that stress, sleep, activity levels, and screen time significantly impact mental and physical well-being.  
- The results suggest that reducing stress, improving sleep quality, and moderating screen time are essential for better mental and physical health.  
- Future research could focus on longitudinal trends (e.g., seasonal variations in stress) or integrate additional lifestyle factors for a more comprehensive perspective.  

This project contributes to a data-driven understanding of well-being, providing actionable insights for individuals seeking to optimize their daily habits for better health outcomes.

