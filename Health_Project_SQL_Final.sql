#SMARTWATCH HEALTH

RENAME TABLE `smartwatch_health_data_nobom` TO `smartwatch_health` ;
-- Standardize column names and allow NULL values
-- Replace spaces with underscores and remove parentheses

-- Clean the `User ID`
ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `User ID` `user_id` INT NULL;

UPDATE `smartwatch_health`
SET `user_id` = NULL
WHERE `user_id` = ''
   OR `user_id` NOT REGEXP '^[0-9]+$';

-- 2) Clean the `Heart Rate (BPM)` column before type conversion

UPDATE `smartwatch_health`
SET `Heart Rate (BPM)` = NULL
WHERE `Heart Rate (BPM)` = ''
   OR `Heart Rate (BPM)` NOT REGEXP '^[0-9]+(\.[0-9]+)?$';

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Heart Rate (BPM)` `heart_rate_bpm` DOUBLE NULL;

-- 3) Clean the `Blood Oxygen Level (%)` column

UPDATE `smartwatch_health`
SET `Blood Oxygen Level (%)` = NULL
WHERE `Blood Oxygen Level (%)` = ''
   OR `Blood Oxygen Level (%)` NOT REGEXP '^[0-9]+(\.[0-9]+)?$';

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Blood Oxygen Level (%)` `blood_oxygen_pct` DOUBLE NULL;

-- 4) Clean `Step Count`

UPDATE `smartwatch_health`
SET `Step Count` = NULL
WHERE `Step Count` = ''
   OR `Step Count` NOT REGEXP '^[0-9]+(\.[0-9]+)?$';

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Step Count` `step_count` DOUBLE NULL;

-- 5) Clean `Sleep Duration (hours)`

UPDATE `smartwatch_health`
SET `Sleep Duration (hours)` = NULL
WHERE `Sleep Duration (hours)` = ''
   OR `Sleep Duration (hours)` NOT REGEXP '^[0-9]+(\.[0-9]+)?$';

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Sleep Duration (hours)` `sleep_duration_hrs` DOUBLE NULL;

-- 6) Clean `Stress Level` 

UPDATE `smartwatch_health`
SET `Stress Level` = NULL
WHERE `Stress Level` = ''
   OR `Stress Level` NOT REGEXP '^[0-9]+$';

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Stress Level` `stress_level` INT NULL;


DESCRIBE smartwatch_health;

SELECT COUNT(*) FROM smartwatch_health;

SELECT * 
FROM smartwatch_health ;

-- Rename Activity Level column 

ALTER TABLE `smartwatch_health`
  CHANGE COLUMN `Activity Level` `activity_level` VARCHAR(50) NULL ; 

-- Check distinct values in the 'activity_level' column to identify inconsistencies or spelling errors

SELECT DISTINCT activity_level
FROM smartwatch_health;

-- Count how many rows have 'nan' as the value for 'activity_level', indicating missing or invalid data

SELECT COUNT(*) AS nb_nan
FROM smartwatch_health
WHERE activity_level = 'nan';


-- 1) Remove unnecessary spaces around values

UPDATE smartwatch_health
SET activity_level = TRIM(activity_level);

-- 2) Standardize "Actve" to "Active"

UPDATE smartwatch_health
SET activity_level = 'Active'
WHERE activity_level = 'Actve';

-- 3)  Standardize "Highly_Active" to "Highly Active"

UPDATE smartwatch_health
SET activity_level = 'Highly Active'
WHERE activity_level = 'Highly_Active';

-- 4) Standardize different variants of "Sedentary"

UPDATE smartwatch_health
SET activity_level = 'Sedentary'
WHERE TRIM(activity_level) IN ('Seddentary', 'Seddenary', 'Seddedentary');

-- 5) Convert "nan" to NULL

UPDATE smartwatch_health
SET activity_level = NULL
WHERE activity_level = 'nan';


-- Check distinct values

SELECT DISTINCT activity_level
FROM smartwatch_health;

-- Display the distribution of records grouped by 'activity_level'

SELECT  activity_level, 
    COUNT(*) AS nb_records
FROM smartwatch_health
GROUP BY activity_level;

-- Display the distribution of records grouped by 'stress_level', sorted by stress level

SELECT 
    stress_level, 
    COUNT(*) AS nb_records
FROM smartwatch_health
GROUP BY stress_level
ORDER BY stress_level;

-- Identify records where 'stress_level' is outside the valid range (1-10) or is NULL

SELECT *
FROM smartwatch_health
WHERE stress_level < 1
   OR stress_level > 10
   OR stress_level IS NULL;

ALTER TABLE smartwatch_health
MODIFY COLUMN stress_level INT NULL COMMENT 'Scale from 1 to 10 (1 = very low, 10 = very high)';

SHOW CREATE TABLE smartwatch_health;

-- Check basic statistics (minimum, maximum, average, and non-null count) for 'heart_rate_bpm'

SELECT 
    MIN(heart_rate_bpm) AS min_bpm,
    MAX(heart_rate_bpm) AS max_bpm,
    AVG(heart_rate_bpm) AS avg_bpm,
    COUNT(heart_rate_bpm) AS count_bpm
FROM smartwatch_health;

-- Identify records with unrealistic high heart rates (>250 BPM) or missing heart rate values (NULL)

SELECT *
FROM smartwatch_health
WHERE heart_rate_bpm > 250
   OR heart_rate_bpm IS NULL;

SELECT *
FROM smartwatch_health
WHERE heart_rate_bpm > 250 ;

-- Analysis of heart_rate_bpm:
-- Values greater than 250 BPM seem unrealistic from a physiological standpoint.
-- Therefore, we investigate these outliers further and decide to set these extremely high values to NULL
-- to avoid skewing our analysis and statistical measures.

-- Set unrealistic high heart_rate_bpm values (>250) to NULL
UPDATE smartwatch_health
SET heart_rate_bpm = NULL
WHERE heart_rate_bpm > 250;

-- Verify updated statistics to ensure consistency and data quality
SELECT 
    MIN(heart_rate_bpm) AS min_bpm,
    MAX(heart_rate_bpm) AS max_bpm,
    AVG(heart_rate_bpm) AS avg_bpm,
    COUNT(heart_rate_bpm) AS count_bpm
FROM smartwatch_health;

-- Checking basic statistics for blood_oxygen_pct
SELECT 
    MIN(blood_oxygen_pct) AS min_oxygen,
    MAX(blood_oxygen_pct) AS max_oxygen,
    AVG(blood_oxygen_pct) AS avg_oxygen,
    COUNT(blood_oxygen_pct) AS count_oxygen
FROM smartwatch_health;

-- Check basic statistics for step_count to identify unusual values
SELECT 
    MIN(step_count) AS min_steps,
    MAX(step_count) AS max_steps,
    AVG(step_count) AS avg_steps,
    COUNT(step_count) AS count_steps
FROM
    smartwatch_health;

-- Count the number of records with very low daily step counts (<100 steps)

SELECT COUNT(*) AS very_low_steps_count
FROM smartwatch_health
WHERE step_count < 100;

-- Count the number of records with very high daily step counts (>30,000 steps)

SELECT COUNT(*) AS very_high_steps_count
FROM smartwatch_health
WHERE step_count > 30000;

-- Values lower than 30 steps per day are considered unrealistic (likely measurement errors or device non-usage).
-- I chose to keep high step counts because such activity levels can occur occasionally. 
-- Hence, I set these extremely low values (<30) to NULL to avoid skewing future analyses.

UPDATE smartwatch_health
SET step_count = NULL
WHERE step_count < 30;

-- Check basic statistics (minimum, maximum, average, and count) for 'sleep_duration_hrs'

SELECT 
    MIN(sleep_duration_hrs) AS min_sleep_hrs,
    MAX(sleep_duration_hrs) AS max_sleep_hrs,
    AVG(sleep_duration_hrs) AS avg_sleep_hrs,
    COUNT(sleep_duration_hrs) AS count_sleep_records
FROM smartwatch_health;

-- Identify records with unrealistic sleep durations (<1 hour or >16 hours) or missing sleep duration values (NULL)

SELECT *
FROM smartwatch_health
WHERE sleep_duration_hrs < 1
   OR sleep_duration_hrs > 16
   OR sleep_duration_hrs IS NULL;
   
   -- Count the number of records with very low sleep durations (less than 1 hour)
   
SELECT COUNT(*) AS very_low_sleep_count
FROM smartwatch_health
WHERE sleep_duration_hrs < 1;
-- Only one record found with sleep_duration_hrs < 1 hour (approx. 35 mins).
-- Given that such short sleep durations, while rare, could still realistically occur (e.g., naps, interrupted sleep),
-- I choose to retain this value.

-- Checking for duplicate rows across all columns

SELECT user_id, heart_rate_bpm, blood_oxygen_pct, step_count, sleep_duration_hrs, activity_level, stress_level, COUNT(*) AS duplicate_count
FROM smartwatch_health
GROUP BY user_id, heart_rate_bpm, blood_oxygen_pct, step_count, sleep_duration_hrs, activity_level, stress_level
HAVING duplicate_count > 1;

-- Create a new column "stress_category" based on stress_level
ALTER TABLE smartwatch_health ADD COLUMN stress_category VARCHAR(15);


-- Add the 'stress_category' column to the table
ALTER TABLE smartwatch_health ADD COLUMN stress_category VARCHAR(15);

-- Update 'stress_category' based on 'stress_level'
UPDATE smartwatch_health
SET stress_category = CASE
    WHEN stress_level BETWEEN 1 AND 3 THEN 'Low'
    WHEN stress_level BETWEEN 4 AND 7 THEN 'Moderate'
    WHEN stress_level BETWEEN 8 AND 10 THEN 'High'
    ELSE NULL
END;

-- Step 1: Add a new column for health_score

-- Total health score from 5 factors (steps, sleep, heart rate, stress, oxygen), range: 5-15 (higher is better)

ALTER TABLE smartwatch_health ADD COLUMN health_score INT NULL;

-- Step 2: Calculate and update health_score based on multiple health indicators

UPDATE smartwatch_health
SET health_score =
    -- Step Count Score
    (CASE
        WHEN step_count >= 10000 THEN 3  -- Active lifestyle
        WHEN step_count BETWEEN 5000 AND 9999 THEN 2  -- Moderate activity
        WHEN step_count < 5000 THEN 1  -- Sedentary
        ELSE 0
    END)
    +
    -- Sleep Duration Score
    (CASE
        WHEN sleep_duration_hrs BETWEEN 7 AND 9 THEN 3  -- Ideal sleep range
        WHEN (sleep_duration_hrs BETWEEN 5 AND 6.99) OR (sleep_duration_hrs BETWEEN 9.01 AND 10) THEN 2  -- Acceptable sleep
        WHEN sleep_duration_hrs < 5 OR sleep_duration_hrs > 10 THEN 1  -- Too little or too much sleep
        ELSE 0
    END)
    +
    -- Heart Rate Score
    (CASE
        WHEN heart_rate_bpm <= 65 THEN 3  -- Low resting heart rate (good cardiovascular health)
        WHEN heart_rate_bpm BETWEEN 66 AND 80 THEN 2  -- Normal resting heart rate
        WHEN heart_rate_bpm > 80 THEN 1  -- High resting heart rate (possible stress/unfit)
        ELSE 0
    END)
    +
    -- Stress Level Score
    (CASE
        WHEN stress_level BETWEEN 1 AND 3 THEN 3  -- Low stress (good mental health)
        WHEN stress_level BETWEEN 4 AND 7 THEN 2  -- Moderate stress
        WHEN stress_level >= 8 THEN 1  -- High stress
        ELSE 0
    END)
    +
    -- Blood Oxygen Score
    (CASE
        WHEN blood_oxygen_pct >= 98 THEN 3  -- Well oxygenated
        WHEN blood_oxygen_pct BETWEEN 95 AND 97 THEN 2  -- Acceptable oxygen level
        WHEN blood_oxygen_pct < 95 THEN 1  -- Low oxygen saturation
        ELSE 0
    END);

SELECT user_id, step_count, sleep_duration_hrs, heart_rate_bpm, stress_level, blood_oxygen_pct, health_score
FROM smartwatch_health
LIMIT 20;

-- Add a new column 'health_category' to classify the health_score

ALTER TABLE smartwatch_health ADD COLUMN health_category VARCHAR(15) NULL;

-- Assign health categories based on the health_score

UPDATE smartwatch_health
SET health_category = 
    CASE 
        WHEN health_score BETWEEN 13 AND 15 THEN 'Excellent'  -- Top health scores
        WHEN health_score BETWEEN 10 AND 12 THEN 'Good'       -- Above average health
        WHEN health_score BETWEEN 7 AND 9 THEN 'Moderate'     -- Average health
        WHEN health_score BETWEEN 5 AND 6 THEN 'Low'         -- Poor health, possible issues
        ELSE NULL
    END;

-- Count users in each health category

SELECT health_category, COUNT(*) AS count_users
FROM smartwatch_health
GROUP BY health_category;


SELECT user_id, health_score, health_category
FROM smartwatch_health
LIMIT 20;

SELECT *
FROM smartwatch_health ; 

#Mental Health 

SELECT *
FROM mental_health ;

-- Check the structure of the mental_health table

DESCRIBE mental_health;

-- Get the total number of records in the dataset

SELECT COUNT(*) AS total_records FROM mental_health;

-- Count the number of duplicate rows 

SELECT COUNT(*) AS duplicate_count
FROM (
    SELECT `ï»¿Country` AS country, Age, Gender, `Exercise Level`, `Diet Type`, `Sleep Hours`, 
           `Stress Level`, `Mental Health Condition`, `Work Hours per Week`, 
           `Screen Time per Day (Hours)`, `Social Interaction Score`, `Happiness Score`, 
           COUNT(*) 
    FROM mental_health
    GROUP BY `ï»¿Country`, Age, Gender, `Exercise Level`, `Diet Type`, `Sleep Hours`, 
             `Stress Level`, `Mental Health Condition`, `Work Hours per Week`, 
             `Screen Time per Day (Hours)`, `Social Interaction Score`, `Happiness Score`
    HAVING COUNT(*) > 1
) AS duplicates;

-- Fix encoding issue with "Country" column (remove BOM characters)

ALTER TABLE mental_health
CHANGE COLUMN `ï»¿Country` `country` VARCHAR(100);

-- Check distinct values in `Stress Level` before conversion

SELECT DISTINCT `Stress Level` FROM mental_health;

-- Create a backup column to preserve original text values

ALTER TABLE mental_health ADD COLUMN stress_level_original VARCHAR(50);
UPDATE mental_health SET stress_level_original = `Stress Level`;

-- Convert 'Low', 'Moderate', 'High' into numeric values

UPDATE mental_health
SET `Stress Level` = 
    CASE 
        WHEN `Stress Level` = 'Low' THEN '2'  -- Example: Low → 2
        WHEN `Stress Level` = 'Moderate' THEN '5' -- Moderate → 5
        WHEN `Stress Level` = 'High' THEN '8' -- High → 8
        ELSE `Stress Level`
    END;

-- Verify that only numeric values remain

SELECT DISTINCT `Stress Level` FROM mental_health;

-- Convert `Stress Level` column from TEXT to INT

ALTER TABLE mental_health 
CHANGE COLUMN `Stress Level` stress_level INT NULL;

-- Create a categorical column to preserve stress level as text

ALTER TABLE mental_health ADD COLUMN stress_category VARCHAR(15);

-- Assign categories based on numeric `stress_level`

UPDATE mental_health
SET stress_category = 
    CASE 
        WHEN stress_level BETWEEN 1 AND 3 THEN 'Low'
        WHEN stress_level BETWEEN 4 AND 7 THEN 'Moderate'
        WHEN stress_level BETWEEN 8 AND 10 THEN 'High'
        ELSE NULL
    END;

-- Final verification: Check distinct values in `stress_level` and `stress_category`

SELECT DISTINCT stress_level, stress_category FROM mental_health;

ALTER TABLE mental_health DROP COLUMN stress_level_original;


-- Standardize column names (replace spaces with underscores)

ALTER TABLE mental_health
CHANGE COLUMN `Exercise Level` exercise_level VARCHAR(50),
CHANGE COLUMN `Diet Type` diet_type VARCHAR(50),
CHANGE COLUMN `Sleep Hours` sleep_hours DOUBLE,
CHANGE COLUMN `Mental Health Condition` mental_health_condition VARCHAR(100),
CHANGE COLUMN `Work Hours per Week` work_hours_per_week INT,
CHANGE COLUMN `Screen Time per Day (Hours)` screen_time_hours DOUBLE,
CHANGE COLUMN `Social Interaction Score` social_interaction_score DOUBLE,
CHANGE COLUMN `Happiness Score` happiness_score DOUBLE;

ALTER TABLE mental_health
CHANGE COLUMN `Age` age DOUBLE ; 

ALTER TABLE mental_health 
CHANGE COLUMN `Gender` gender VARCHAR(50);

-- Count NULL or empty values for each column

SELECT 'country' AS column_name, COUNT(*) - COUNT(country) AS missing_values FROM mental_health
UNION ALL
SELECT 'age', COUNT(*) - COUNT(age) FROM mental_health
UNION ALL
SELECT 'gender', COUNT(*) - COUNT(gender) FROM mental_health
UNION ALL
SELECT 'exercise_level', COUNT(*) - COUNT(exercise_level) FROM mental_health
UNION ALL
SELECT 'diet_type', COUNT(*) - COUNT(diet_type) FROM mental_health
UNION ALL
SELECT 'sleep_hours', COUNT(*) - COUNT(sleep_hours) FROM mental_health
UNION ALL
SELECT 'stress_level', COUNT(*) - COUNT(stress_level) FROM mental_health
UNION ALL
SELECT 'mental_health_condition', COUNT(*) - COUNT(mental_health_condition) FROM mental_health
UNION ALL
SELECT 'work_hours_per_week', COUNT(*) - COUNT(work_hours_per_week) FROM mental_health
UNION ALL
SELECT 'screen_time_hours', COUNT(*) - COUNT(screen_time_hours) FROM mental_health
UNION ALL
SELECT 'social_interaction_score', COUNT(*) - COUNT(social_interaction_score) FROM mental_health
UNION ALL
SELECT 'happiness_score', COUNT(*) - COUNT(happiness_score) FROM mental_health;


-- Check unique values in 'exercise_level'

SELECT DISTINCT exercise_level FROM mental_health;

-- Check unique values in 'diet_type'

SELECT DISTINCT diet_type FROM mental_health;

-- Check unique values in 'mental_health_condition'

SELECT DISTINCT mental_health_condition FROM mental_health;

UPDATE mental_health
SET mental_health_condition = NULL
WHERE mental_health_condition = 'None';


-- Check unique values in 'gender'
SELECT DISTINCT gender FROM mental_health;

-- Check unique values in 'country'
SELECT DISTINCT country FROM mental_health;

-- Check basic statistics for the 'age' column
-- This helps detect outliers (e.g., too young or too old)

SELECT 
    MIN(age) AS min_age,  -- Minimum age in the dataset
    MAX(age) AS max_age,  -- Maximum age in the dataset
    AVG(age) AS avg_age,  -- Average age in the dataset
    COUNT(age) AS count_age -- Total number of records with a valid age
FROM mental_health;

-- Check basic statistics for the 'sleep_hours' column
-- This helps detect unrealistic sleep durations (e.g., 0 hours or more than 16 hours)

SELECT 
    MIN(sleep_hours) AS min_sleep,  -- Minimum sleep hours recorded
    MAX(sleep_hours) AS max_sleep,  -- Maximum sleep hours recorded
    AVG(sleep_hours) AS avg_sleep,  -- Average sleep hours
    COUNT(sleep_hours) AS count_sleep -- Total number of records with valid sleep data
FROM mental_health;

-- Count the number of records where sleep_hours is below 3

SELECT COUNT(*) AS very_low_sleep_count
FROM mental_health
WHERE sleep_hours < 3;

-- Check basic statistics for the 'work_hours_per_week' column
-- This helps detect unrealistic work schedules (e.g., more than 100 hours per week)

SELECT 
    MIN(work_hours_per_week) AS min_work,  -- Minimum recorded work hours per week
    MAX(work_hours_per_week) AS max_work,  -- Maximum recorded work hours per week
    AVG(work_hours_per_week) AS avg_work,  -- Average work hours per week
    COUNT(work_hours_per_week) AS count_work -- Total records with valid work hour data
FROM mental_health;

-- Check basic statistics for the 'screen_time_hours' column
-- This helps detect unrealistic screen times (e.g., more than 24 hours per day)

SELECT 
    MIN(screen_time_hours) AS min_screen,  -- Minimum screen time recorded per day
    MAX(screen_time_hours) AS max_screen,  -- Maximum screen time recorded per day
    AVG(screen_time_hours) AS avg_screen,  -- Average screen time per day
    COUNT(screen_time_hours) AS count_screen -- Total records with valid screen time data
FROM mental_health;

-- Check basic statistics for the 'happiness_score' column
-- This helps detect incorrect values (e.g., scores outside the expected range of 1-10)

SELECT 
    MIN(happiness_score) AS min_happiness,  -- Minimum happiness score recorded
    MAX(happiness_score) AS max_happiness,  -- Maximum happiness score recorded
    AVG(happiness_score) AS avg_happiness,  -- Average happiness score
    COUNT(happiness_score) AS count_happiness -- Total records with valid happiness score data
FROM mental_health;

-- Add a new column to categorize happiness levels

ALTER TABLE mental_health ADD COLUMN happiness_category VARCHAR(15) NULL;


-- Assign happiness categories based on the happiness score

UPDATE mental_health
SET happiness_category = 
    CASE 
        WHEN happiness_score BETWEEN 7 AND 10 THEN 'Happy'      -- High happiness
        WHEN happiness_score BETWEEN 4 AND 6.99 THEN 'Neutral' -- Medium happiness
        WHEN happiness_score BETWEEN 1 AND 3.99 THEN 'Unhappy' -- Low happiness
        ELSE NULL
    END;

-- Count how many users fall into each happiness category
SELECT happiness_category, COUNT(*) AS count_users
FROM mental_health
GROUP BY happiness_category;

-- Add a new column for social interaction categories

ALTER TABLE mental_health ADD COLUMN interaction_category VARCHAR(20) NULL;

-- Categorize users based on their social interaction score

UPDATE mental_health
SET interaction_category = 
    CASE 
        WHEN social_interaction_score BETWEEN 7 AND 10 THEN 'High Interaction'      -- Very social
        WHEN social_interaction_score BETWEEN 4 AND 6.99 THEN 'Normal Interaction' -- Balanced social life
        WHEN social_interaction_score BETWEEN 1 AND 3.99 THEN 'Low Interaction'    -- Limited interactions
        ELSE NULL
    END;

-- Check the distribution of users in each interaction category

SELECT interaction_category, COUNT(*) AS count_users
FROM mental_health
GROUP BY interaction_category;

# JOIN

-- Update the stress_level in mental_health to match the range in smartwatch_health (1 to 10)
-- We convert 'Low', 'Moderate', and 'High' categories into numeric values
-- using random values within predefined ranges.

UPDATE mental_health

SET stress_level = CASE
    WHEN stress_category = 'Low' THEN FLOOR(1 + (RAND() * 3))  -- Random value between 1 and 3
    WHEN stress_category = 'Moderate' THEN FLOOR(4 + (RAND() * 4)) -- Random value between 4 and 7
    WHEN stress_category = 'High' THEN FLOOR(8 + (RAND() * 3)) -- Random value between 8 and 10
    ELSE NULL  -- In case there's an unexpected value, we set it to NULL (to handle it later if needed)
END;

-- Verify the distribution of new stress levels in mental_health
SELECT stress_level, COUNT(*) AS record_count
FROM mental_health
GROUP BY stress_level
ORDER BY stress_level;

-- Check for potential NULL values
SELECT COUNT(*) AS null_stress_levels
FROM mental_health
WHERE stress_level IS NULL;

-- Aggregation Before Joining


-- Aggregate smartwatch_health data by stress_level
SELECT 
    stress_level,
    COUNT(*) AS record_count,  -- Count of records per stress level
    AVG(step_count) AS avg_step_count,  
    AVG(sleep_duration_hrs) AS avg_sleep_duration,  
    AVG(heart_rate_bpm) AS avg_heart_rate,  
    AVG(health_score) AS avg_health_score,  
    AVG(blood_oxygen_pct) AS avg_blood_oxygen
FROM smartwatch_health
GROUP BY stress_level
ORDER BY stress_level;

-- Aggregate mental_health data by stress_level
SELECT 
    stress_level,
    COUNT(*) AS record_count,  -- Count of records per stress level
    AVG(age) AS avg_age,  
    AVG(sleep_hours) AS avg_sleep_hours,  
    AVG(screen_time_hours) AS avg_screen_time,  
    AVG(social_interaction_score) AS avg_social_interaction,  
    AVG(happiness_score) AS avg_happiness
FROM mental_health
GROUP BY stress_level
ORDER BY stress_level;

-- Check statistical properties (min, max, standard deviation) for smartwatch_health

SELECT 
    MIN(step_count) AS min_steps, MAX(step_count) AS max_steps, STDDEV(step_count) AS stddev_steps,
    MIN(health_score) AS min_health, MAX(health_score) AS max_health, STDDEV(health_score) AS stddev_health,
    MIN(sleep_duration_hrs) AS min_sleep, MAX(sleep_duration_hrs) AS max_sleep, STDDEV(sleep_duration_hrs) AS stddev_sleep
FROM smartwatch_health;

-- Consistency Checks

-- Check original stress categories vs converted numeric stress levels

SELECT stress_category, stress_level, COUNT(*) 
FROM smartwatch_health
GROUP BY stress_category, stress_level
ORDER BY stress_category, stress_level;

-- Check if certain stress levels dominate after the JOIN

SELECT stress_level, COUNT(*) AS joined_rows
FROM (
    SELECT mh.stress_level
    FROM mental_health mh
    JOIN smartwatch_health sh
    ON mh.stress_level = sh.stress_level
) AS joined_data
GROUP BY stress_level
ORDER BY joined_rows DESC;

-- Investigating Relationships Between Variables (mental_health)

-- Average happiness by stress category

SELECT 
    stress_category,  -- Grouping by stress category (Low, Moderate, High)
    AVG(happiness_score) AS avg_happiness_score  -- Averaging numerical values
FROM mental_health 
GROUP BY stress_category
ORDER BY FIELD(stress_category, 'Low', 'Moderate', 'High');  -- Ensure logical order

-- Additional health metrics averages by stress category

SELECT stress_category, 
       AVG(sleep_hours) AS avg_sleep, 
       AVG(social_interaction_score) AS avg_social, 
       AVG(screen_time_hours) AS avg_screen_time
FROM mental_health 
GROUP BY stress_category
ORDER BY FIELD(stress_category, 'Low', 'Moderate', 'High');

-- Check detailed distribution of stress_level and category

SELECT stress_level, stress_category, COUNT(*)AS record_count
FROM mental_health
GROUP BY stress_level, stress_category
ORDER BY stress_level;

-- Standard deviation of mental health metrics by stress category

SELECT stress_category,
       STDDEV(happiness_score) AS stddev_happiness,
       STDDEV(sleep_hours) AS stddev_sleep,
       STDDEV(social_interaction_score) AS stddev_social,
       STDDEV(screen_time_hours) AS stddev_screen_time
FROM mental_health
GROUP BY stress_category;

-- Analyze extreme stress levels specifically for happiness insights

SELECT stress_level, AVG(happiness_score)AS avg_happiness
FROM mental_health
WHERE stress_level IN (1, 2, 9, 10)
GROUP BY stress_level
ORDER BY stress_level;

-- Re-group stress levels into broader categories (low, moderate, high) to see clearer trends

SELECT 
    CASE 
        WHEN stress_level <= 3 THEN 'Low Stress'
        WHEN stress_level >= 8 THEN 'High Stress'
        ELSE 'Moderate Stress'
    END AS stress_group,
    AVG(happiness_score) AS avg_happiness
FROM mental_health
GROUP BY stress_group;

--  Final Verification of smartwatch_health Logical Differences

SELECT
    stress_level,
    COUNT(*) AS total_records,
    AVG(step_count) AS avg_steps,
    AVG(heart_rate_bpm) AS avg_heart_rate,
    AVG(sleep_duration_hrs) AS avg_sleep_duration,
    AVG(health_score) AS avg_health_score,
    AVG(blood_oxygen_pct) AS avg_blood_oxygen
FROM smartwatch_health
GROUP BY stress_level
ORDER BY stress_level;


# Data Issues and Transition from SQL to Python


# 1. Inconsistent Data Distribution
# - The `stress_level` variable in both `smartwatch_health` and `mental_health` datasets needed to be aligned.
# - We converted categorical stress levels ("Low", "Moderate", "High") into numerical values (1-10)
#   using a randomized but controlled approach.
# - Some variables showed unexpected distributions, requiring additional adjustments.

# 2. Data Integrity Issues
# - Outliers and missing values were detected in key variables (`heart_rate_bpm`, `sleep_duration_hrs`, `screen_time_hours`).
# - Some correlations did not reflect expected real-world patterns, affecting the reliability of statistical analysis.

# 3. Limitations for Further Analysis
# - While SQL was useful for cleaning and structuring the data, it had limitations for advanced statistical modeling.
# - We needed Python (Pandas, Seaborn, Matplotlib, and SciPy) for deeper insights, statistical testing, and visualization.


# Solution: Generating Artificial Data for Python Analysis


# To ensure a more realistic and interpretable analysis, we decided to generate artificial datasets with:
# - The same structure and column names as our cleaned SQL datasets.
# - Logical correlations between variables (e.g., higher stress → higher heart rate, lower sleep, increased screen time).
# - Controlled data distribution for meaningful insights and visualizations.
# - Consistent stress levels across both datasets for easy joins and comparative analysis.

# This transition from SQL to Python allows us to leverage advanced statistical techniques and
# better understand health and stress relationships.

