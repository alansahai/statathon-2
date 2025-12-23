import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 respondents
n = 1000

data = {
    'Respondent_ID': range(1, n + 1),
    'Age': np.random.randint(18, 80, n),
    'Income': np.random.normal(50000, 15000, n),  # Normal distribution
    'Household_Size': np.random.randint(1, 6, n),
    'Survey_Weight': np.random.uniform(0.8, 1.5, n) # Design weights
}

df = pd.DataFrame(data)

# --- INJECT DEFECTS FOR DEMO ---

# 1. Inject Missing Values (Demo Imputation)
# Make 50 random 'Income' values NaN
df.loc[np.random.choice(df.index, 50), 'Income'] = np.nan

# 2. Inject Outliers (Demo Outlier Removal)
# Add 5 crazy high incomes
outlier_indices = np.random.choice(df.index, 5)
df.loc[outlier_indices, 'Income'] = df.loc[outlier_indices, 'Income'] * 10 

# 3. Inject Rule Violations (Demo Rule Validation - NEW)
# Set 10 people to have negative Age (impossible)
violation_indices = np.random.choice(df.index, 10)
df.loc[violation_indices, 'Age'] = -5

# 4. Rename a column (Demo Schema Mapping)
# Rename 'Age' to 'Q1_Age_Years' to show mapping capability
df.rename(columns={'Age': 'Q1_Age_Years'}, inplace=True)

# Save
df.to_csv('demo_survey_data.csv', index=False)
print("✅ 'demo_survey_data.csv' generated successfully!")
print("   - [Cleaning] Includes Missing Values")
print("   - [Cleaning] Includes Outliers")
print("   - [Rules]    Includes Negative Ages (Rule Violation)")
print("   - [Schema]   Includes Non-standard column 'Q1_Age_Years'")