import pandas as pd

df = pd.read_csv('/Users/cagri.demir/ml-job-analyzer/data/postings.csv')

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Target column distribution
print("\nExperience levels:")
print(df['formatted_experience_level'].value_counts())

print("\nSalary statistics:")
print(df['normalized_salary'].describe())

# Select useful columns and drop missing values
useful_cols = ['title', 'formatted_experience_level', 'formatted_work_type',
               'location', 'normalized_salary', 'remote_allowed']
df_clean = df[useful_cols].dropna(subset=['formatted_experience_level', 'normalized_salary'])
print(f"\nCleaned dataset shape: {df_clean.shape}")
print(df_clean['formatted_experience_level'].value_counts())