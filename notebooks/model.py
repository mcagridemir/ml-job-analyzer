import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('/Users/cagri.demir/ml-job-analyzer/data/postings.csv')

# Select relevant columns
useful_cols = ['title', 'formatted_experience_level', 'formatted_work_type',
               'location', 'normalized_salary', 'remote_allowed']
df_clean = df[useful_cols].dropna(subset=['formatted_experience_level', 'normalized_salary'])

# Filter to 3 experience levels
df_clean = df_clean[df_clean['formatted_experience_level'].isin(['Entry level', 'Mid-Senior level', 'Director'])]

# Encode categorical columns
le = LabelEncoder()
df_clean['work_type_encoded'] = le.fit_transform(df_clean['formatted_work_type'])
df_clean['remote_encoded'] = df_clean['remote_allowed'].fillna(0).astype(int)

# Extract keyword features from job title
df_clean['is_engineer'] = df_clean['title'].str.contains('Engineer|Developer', case=False, na=False).astype(int)
df_clean['is_manager'] = df_clean['title'].str.contains('Manager|Lead|Head', case=False, na=False).astype(int)
df_clean['is_data'] = df_clean['title'].str.contains('Data|ML|AI|Machine Learning', case=False, na=False).astype(int)

# Define features and target
X = df_clean[['normalized_salary', 'work_type_encoded', 'remote_encoded',
              'is_engineer', 'is_manager', 'is_data']]
y = df_clean['formatted_experience_level']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Performance:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature Importances:")
feature_names = ['normalized_salary', 'work_type', 'remote', 'is_engineer', 'is_manager', 'is_data']
for name, importance in sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.3f}")

# Save model
os.makedirs('/Users/cagri.demir/ml-job-analyzer/models', exist_ok=True)
with open('/Users/cagri.demir/ml-job-analyzer/models/job_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")