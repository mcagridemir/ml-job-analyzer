import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('/Users/cagri.demir/ml-job-analyzer/models/job_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🤖 Job Level Predictor")
st.subheader("LinkedIn Job Posting Analyzer")
st.write("Enter job details to predict the experience level.")

# Input fields
salary = st.slider("Annual Salary (USD)", 30000, 300000, 80000, step=5000)

work_type = st.selectbox("Work Type", ["Full-time", "Part-time", "Contract", "Internship"])
work_type_map = {"Full-time": 1, "Part-time": 3, "Contract": 0, "Internship": 2}
work_type_encoded = work_type_map[work_type]

remote = st.checkbox("Remote Allowed")

title = st.text_input("Job Title", placeholder="e.g. Senior ML Engineer")

# Extract keyword features
is_engineer = 1 if any(word in title.lower() for word in ['engineer', 'developer']) else 0
is_manager = 1 if any(word in title.lower() for word in ['manager', 'lead', 'head']) else 0
is_data = 1 if any(word in title.lower() for word in ['data', 'ml', 'ai', 'machine learning']) else 0

# Predict
if st.button("Predict Level"):
    features = np.array([[salary, work_type_encoded, int(remote),
                          is_engineer, is_manager, is_data]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max() * 100

    color = {"Entry level": "🟢", "Mid-Senior level": "🟡", "Director": "🔴"}
    st.markdown(f"## {color.get(prediction, '⚪')} {prediction}")
    st.write(f"Confidence: **{confidence:.1f}%**")

    st.divider()
    st.write("**Detected keywords:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Engineer/Dev", "✅" if is_engineer else "❌")
    col2.metric("Manager/Lead", "✅" if is_manager else "❌")
    col3.metric("Data/ML/AI", "✅" if is_data else "❌")