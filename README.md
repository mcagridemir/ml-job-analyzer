# 🤖 ML Job Analyzer

A machine learning pipeline that analyzes LinkedIn job postings and predicts experience levels based on salary, work type, and job title keywords.

## 🎯 Project Overview

This project demonstrates an end-to-end ML pipeline built on real LinkedIn job posting data (123,000+ postings). Given a job title, salary, and work type, the model predicts whether a position is Entry level, Mid-Senior, or Director level.

## 🛠️ Tech Stack

- **Data Processing:** pandas, scikit-learn
- **Model:** Random Forest Classifier (72% accuracy)
- **Dashboard:** Streamlit
- **Orchestration:** Apache Airflow (coming soon)
- **Environment:** Docker

## 📊 Dataset

[LinkedIn Job Postings - Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

- 123,849 job postings
- 31 features
- Filtered to 27,954 records with complete salary and experience data

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ml-job-analyzer.git
cd ml-job-analyzer
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python notebooks/model.py
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

## 📈 Model Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Entry level | 0.73 | 0.69 | 0.71 |
| Mid-Senior level | 0.74 | 0.80 | 0.77 |
| Director | 0.36 | 0.23 | 0.28 |
| **Overall Accuracy** | | | **0.72** |

## 🔑 Key Features

- Salary range
- Work type (Full-time, Part-time, Contract)
- Remote work availability
- Job title keyword extraction (Engineer, Manager, Data/ML/AI)

## 🗺️ Roadmap

- [ ] Airflow DAG for automated data pipeline
- [ ] Docker containerization
- [ ] More keyword features for better accuracy
- [ ] Add more experience levels