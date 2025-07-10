import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model and feature columns
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

st.title("👨‍💼 Employee Attrition Predictor")

st.sidebar.header("Enter Employee Details")

# ---------------------------------
# Collect input from user
# ---------------------------------
age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
education = st.sidebar.selectbox("Education", [1, 2, 3, 4, 5])
work_life_balance = st.sidebar.selectbox("Work-Life Balance (1 = Bad, 4 = Excellent)", [1, 2, 3, 4])
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 5)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 3)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 10, 1)
performance_rating = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
job_role = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
    "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education_field = st.sidebar.selectbox("Education Field", [
    "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
])
business_travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# ---------------------------------
# Create base input dictionary
# ---------------------------------
input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "Education": education,
    "WorkLifeBalance": work_life_balance,
    "JobLevel": job_level,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "NumCompaniesWorked": num_companies_worked,
    "PerformanceRating": performance_rating
}

# ---------------------------------
# One-hot encode categorical fields
# ---------------------------------
# Helper function to add one-hot
def encode_one_hot(input_dict, feature_name, selected_value, possible_values):
    for val in possible_values:
        key = f"{feature_name}_{val}"
        input_dict[key] = 1 if selected_value == val else 0

encode_one_hot(input_dict, "OverTime", overtime, ["Yes"])
encode_one_hot(input_dict, "Gender", gender, ["Male"])
encode_one_hot(input_dict, "MaritalStatus", marital_status, ["Single", "Married"])
encode_one_hot(input_dict, "JobRole", job_role, [
    "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
    "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
])
encode_one_hot(input_dict, "Department", department, ["Sales", "Research & Development"])
encode_one_hot(input_dict, "EducationField", education_field, [
    "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"
])
encode_one_hot(input_dict, "BusinessTravel", business_travel, ["Travel_Rarely", "Travel_Frequently"])

# ---------------------------------
# Convert to DataFrame
# ---------------------------------
input_df = pd.DataFrame([input_dict])

# Fill missing features with 0 (to match training features)
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_cols]

# ---------------------------------
# Predict
# ---------------------------------

if st.button("🔮 Predict Attrition"):
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction:")
    st.success("🔴 Employee is likely to leave!" if prediction == 1 else "🟢 Employee will stay.")
