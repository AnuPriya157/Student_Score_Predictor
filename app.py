import streamlit as st
import joblib
import pandas as pd

# Load model and columns
model = joblib.load("student_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("🎓 Student Score Predictor")

# Input fields
hours = st.number_input("Hours Studied", 0.0, 24.0)
attendance = st.number_input("Attendance", 0.0, 100.0)
previous = st.number_input("Previous Score", 0.0, 100.0)
sleep = st.number_input("Sleep Hours", 0.0, 12.0)

motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
teacher = st.selectbox("Teacher Quality", ["Poor", "Average", "Good"])
school = st.selectbox("School Type", ["Public", "Private"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
income = st.selectbox("Family Income", ["Low", "Medium", "High"])
parent = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
education = st.selectbox("Parent Education", ["School", "College"])
peer = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
resources = st.selectbox("Learning Resources", ["Low", "Medium", "High"])
activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# Button
if st.button("Predict Score"):

    data = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": previous,
        "Sleep_Hours": sleep,

        "Motivation_Level": motivation,
        "Teacher_Quality": teacher,
        "School_Type": school,
        "Internet_Access": internet,
        "Family_Income": income,
        "Parental_Involvement": parent,
        "Parental_Education_Level": education,
        "Peer_Influence": peer,
        "Learning_Resources": resources,
        "Extracurricular_Activities": activities
    }

    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"🎯 Predicted Exam Score: {round(prediction[0], 2)}")