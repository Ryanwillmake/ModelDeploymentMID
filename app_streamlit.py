import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cls_model = joblib.load("model_klasifikasi.pkl")
reg_model = joblib.load("model_regresi.pkl")

st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("🎓 Student Placement & Salary Predictor")
st.markdown("Prediksi status penempatan kerja dan estimasi gaji mahasiswa berdasarkan profil akademik dan skill.")

with st.sidebar:
    st.header("Input Data Mahasiswa")
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc = st.slider("SSC Percentage", 50, 95, 70)
    hsc = st.slider("HSC Percentage", 50, 94, 70)
    degree = st.slider("Degree Percentage", 55, 89, 70)
    cgpa = st.slider("CGPA", 5.5, 9.8, 7.5)
    entrance = st.slider("Entrance Exam Score", 40, 100, 70)
    technical = st.slider("Technical Skill Score", 40, 100, 70)
    soft = st.slider("Soft Skill Score", 40, 100, 70)
    internship = st.slider("Internship Count", 0, 5, 1)
    projects = st.slider("Live Projects", 0, 5, 1)
    workexp = st.slider("Work Experience (Months)", 0, 24, 6)
    certs = st.slider("Certifications", 0, 5, 1)
    attendance = st.slider("Attendance Percentage", 50, 100, 75)
    backlogs = st.slider("Backlogs", 0, 5, 0)
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    predict_btn = st.button("Prediksi", type="primary")

gender_enc = 1 if gender == "Male" else 0
extra_enc = 1 if extra == "Yes" else 0
score_avg = (technical + soft) / 2
academic_avg = (ssc + hsc + degree) / 3

input_data = pd.DataFrame([[
    gender_enc, ssc, hsc, degree, cgpa, entrance,
    technical, soft, internship, projects, workexp,
    certs, attendance, backlogs, extra_enc, score_avg, academic_avg
]], columns=[
    'gender', 'ssc_percentage', 'hsc_percentage', 'degree_percentage',
    'cgpa', 'entrance_exam_score', 'technical_skill_score', 'soft_skill_score',
    'internship_count', 'live_projects', 'work_experience_months',
    'certifications', 'attendance_percentage', 'backlogs',
    'extracurricular_activities', 'score_avg', 'academic_avg'
])

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Profil Mahasiswa")
    profile_data = {
        'Fitur': ['CGPA', 'Technical Skill', 'Soft Skill', 'Backlogs', 'Attendance'],
        'Nilai': [cgpa, technical, soft, backlogs, attendance]
    }
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(profile_data['Fitur'], profile_data['Nilai'], color='steelblue')
    ax.set_xlabel('Nilai')
    ax.set_title('Profil Skill & Akademik')
    st.pyplot(fig)

with col2:
    st.subheader("🔮 Hasil Prediksi")
    if predict_btn:
        placement_pred = cls_model.predict(input_data)[0]
        placement_prob = cls_model.predict_proba(input_data)[0]
        salary_pred = reg_model.predict(input_data)[0]

        if placement_pred == 1:
            st.success("✅ Mahasiswa diprediksi **DITEMPATKAN**")
            st.metric("Probabilitas Placed", f"{placement_prob[1]*100:.1f}%")
            st.metric("Estimasi Gaji", f"{salary_pred:.2f} LPA")
        else:
            st.error("❌ Mahasiswa diprediksi **TIDAK DITEMPATKAN**")
            st.metric("Probabilitas Tidak Placed", f"{placement_prob[0]*100:.1f}%")
            st.metric("Estimasi Gaji", "0 LPA")
    else:
        st.info("Atur input di sidebar lalu klik **Prediksi**")

st.divider()
st.subheader("📈 Distribusi Data Training")
df = pd.read_csv("B.csv")
fig2, axes = plt.subplots(1, 3, figsize=(14, 3))
df['placement_status'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue','salmon'])
axes[0].set_title('Placement Status')
df['cgpa'].hist(ax=axes[1], bins=20, color='steelblue')
axes[1].set_title('Distribusi CGPA')
df['salary_package_lpa'].hist(ax=axes[2], bins=20, color='salmon')
axes[2].set_title('Distribusi Salary')
plt.tight_layout()
st.pyplot(fig2)