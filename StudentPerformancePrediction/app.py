# ==========================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# Developed by Team Techno Developers
# ==========================================================

# ==========================================================
# PART 1 : IMPORT LIBRARIES
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Student_Performance.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "student_performance_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Main Background */

.stApp{
    background:linear-gradient(135deg,#0F172A,#1E293B,#334155);
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

/* Headings */

h1,h2,h3,h4,h5,h6{
    color:white;
    font-family:Arial,Helvetica,sans-serif;
}

/* Paragraph */

p,label,span{
    color:white !important;
}

/* Glass Card */

.glass{

    background:rgba(255,255,255,0.08);

    border:1px solid rgba(255,255,255,0.20);

    border-radius:15px;

    padding:20px;

    margin-bottom:20px;

    backdrop-filter:blur(10px);

}

/* Metric Cards */

div[data-testid="metric-container"]{

    background:#1E3A8A;

    padding:15px;

    border-radius:12px;

    color:white;

    box-shadow:0px 4px 12px rgba(0,0,0,0.30);

}

/* Buttons */

.stButton>button{

    width:100%;

    background:#2563EB;

    color:white;

    border:none;

    border-radius:10px;

    height:50px;

    font-size:18px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#1D4ED8;

    color:white;

}

/* Download Button */

.stDownloadButton>button{

    width:100%;

    background:#16A34A;

    color:white;

    border:none;

    border-radius:10px;

    height:48px;

    font-size:17px;

    font-weight:bold;

}

.stDownloadButton>button:hover{

    background:#15803D;

}

/* DataFrame */

[data-testid="stDataFrame"]{

    border-radius:10px;

}

/* Footer */

.footer{

    text-align:center;

    color:white;

    padding:20px;

    font-size:16px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# PART 2 : LOAD DATASET, MODEL & SCALER
# ==========================================================

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():
    """
    Load Student Performance Dataset
    """

    df = pd.read_csv(DATASET_PATH)

    return df


# ==========================================================
# LOAD MODEL & SCALER
# ==========================================================

@st.cache_resource
def load_model_and_scaler():
    """
    Load Random Forest Model and StandardScaler
    """

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ==========================================================
# LOAD ALL FILES
# ==========================================================

try:

    df = load_dataset()

    model, scaler = load_model_and_scaler()

except FileNotFoundError as e:

    st.error("❌ Required file not found.")
    st.error(str(e))
    st.stop()

except Exception as e:

    st.error("❌ Error while loading project files.")
    st.exception(e)
    st.stop()


# ==========================================================
# VERIFY FILES
# ==========================================================

st.success("✅ Dataset Loaded Successfully")

st.success("✅ Random Forest Model Loaded Successfully")

st.success("✅ StandardScaler Loaded Successfully")


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

with st.expander("📂 Project Information", expanded=False):

    st.write("### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("Rows :", df.shape[0])

        st.write("Columns :", df.shape[1])

    with col2:

        st.write("Model :", type(model).__name__)

        st.write("Scaler :", type(scaler).__name__)

    st.write("")

    st.write("### Dataset Columns")

    st.dataframe(
        pd.DataFrame(
            {
                "Columns": df.columns
            }
        ),
        use_container_width=True
    )

    st.write("")

    st.write("### Missing Values")

    missing_df = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isnull().sum().values

    })

    st.dataframe(

        missing_df,

        use_container_width=True

    )

    st.write("")

    st.write("### Dataset Preview")

    st.dataframe(

        df.head(),

        use_container_width=True

    )

    # ==========================================================
# PART 3 : SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    # ======================================================
    # PROJECT TITLE
    # ======================================================

    st.markdown("""
    <h1 style='text-align:center;color:#60A5FA;'>
    🎓 Student Performance
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style='text-align:center;color:white;'>
    Prediction System
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # NAVIGATION MENU
    # ======================================================

    menu = st.radio(

        "📌 Navigation",

        (

            "🏠 Home",

            "📊 Dashboard",

            "🎯 Single Prediction",

            "📂 Bulk Prediction",

            "ℹ️ About Project"

        )

    )

    st.markdown("---")

    # ======================================================
    # PROJECT INFORMATION
    # ======================================================

    st.subheader("📁 Project Information")

    st.info("""

Student Performance Prediction System

Machine Learning Model :
Random Forest Regressor

Framework :
Streamlit

Language :
Python

Dataset :
Student_Performance.csv

""")

    st.markdown("---")

    # ======================================================
    # MODEL STATUS
    # ======================================================

    st.subheader("🤖 Model Status")

    st.success("✅ Random Forest Model Loaded")

    st.success("✅ Dataset Loaded")

    st.success("✅ Scaler Loaded")

    st.markdown("---")

    # ======================================================
    # DATASET DETAILS
    # ======================================================

    st.subheader("📊 Dataset")

    st.write(f"Rows : **{df.shape[0]}**")

    st.write(f"Columns : **{df.shape[1]}**")

    st.markdown("---")

    # ======================================================
    # TEAM DETAILS
    # ======================================================

    st.subheader("👨‍💻 Developed By")

    st.success("""

Team Techno Developers

• Harshit Kumar

• Shobhit Mishra

• Vedant Mishra

""")

    st.markdown("---")

    # ======================================================
    # VERSION
    # ======================================================

    st.caption("Version 2.0")

    st.caption("Random Forest Edition")

    st.markdown("---")

    # ======================================================
    # FOOTER
    # ======================================================

    st.markdown(
        """
        <div style="text-align:center;
                    color:white;
                    font-size:14px;">

        © 2026 Student Performance Prediction System

        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================
# PART 4 : HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    # ======================================================
    # HEADER
    # ======================================================

    st.markdown("""
    <div class="glass">

    <h1 style="text-align:center;color:#60A5FA;">
    🎓 Student Performance Prediction System
    </h1>

    <h4 style="text-align:center;color:white;">
    Machine Learning Based Student Performance Prediction
    </h4>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ======================================================
    # PROJECT OVERVIEW
    # ======================================================

    st.markdown("""
    <div class="glass">

    <h2 style="color:#38BDF8;">
    📖 Project Overview
    </h2>

    <p style="font-size:18px;">

    The Student Performance Prediction System is a Machine Learning
    application that predicts a student's overall academic performance
    using demographic information, attendance, study habits and subject
    marks.

    This project uses a Random Forest Regression model to estimate
    the student's final performance accurately.

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ======================================================
    # KEY FEATURES
    # ======================================================

    st.header("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="glass">

        ✅ Student Performance Prediction

        <br><br>

        ✅ Interactive Dashboard

        <br><br>

        ✅ Subject-wise Analysis

        <br><br>

        ✅ Performance Visualization

        <br><br>

        ✅ Prediction Report Download

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="glass">

        ✅ Bulk CSV Prediction

        <br><br>

        ✅ Random Forest Machine Learning Model

        <br><br>

        ✅ Professional UI

        <br><br>

        ✅ Real-Time Prediction

        <br><br>

        ✅ Easy Navigation

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ======================================================
    # PROJECT WORKFLOW
    # ======================================================

    st.header("🔄 Project Workflow")

    st.markdown("""
    <div class="glass">

    1️⃣ Load Student Dataset

    <br><br>

    ⬇

    <br><br>

    2️⃣ Data Preprocessing

    <br><br>

    ⬇

    <br><br>

    3️⃣ Feature Engineering

    <br><br>

    ⬇

    <br><br>

    4️⃣ Random Forest Prediction

    <br><br>

    ⬇

    <br><br>

    5️⃣ Performance Analysis

    <br><br>

    ⬇

    <br><br>

    6️⃣ Download Prediction Report

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ======================================================
    # TECHNOLOGIES
    # ======================================================

    st.header("🛠 Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="glass">

        <h3>🐍 Python</h3>

        ✔ Pandas

        <br>

        ✔ NumPy

        <br>

        ✔ Matplotlib

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="glass">

        <h3>🤖 Machine Learning</h3>

        ✔ Random Forest

        <br>

        ✔ Scikit-Learn

        <br>

        ✔ Joblib

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="glass">

        <h3>🌐 Frontend</h3>

        ✔ Streamlit

        <br>

        ✔ HTML

        <br>

        ✔ CSS

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    st.header("🤖 Model Information")

    info1, info2 = st.columns(2)

    with info1:

        st.info(f"""
Algorithm

• Random Forest Regressor

Dataset Records

• {df.shape[0]}

Dataset Columns

• {df.shape[1]}
""")

    with info2:

        st.info("""
Prediction Type

• Regression

Target

• Overall Student Performance

Framework

• Streamlit
""")

    st.write("")

    # ======================================================
    # PROJECT STATISTICS
    # ======================================================

    st.header("📊 Project Statistics")

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Students", len(df))

    with b:
        st.metric("Columns", df.shape[1])

    with c:
        st.metric("Model", "Random Forest")

    with d:
        st.metric("Status", "Ready")

    st.write("")

    # ======================================================
    # TEAM MEMBERS
    # ======================================================

    st.header("👨‍💻 Developed By")

    st.success("""
Team Techno Developers

• Harshit Kumar

• Shobhit Mishra

• Vedant Mishra
""")

    st.write("")

    # ======================================================
    # FOOTER
    # ======================================================

    st.markdown("""
    <div class="footer">

    © 2026 Student Performance Prediction System

    Developed using Python, Streamlit & Random Forest

    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
# PART 5 : DASHBOARD
# ==========================================================

if menu == "📊 Dashboard":

    st.title("📊 Student Performance Dashboard")

    st.markdown("---")

    # ======================================================
    # KPI CARDS
    # ======================================================

    total_students = len(df)

    average_score = round(df["overall_score"].mean(), 2)

    average_attendance = round(df["attendance_percentage"].mean(), 2)

    average_study_hours = round(df["study_hours"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👨‍🎓 Total Students",
            total_students
        )

    with col2:
        st.metric(
            "📈 Average Score",
            average_score
        )

    with col3:
        st.metric(
            "📅 Avg Attendance",
            f"{average_attendance}%"
        )

    with col4:
        st.metric(
            "📚 Avg Study Hours",
            average_study_hours
        )

    st.markdown("---")

    # ======================================================
    # DATASET PREVIEW
    # ======================================================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    left, right = st.columns(2)

    with left:

        st.subheader("Dataset Shape")

        shape_df = pd.DataFrame({

            "Information":[
                "Rows",
                "Columns"
            ],

            "Value":[
                df.shape[0],
                df.shape[1]
            ]

        })

        st.table(shape_df)

    with right:

        st.subheader("Missing Values")

        missing = df.isnull().sum().reset_index()

        missing.columns = [
            "Column",
            "Missing Values"
        ]

        st.dataframe(
            missing,
            use_container_width=True
        )

    st.markdown("---")

    # ======================================================
    # SUBJECT AVERAGE BAR CHART
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📚 Average Subject Scores")

        avg_marks = [

            df["math_score"].mean(),

            df["science_score"].mean(),

            df["english_score"].mean()

        ]

        fig, ax = plt.subplots(figsize=(5,4))

        ax.bar(

            ["Math","Science","English"],

            avg_marks

        )

        ax.set_ylim(0,100)

        ax.set_ylabel("Average Marks")

        st.pyplot(fig)

    # ======================================================
    # GENDER DISTRIBUTION
    # ======================================================

    with col2:

        st.subheader("👨👩 Gender Distribution")

        gender_counts = df["gender"].value_counts()

        fig, ax = plt.subplots(figsize=(5,4))

        ax.pie(

            gender_counts,

            labels=gender_counts.index,

            autopct="%1.1f%%",

            startangle=90

        )

        st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # SCORE DISTRIBUTION
    # ======================================================

    left, right = st.columns(2)

    with left:

        st.subheader("📈 Overall Score Distribution")

        fig, ax = plt.subplots(figsize=(5,4))

        ax.hist(

            df["overall_score"],

            bins=20

        )

        ax.set_xlabel("Overall Score")

        ax.set_ylabel("Students")

        st.pyplot(fig)

    # ======================================================
    # SCHOOL TYPE
    # ======================================================

    with right:

        st.subheader("🏫 School Type Distribution")

        school = df["school_type"].value_counts()

        fig, ax = plt.subplots(figsize=(5,4))

        ax.bar(

            school.index,

            school.values

        )

        ax.set_ylabel("Students")

        st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # CORRELATION HEATMAP
    # ======================================================

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))

    img = ax.imshow(

        corr,

        cmap="Blues"

    )

    ax.set_xticks(range(len(corr.columns)))

    ax.set_xticklabels(

        corr.columns,

        rotation=90

    )

    ax.set_yticks(range(len(corr.columns)))

    ax.set_yticklabels(corr.columns)

    plt.colorbar(img)

    st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # DATASET STATISTICS
    # ======================================================

    st.subheader("📊 Dataset Statistics")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

    st.markdown("---")

    # ======================================================
    # TOP STUDENTS
    # ======================================================

    st.subheader("🏆 Top 10 Students")

    top_students = df.sort_values(

        "overall_score",

        ascending=False

    ).head(10)

    st.dataframe(

        top_students,

        use_container_width=True

    )

    st.markdown("---")

    # ======================================================
    # BOTTOM STUDENTS
    # ======================================================

    st.subheader("📉 Bottom 10 Students")

    bottom_students = df.sort_values(

        "overall_score"

    ).head(10)

    st.dataframe(

        bottom_students,

        use_container_width=True

    )

    st.markdown("---")

    st.success("✅ Dashboard Loaded Successfully")

# ==========================================================
# PART 6.1 : SINGLE PREDICTION PAGE
# ==========================================================

if menu == "🎯 Single Prediction":

    st.title("🎯 Student Performance Prediction")

    st.markdown(
        "Fill all student details and click **Predict Performance**."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # ======================================================
        # LEFT COLUMN
        # ======================================================

        with col1:

            student_id = st.number_input(
                "Student ID",
                min_value=1,
                value=1
            )

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=25,
                value=18
            )

            gender = st.selectbox(
                "Gender",
                [
                    "female",
                    "male",
                    "other"
                ]
            )

            school_type = st.selectbox(
                "School Type",
                [
                    "private",
                    "public"
                ]
            )

            parent_education = st.selectbox(
                "Parent Education",
                [
                    "graduate",
                    "high school",
                    "no formal",
                    "phd",
                    "post graduate"
                ]
            )

            study_hours = st.number_input(
                "Study Hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

            attendance_percentage = st.slider(
                "Attendance Percentage",
                0.0,
                100.0,
                80.0
            )

        # ======================================================
        # RIGHT COLUMN
        # ======================================================

        with col2:

            internet_access = st.selectbox(
                "Internet Access",
                [
                    "no",
                    "yes"
                ]
            )

            travel_time = st.selectbox(
                "Travel Time",
                [
                    "30-60 min",
                    "<15 min",
                    ">60 min",
                    "15-30 min"
                ]
            )

            extra_activities = st.selectbox(
                "Extra Activities",
                [
                    "no",
                    "yes"
                ]
            )

            study_method = st.selectbox(
                "Study Method",
                [
                    "group study",
                    "mixed",
                    "notes",
                    "online videos",
                    "textbook"
                ]
            )

            math_score = st.number_input(
                "Math Score",
                0.0,
                100.0,
                70.0
            )

            science_score = st.number_input(
                "Science Score",
                0.0,
                100.0,
                70.0
            )

            english_score = st.number_input(
                "English Score",
                0.0,
                100.0,
                70.0
            )

        predict_btn = st.form_submit_button(
            "🎯 Predict Performance",
            use_container_width=True
        )
# ==========================================================
# PART 6.2 : CREATE INPUT DATAFRAME
# ==========================================================

    if predict_btn:

        # ---------------------------------------------
        # Calculate Overall Score
        # ---------------------------------------------

        overall_score = round(
            (math_score + science_score + english_score) / 3,
            2
        )

        # ---------------------------------------------
        # Temporary Final Grade
        # (Required because model was trained with it)
        # ---------------------------------------------

        final_grade = "b"

        # ---------------------------------------------
        # Create Input DataFrame
        # ---------------------------------------------

        input_df = pd.DataFrame({

            "student_id": [student_id],

            "age": [age],

            "gender": [gender],

            "school_type": [school_type],

            "parent_education": [parent_education],

            "study_hours": [study_hours],

            "attendance_percentage": [attendance_percentage],

            "internet_access": [internet_access],

            "travel_time": [travel_time],

            "extra_activities": [extra_activities],

            "study_method": [study_method],

            "math_score": [math_score],

            "science_score": [science_score],

            "english_score": [english_score],

            "overall_score": [overall_score],

            "final_grade": [final_grade]

        })
# ==========================================================
# PART 6.3 : PREPROCESS DATA & PREDICT
# ==========================================================

        # --------------------------------------------------
        # Convert Categorical Variables
        # --------------------------------------------------

        input_df = pd.get_dummies(input_df)

        # --------------------------------------------------
        # Match Training Features
        # --------------------------------------------------

        input_df = input_df.reindex(
            columns=scaler.feature_names_in_,
            fill_value=0
        )

        # --------------------------------------------------
        # Scale Input Data
        # --------------------------------------------------

        scaled_input = scaler.transform(input_df)

        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        prediction = model.predict(scaled_input)

        predicted_score = round(float(prediction[0]), 2)

        # --------------------------------------------------
        # Limit Score Between 0 and 100
        # --------------------------------------------------

        predicted_score = max(0, min(100, predicted_score))

        # --------------------------------------------------
        # Assign Grade
        # --------------------------------------------------

        if predicted_score >= 90:
            grade = "A+"
            remark = "Excellent"

        elif predicted_score >= 80:
            grade = "A"
            remark = "Very Good"

        elif predicted_score >= 70:
            grade = "B"
            remark = "Good"

        elif predicted_score >= 60:
            grade = "C"
            remark = "Average"

        elif predicted_score >= 50:
            grade = "D"
            remark = "Needs Improvement"

        else:
            grade = "F"
            remark = "Poor"     

# ==========================================================
# PART 6.4 : DISPLAY PREDICTION RESULT
# ==========================================================

        st.markdown("---")

        st.success("✅ Prediction Completed Successfully!")

        # --------------------------------------------------
        # RESULT METRICS
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Score",
                f"{predicted_score:.2f}%"
            )

        with col2:
            st.metric(
                "Grade",
                grade
            )

        with col3:
            st.metric(
                "Performance",
                remark
            )

        # --------------------------------------------------
        # PROGRESS BAR
        # --------------------------------------------------

        st.subheader("Overall Performance")

        st.progress(predicted_score / 100)

        st.write(f"### Predicted Score : {predicted_score:.2f}%")

        # --------------------------------------------------
        # PERFORMANCE MESSAGE
        # --------------------------------------------------

        if predicted_score >= 90:

            st.success("🏆 Outstanding Performance")

        elif predicted_score >= 80:

            st.success("🎉 Very Good Performance")

        elif predicted_score >= 70:

            st.info("👍 Good Performance")

        elif predicted_score >= 60:

            st.warning("🙂 Average Performance")

        elif predicted_score >= 50:

            st.warning("⚠ Needs Improvement")

        else:

            st.error("❌ Poor Performance")

        st.markdown("---")

        # --------------------------------------------------
        # PREDICTION SUMMARY
        # --------------------------------------------------

        summary_df = pd.DataFrame({

            "Parameter":[

                "Student ID",

                "Overall Score",

                "Predicted Score",

                "Grade",

                "Performance"

            ],

            "Value":[

                student_id,

                overall_score,

                predicted_score,

                grade,

                remark

            ]

        })

        st.subheader("Prediction Summary")

        st.table(summary_df)

        st.markdown("---")

        # --------------------------------------------------
        # VIEW MODEL INPUT
        # --------------------------------------------------

        with st.expander("📄 View Model Input"):

            st.dataframe(
                input_df,
                use_container_width=True
            )    

# ==========================================================
# PART 6.5 : CHARTS & DOWNLOAD REPORT
# ==========================================================

        st.header("📊 Prediction Analysis")

        col1, col2 = st.columns(2)

        # --------------------------------------------------
        # SUBJECT SCORE BAR CHART
        # --------------------------------------------------

        with col1:

            st.subheader("Subject-wise Scores")

            fig, ax = plt.subplots(figsize=(5,4))

            subjects = [
                "Math",
                "Science",
                "English"
            ]

            marks = [
                math_score,
                science_score,
                english_score
            ]

            ax.bar(subjects, marks)

            ax.set_ylim(0,100)

            ax.set_ylabel("Marks")

            st.pyplot(fig)

        # --------------------------------------------------
        # PREDICTION PIE CHART
        # --------------------------------------------------

        with col2:

            st.subheader("Predicted Performance")

            remaining = max(0,100-predicted_score)

            fig, ax = plt.subplots(figsize=(5,4))

            ax.pie(
                [predicted_score, remaining],
                labels=["Achieved","Remaining"],
                autopct="%1.1f%%",
                startangle=90
            )

            st.pyplot(fig)

        st.markdown("---")

        # --------------------------------------------------
        # REPORT TABLE
        # --------------------------------------------------

        report = pd.DataFrame({

            "Student ID":[student_id],

            "Age":[age],

            "Gender":[gender],

            "School Type":[school_type],

            "Study Hours":[study_hours],

            "Attendance (%)":[attendance_percentage],

            "Math Score":[math_score],

            "Science Score":[science_score],

            "English Score":[english_score],

            "Overall Score":[overall_score],

            "Predicted Score":[predicted_score],

            "Grade":[grade],

            "Performance":[remark]

        })

        st.subheader("Prediction Report")

        st.dataframe(
            report,
            use_container_width=True
        )

        st.markdown("---")

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        csv = report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            label="📥 Download Prediction Report",

            data=csv,

            file_name="Student_Prediction_Report.csv",

            mime="text/csv",

            use_container_width=True

        )

        st.success("✅ Prediction Report Generated Successfully.")

# ==========================================================
# PART 7.1 : BULK PREDICTION PAGE
# ==========================================================

if menu == "📂 Bulk Prediction":

    st.title("📂 Bulk Student Performance Prediction")

    st.markdown("""
    Upload a CSV file containing student details.
    The system will predict the performance of all students.
    """)

    st.markdown("---")

    uploaded_file = st.file_uploader(

        "📁 Upload Student CSV File",

        type=["csv"]

    )

    # ======================================================
    # CHECK FILE UPLOADED
    # ======================================================

    if uploaded_file is not None:

        bulk_df = pd.read_csv(uploaded_file)

        st.success("✅ CSV File Uploaded Successfully")

        st.markdown("---")

        # ==================================================
        # DATASET PREVIEW
        # ==================================================

        st.subheader("📋 Dataset Preview")

        st.dataframe(

            bulk_df.head(),

            use_container_width=True

        )

        st.markdown("---")

        # ==================================================
        # DATASET INFORMATION
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Total Rows",

                bulk_df.shape[0]

            )

        with col2:

            st.metric(

                "Total Columns",

                bulk_df.shape[1]

            )

        with col3:

            st.metric(

                "Missing Values",

                int(bulk_df.isnull().sum().sum())

            )

        st.markdown("---")

        # ==================================================
        # SHOW COLUMN NAMES
        # ==================================================

        st.subheader("📄 Uploaded Columns")

        st.write(list(bulk_df.columns))

        st.markdown("---")

        # ==================================================
        # START BULK PREDICTION BUTTON
        # ==================================================

        predict_all_btn = st.button(

            "🚀 Predict All Students",

            use_container_width=True

        )  

# ==========================================================
# PART 7.2 : DATA PREPROCESSING
# ==========================================================

        bulk_df = bulk_df.copy()

        # --------------------------------------------------
        # REQUIRED COLUMNS
        # --------------------------------------------------

        required_columns = [

            "student_id",

            "age",

            "gender",

            "school_type",

            "parent_education",

            "study_hours",

            "attendance_percentage",

            "internet_access",

            "travel_time",

            "extra_activities",

            "study_method",

            "math_score",

            "science_score",

            "english_score"

        ]

        # --------------------------------------------------
        # CHECK MISSING COLUMNS
        # --------------------------------------------------

        missing_columns = [

            col for col in required_columns

            if col not in bulk_df.columns

        ]

        if len(missing_columns) > 0:

            st.error("❌ Required columns are missing.")

            st.write(missing_columns)

            st.stop()

        # --------------------------------------------------
        # CALCULATE OVERALL SCORE
        # --------------------------------------------------

        bulk_df["overall_score"] = (

            bulk_df["math_score"]

            + bulk_df["science_score"]

            + bulk_df["english_score"]

        ) / 3

        # --------------------------------------------------
        # TEMPORARY FINAL GRADE
        # --------------------------------------------------

        bulk_df["final_grade"] = "b"

        # --------------------------------------------------
        # HANDLE MISSING VALUES
        # --------------------------------------------------

        bulk_df.fillna({

            "gender": "female",

            "school_type": "private",

            "parent_education": "graduate",

            "internet_access": "yes",

            "travel_time": "15-30 min",

            "extra_activities": "no",

            "study_method": "notes",

            "study_hours": 0,

            "attendance_percentage": 0,

            "math_score": 0,

            "science_score": 0,

            "english_score": 0,

            "overall_score": 0

        }, inplace=True)

        st.success("✅ Dataset preprocessing completed successfully.") 

# ==========================================================
# PART 7.3 : PREPROCESS, SCALE & PREDICT
# ==========================================================

if predict_all_btn:

    # Keep Original Data
    original_df = bulk_df.copy()

    # ---------------------------------------------
    # Required Columns

    required_columns = [

        "student_id",
        "age",
        "gender",
        "school_type",
        "parent_education",
        "study_hours",
        "attendance_percentage",
        "internet_access",
        "travel_time",
        "extra_activities",
        "study_method",
        "math_score",
        "science_score",
        "english_score"

    ]

    missing_columns = [

        col for col in required_columns

        if col not in original_df.columns

    ]

    if len(missing_columns) > 0:

        st.error("Missing Columns")

        st.write(missing_columns)

        st.stop()

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    original_df["overall_score"] = (

        original_df["math_score"] +

        original_df["science_score"] +

        original_df["english_score"]

    ) / 3

    original_df["final_grade"] = "b"

    # ---------------------------------------------
    # One Hot Encoding
    # ---------------------------------------------

    encoded_df = pd.get_dummies(original_df)

    # ---------------------------------------------
    # Match Model Features
    # ---------------------------------------------

    encoded_df = encoded_df.reindex(

        columns=scaler.feature_names_in_,

        fill_value=0

    )

    # ---------------------------------------------
    # Scale
    # ---------------------------------------------

    scaled_data = scaler.transform(encoded_df)

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    predictions = model.predict(scaled_data)

    predictions = np.clip(predictions,0,100)

    predictions = predictions.round(2)

    st.success("Prediction Completed Successfully")


# ==========================================================
# PART 7.4 : RESULT TABLE
# ==========================================================

    result_df = original_df.copy()

    result_df["Predicted Score"] = predictions

    grades = []

    remarks = []

    for score in predictions:

        if score >= 90:

            grades.append("A+")

            remarks.append("Excellent")

        elif score >= 80:

            grades.append("A")

            remarks.append("Very Good")

        elif score >= 70:

            grades.append("B")

            remarks.append("Good")

        elif score >= 60:

            grades.append("C")

            remarks.append("Average")

        elif score >= 50:

            grades.append("D")

            remarks.append("Needs Improvement")

        else:

            grades.append("F")

            remarks.append("Poor")

    result_df["Grade"] = grades

    result_df["Performance"] = remarks

    st.markdown("---")

    st.subheader("Prediction Results")

    st.dataframe(

        result_df,

        use_container_width=True

    )

    st.markdown("---")

    col1,col2,col3=st.columns(3)

    with col1:

        st.metric(

            "Total Students",

            len(result_df)

        )

    with col2:

        st.metric(

            "Average Score",

            round(result_df["Predicted Score"].mean(),2)

        )

    with col3:

        st.metric(

            "Highest Score",

            round(result_df["Predicted Score"].max(),2)

        )

# ==========================================================
# PART 7.5 : VISUALIZATION & DOWNLOAD REPORT
# ==========================================================

    st.markdown("---")

    # ======================================================
    # GRADE DISTRIBUTION
    # ======================================================

    st.subheader("📊 Grade Distribution")

    grade_count = result_df["Grade"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(

        grade_count.index,

        grade_count.values

    )

    ax.set_xlabel("Grade")

    ax.set_ylabel("Number of Students")

    ax.set_title("Predicted Grade Distribution")

    st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # PREDICTED SCORE DISTRIBUTION
    # ======================================================

    st.subheader("📈 Predicted Score Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(

        result_df["Predicted Score"],

        bins=10

    )

    ax.set_xlabel("Predicted Score")

    ax.set_ylabel("Students")

    st.pyplot(fig)

    st.markdown("---")

    # ======================================================
    # TOP 10 STUDENTS
    # ======================================================

    st.subheader("🏆 Top 10 Students")

    top_students = result_df.sort_values(

        by="Predicted Score",

        ascending=False

    ).head(10)

    st.dataframe(

        top_students,

        use_container_width=True

    )

    st.markdown("---")

    # ======================================================
    # DOWNLOAD CSV
    # ======================================================

    csv = result_df.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        label="📥 Download Prediction Results",

        data=csv,

        file_name="Bulk_Prediction_Result.csv",

        mime="text/csv",

        use_container_width=True

    )

    st.success("✅ Bulk Prediction Completed Successfully!")

    st.balloons()

if predict_all_btn:

# ==========================================================
# PART 10 : ABOUT PROJECT
# ==========================================================

 if menu == "ℹ️ About Project":

    st.title("ℹ️ About Student Performance Prediction System")

    st.markdown("---")

    # ======================================================
    # PROJECT OVERVIEW
    # ======================================================

    st.markdown("""
    <div class='glass'>
        <h2 style='color:#60A5FA;'>🎓 Project Overview</h2>

        <p style='font-size:18px;'>

        The Student Performance Prediction System is a Machine Learning
        web application developed using Python and Streamlit.

        It predicts the academic performance of students using
        demographic details, attendance, study habits,
        and subject-wise marks.

        The prediction model is built using XGBoost Regressor,
        which provides accurate and fast predictions.

        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # FEATURES
    # ======================================================

    st.subheader("✨ Project Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
✔ Student Performance Prediction

✔ Interactive Dashboard

✔ Data Visualization

✔ Single Student Prediction

✔ Bulk Prediction
""")

    with col2:

        st.success("""
✔ CSV Report Download

✔ XGBoost Machine Learning Model

✔ User Friendly Interface

✔ Fast Prediction

✔ Professional Dashboard
""")

    st.markdown("---")

    # ======================================================
    # TECHNOLOGIES
    # ======================================================

    st.subheader("🛠 Technologies Used")

    tech_df = pd.DataFrame({

        "Technology":[
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Scikit-Learn",
            "Joblib",
            "XGBoost"
        ],

        "Purpose":[
            "Programming",
            "Web Application",
            "Data Processing",
            "Numerical Computing",
            "Visualization",
            "Preprocessing",
            "Model Loading",
            "Machine Learning"
        ]

    })

    st.dataframe(
        tech_df,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    st.subheader("🤖 Model Information")

    left, right = st.columns(2)

    with left:

        st.info(f"""

Model Name

student_performance_model.pkl

Algorithm

XGBoost Regressor

Prediction Type

Regression
""")

    with right:

        st.info(f"""

Dataset

Student_Performance.csv

Scaler

scaler.pkl

Framework

Streamlit
""")

    st.markdown("---")

    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    st.subheader("📊 Dataset Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Total Students",
            len(df)
        )

    with c2:

        st.metric(
            "Total Features",
            df.shape[1]
        )

    with c3:

        st.metric(
            "Average Score",
            round(df["overall_score"].mean(),2)
        )

    st.markdown("---")

    # ======================================================
    # TEAM MEMBERS
    # ======================================================

    st.subheader("👨‍💻 Developed By")

    team = pd.DataFrame({

        "Name":[
            "Harshit Kumar",
            "Shobhit Mishra",
            "Vedant Mishra"
        ],

        "Role":[
            "Machine Learning Developer",
            "Frontend Developer",
            "Backend Developer"
        ]

    })

    st.table(team)

    st.markdown("---")

    # ======================================================
    # FUTURE SCOPE
    # ======================================================

    st.subheader("🚀 Future Scope")

    st.markdown("""

- AI-based Study Recommendation

- Student Progress Tracking

- Teacher Dashboard

- Parent Dashboard

- Mobile Application

- Cloud Database Integration

- Email Report Generation

- Deep Learning Models

""")

    st.markdown("---")

    # ======================================================
    # VERSION
    # ======================================================

    st.subheader("📌 Project Details")

    details = pd.DataFrame({

        "Field":[
            "Project Name",
            "Version",
            "Language",
            "Framework",
            "Machine Learning",
            "Deployment"
        ],

        "Value":[
            "Student Performance Prediction System",
            "Version 1.0",
            "Python",
            "Streamlit",
            "XGBoost Regressor",
            "Streamlit Cloud"
        ]

    })

    st.table(details)

    st.markdown("---")

    # ======================================================
    # FOOTER
    # ======================================================

    st.markdown("""
    <div class='glass'>

    <h3 style='text-align:center;color:#60A5FA;'>

    🎓 Student Performance Prediction System

    </h3>

    <p style='text-align:center;'>

    Developed with ❤️ using

    <br><br>

    Python | Streamlit | XGBoost | Machine Learning

    <br><br>

    © 2026 Team Techno Developers

    </p>

    </div>
    """, unsafe_allow_html=True)