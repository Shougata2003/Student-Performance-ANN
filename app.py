
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

APP_ENV = os.getenv("APP_ENV", "development").lower()

# Load local .env only during development
if APP_ENV != "production":
    load_dotenv(BASE_DIR / ".env", override=False)


def get_setting(key, default=""):
    """Read configuration from environment variables or Streamlit secrets."""

    value = os.getenv(key)

    if value:
        return value.strip()

    try:
        value = st.secrets.get(key, default)
        return str(value).strip()
    except Exception:
        return default


API_BASE_URL = get_setting(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

API_KEY = get_setting("API_KEY")

PREDICT_URL = f"{API_BASE_URL}/predict"
REQUEST_TIMEOUT = 15


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter the student's academic and personal details "
    "to predict their performance using an Artificial Neural Network."
)

st.divider()


# --------------------------------------------------
# 3. CHECK CONFIGURATION
# --------------------------------------------------

if not API_KEY:
    st.error(
        "API key is not configured. "
        "Please configure API_KEY in your local .env file "
        "or deployment secrets."
    )
    st.stop()


# --------------------------------------------------
# 4. STUDENT INPUT FORM
# --------------------------------------------------

with st.form("student_prediction_form"):

    st.subheader("Student Information")

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=24,
        value=20
    )

    attendance_pct = st.number_input(
        "Attendance Percentage",
        min_value=45.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    study_hours_per_week = st.number_input(
        "Study Hours Per Week",
        min_value=2.0,
        max_value=40.0,
        value=15.0,
        step=1.0
    )

    previous_score = st.number_input(
        "Previous Academic Score",
        min_value=25.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    assignment_completion_pct = st.number_input(
        "Assignment Completion Percentage",
        min_value=20.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    internal_assessment_score = st.number_input(
        "Internal Assessment Score",
        min_value=20.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    participation_score = st.number_input(
        "Participation Score",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=1.0
    )

    sleep_hours_per_day = st.number_input(
        "Sleep Hours Per Day",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

    extracurricular_hours_per_week = st.number_input(
        "Extracurricular Hours Per Week",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=1.0
    )

    failed_subjects = st.number_input(
        "Number of Failed Subjects",
        min_value=0,
        max_value=5,
        value=0
    )

    submitted = st.form_submit_button(
        "Predict Student Performance",
        use_container_width=True
    )


# --------------------------------------------------
# 5. SEND DATA TO FASTAPI
# --------------------------------------------------

if submitted:

    student_data = {
        "age": age,
        "attendance_pct": attendance_pct,
        "study_hours_per_week": study_hours_per_week,
        "previous_score": previous_score,
        "assignment_completion_pct": assignment_completion_pct,
        "internal_assessment_score": internal_assessment_score,
        "participation_score": participation_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "extracurricular_hours_per_week": extracurricular_hours_per_week,
        "failed_subjects": failed_subjects
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    with st.spinner("Analyzing student data..."):

        try:
            response = requests.post(
                PREDICT_URL,
                json=student_data,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            # --------------------------------------
            # 6. HANDLE API ERRORS
            # --------------------------------------

            if response.status_code == 200:

                result = response.json()

                predicted_performance = result.get(
                    "predicted_performance"
                )

                probabilities = result.get("probabilities")

                if predicted_performance is None:
                    st.error(
                        "The API response does not contain "
                        "the predicted performance."
                    )
                    st.stop()

                # ----------------------------------
                # 7. DISPLAY PREDICTION
                # ----------------------------------

                st.divider()
                st.subheader("Prediction Result")

                st.success(
                    f"Predicted Performance: {predicted_performance}"
                )

                # ----------------------------------
                # 8. DISPLAY PROBABILITIES
                # ----------------------------------

                if isinstance(probabilities, dict) and probabilities:

                    st.subheader("Prediction Probabilities")

                    for class_name, probability in probabilities.items():

                        try:
                            probability = float(probability)

                            st.write(
                                f"**{class_name}:** "
                                f"{probability * 100:.2f}%"
                            )

                            st.progress(
                                max(0.0, min(1.0, probability))
                            )

                        except (ValueError, TypeError):
                            st.warning(
                                f"Could not display probability "
                                f"for {class_name}."
                            )

            elif response.status_code == 401:

                st.error(
                    "Authentication failed. "
                    "Check whether the API key is correct."
                )

            elif response.status_code == 422:

                st.error(
                    "Invalid input. Please check the entered values."
                )

                st.json(response.json())

            elif response.status_code == 429:

                st.warning(
                    "Too many prediction requests. "
                    "Please wait before trying again."
                )

            elif response.status_code >= 500:

                st.error(
                    "The prediction server encountered an error. "
                    "Please try again later."
                )

            else:

                st.error(
                    f"API request failed with status code "
                    f"{response.status_code}."
                )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction server took too long to respond."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the backend is running."
            )

        except requests.exceptions.RequestException:

            st.error(
                "An unexpected error occurred while sending "
                "the prediction request."
            )

        except ValueError:

            st.error(
                "The server returned an invalid response."
            )
