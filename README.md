# Student Performance Prediction Using ANN

An AI-powered student performance prediction system built using an **Artificial Neural Network (ANN)**, **FastAPI**, and **Streamlit**. The application analyzes academic and lifestyle-related factors to predict a student's performance category and display the model's prediction probabilities.

---

## Overview

Understanding student performance can help identify students who may need additional academic support.

This project uses a trained Artificial Neural Network to analyze multiple student-related features and predict one of three performance categories:

- **High**
- **Medium**
- **Low**

The application provides an interactive Streamlit interface, while a separate FastAPI backend handles model inference, input validation, API-key authentication, and request rate limiting.

## Key Features

- **ANN-based prediction:** Uses a trained Artificial Neural Network to predict student performance.
- **10 input features:** Analyzes academic, attendance, participation, and lifestyle-related factors.
- **Probability visualization:** Displays the predicted probability for each performance category.
- **Interactive interface:** Provides a user-friendly Streamlit application.
- **REST API:** Uses FastAPI to serve predictions independently of the frontend.
- **Input validation:** Validates incoming data using Pydantic.
- **API-key authentication:** Protects the prediction endpoint using an API key.
- **Rate limiting:** Limits prediction requests to 30 requests per minute per client IP.
- **Preprocessing consistency:** Uses the previously trained scaler and label encoder during inference.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Data Preprocessing | Scikit-learn |
| Model Serialization | Joblib |
| Backend | FastAPI |
| API Server | Uvicorn |
| Rate Limiting | SlowAPI |
| Frontend | Streamlit |
| HTTP Communication | Requests |
| Environment Configuration | Python-dotenv |
| Exploratory Analysis | Jupyter Notebook, Matplotlib, Seaborn |

## System Architecture

The project follows a separated frontend-backend architecture.

```text
              USER
                |
                v
       STREAMLIT FRONTEND
          (app.py)
                |
                | HTTP POST /predict
                | X-API-Key Header
                v
        FASTAPI BACKEND
           (api.py)
                |
                v
       INPUT VALIDATION
           (Pydantic)
                |
                v
       FEATURE PREPROCESSING
          (Saved Scaler)
                |
                v
       TRAINED ANN MODEL
       (student_performance.keras)
                |
                v
       PERFORMANCE PREDICTION
                |
                v
        LABEL ENCODER
                |
                v
       PREDICTION PROBABILITIES
                |
                v
       STREAMLIT RESULTS UI
```

The frontend does not load the ANN model directly. It communicates with the backend through an HTTP API.

## Model Input Features

The ANN uses the following 10 features in the specified training order:

| No. | Feature | Description |
|---:|---|---|
| 1 | `age` | Student's age |
| 2 | `attendance_pct` | Attendance percentage |
| 3 | `study_hours_per_week` | Weekly study hours |
| 4 | `previous_score` | Previous academic score |
| 5 | `assignment_completion_pct` | Assignment completion percentage |
| 6 | `internal_assessment_score` | Internal assessment score |
| 7 | `participation_score` | Academic participation score |
| 8 | `sleep_hours_per_day` | Daily sleep duration |
| 9 | `extracurricular_hours_per_week` | Weekly extracurricular activity hours |
| 10 | `failed_subjects` | Number of failed subjects |

The input features are scaled using the saved preprocessing scaler before being passed to the ANN.

## Project Structure

```text
Student-Performance-ANN/
│
├── api.py
├── app.py
│
├── requirements.txt
├── requirements-api.txt
│
├── .env.example
├── .gitignore
│
├── models/
│   ├── student_performance.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── student_performance_eda.ipynb
│
└── test_rate_limit.py
```

**Note:** The original CSV dataset is not included in this repository. It is intentionally excluded to keep the dataset private.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shougata2003/Student-Performance-ANN.git

cd Student-Performance-ANN
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```bash
python -m pip install -r requirements-api.txt
```

### 4. Install frontend dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

Example:

```env
APP_ENV=development

API_BASE_URL=http://127.0.0.1:8000

API_KEY=replace_with_your_own_secret_key_of_at_least_32_characters
```

Generate a strong API key using:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the generated value into your local `.env` file.

**Security:** Never commit `.env` or expose your actual API key in source code, screenshots, or public repositories.

### 6. Start the FastAPI backend

Open a terminal in the project directory:

```bash
python -m uvicorn api:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

During development, interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Streamlit frontend

Open a second terminal:

```bash
python -m streamlit run app.py
```

The frontend will normally be available at:

```text
http://localhost:8501
```

Make sure the FastAPI backend is running before submitting a prediction.

## API Information

### Health Check

**Endpoint:**

```http
GET /health
```

Checks the backend health status.

### Student Performance Prediction

**Endpoint:**

```http
POST /predict
```

**Authentication header:**

```http
X-API-Key: YOUR_API_KEY
```

**Example request body:**

```json
{
  "age": 20,
  "attendance_pct": 85,
  "study_hours_per_week": 15,
  "previous_score": 70,
  "assignment_completion_pct": 80,
  "internal_assessment_score": 75,
  "participation_score": 7,
  "sleep_hours_per_day": 7,
  "extracurricular_hours_per_week": 3,
  "failed_subjects": 0
}
```

The API returns a predicted performance category and class probabilities.

## Security and Validation

The backend includes several security and reliability measures:

- API-key authentication for the prediction endpoint.
- Constant-time API-key comparison.
- Pydantic validation for incoming student data.
- Rejection of unexpected input fields.
- Bounded input ranges.
- Rate limiting of 30 requests per minute per client IP.
- Generic error responses for unexpected server failures.
- Environment-based configuration for secrets.

**Production consideration:** The current rate limiter uses in-memory storage. A production deployment with multiple workers or instances should use shared storage such as Redis. Production deployment should also use HTTPS and securely managed environment secrets.

## Exploratory Data Analysis

The project includes a Jupyter notebook:

`student_performance_eda.ipynb`

It contains the exploratory analysis and data preparation workflow used during development.

The original dataset is excluded from the repository. Users wishing to reproduce the notebook must obtain an authorized copy and place it in the project root using the expected filename.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Model predictions are estimates, not guarantees of actual academic outcomes.
- Correlations in the dataset should not be interpreted as proof of causation.
- The model may not generalize equally across different institutions or student populations.
- The dataset and model evaluation metrics need to be independently verified before real-world use.
- Predictions should support human decision-making, not replace teachers' or academic advisors' judgment.

## Future Improvements

- Add model evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Introduce explainable AI techniques to understand the factors influencing predictions.
- Improve model generalization through additional validation and testing.
- Add student performance trend analysis and visualization.
- Develop a role-based dashboard for teachers and academic administrators.
- Deploy the frontend and backend as independently hosted services.
- Add automated testing and continuous integration.
- Evaluate fairness across relevant student groups.

## Author

**Shougata**

GitHub: [@Shougata2003](https://github.com/Shougata2003)

## Repository

[Student Performance Prediction Using ANN](https://github.com/Shougata2003/Student-Performance-ANN)

---

*Built as a machine learning and full-stack application project using Python, TensorFlow, FastAPI, and Streamlit.*
