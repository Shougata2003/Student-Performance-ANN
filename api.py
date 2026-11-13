
import os
import secrets
import logging
from pathlib import Path

import pandas as pd
import joblib
import tensorflow as tf

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Request,
    Security
)

from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

APP_ENV = os.getenv("APP_ENV", "development").lower()

# Load local secrets only during development
if APP_ENV != "production":
    load_dotenv(BASE_DIR / ".env")

APP_ENV = os.getenv("APP_ENV", "development").lower()

API_KEY = os.getenv("API_KEY", "").strip()

if len(API_KEY) < 32:
    raise RuntimeError(
        "A valid API_KEY of at least 32 characters is required."
    )

IS_PRODUCTION = APP_ENV == "production"


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

logger = logging.getLogger("student_performance_api")


# ==================================================
# 3. TRAINING FEATURE ORDER
# ==================================================

FEATURE_COLUMNS = [
    "age",
    "attendance_pct",
    "study_hours_per_week",
    "previous_score",
    "assignment_completion_pct",
    "internal_assessment_score",
    "participation_score",
    "sleep_hours_per_day",
    "extracurricular_hours_per_week",
    "failed_subjects"
]


# ==================================================
# 4. LOAD TRAINED MODEL AND PREPROCESSORS
# ==================================================

try:

    model = tf.keras.models.load_model(
        MODEL_DIR / "student_performance.keras"
    )

    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    logger.info("Model artifacts loaded successfully.")

    # Verify feature names and their order
    if hasattr(scaler, "feature_names_in_"):

        scaler_features = list(
            scaler.feature_names_in_
        )

        if scaler_features != FEATURE_COLUMNS:

            raise RuntimeError(
                "Scaler feature names or order do not match API features."
            )

    # Verify model input dimension
    if model.input_shape[-1] != len(FEATURE_COLUMNS):

        raise RuntimeError(
            "ANN input dimension does not match the feature count."
        )

    # Verify output classes
    if model.output_shape[-1] != len(encoder.classes_):

        raise RuntimeError(
            "ANN output dimension does not match encoder classes."
        )

    logger.info("Model feature configuration verified.")

except Exception:
    logger.exception("Model initialization failed.")
    raise RuntimeError(
        "Prediction service initialization failed."
    )


# ==================================================
# 5. RATE LIMITING
# ==================================================

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://"
)


# ==================================================
# 6. INITIALIZE FASTAPI
# ==================================================

app = FastAPI(
    title="Student Performance Prediction API",
    description="ANN-based student performance prediction service.",
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# ==================================================
# 7. API KEY AUTHENTICATION
# ==================================================

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str | None = Security(api_key_header)
):

    if not api_key:

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    if not secrets.compare_digest(
        api_key,
        API_KEY
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid API credentials."
        )


# ==================================================
# 8. INPUT VALIDATION
# ==================================================

class StudentData(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    age: int = Field(
        ge=17, le=24
    )

    attendance_pct: float = Field(
        ge=45, le=100, allow_inf_nan=False
    )

    study_hours_per_week: float = Field(
        ge=2, le=40, allow_inf_nan=False
    )

    previous_score: float = Field(
        ge=25, le=100, allow_inf_nan=False
    )

    assignment_completion_pct: float = Field(
        ge=20, le=100, allow_inf_nan=False
    )

    internal_assessment_score: float = Field(
        ge=20, le=100, allow_inf_nan=False
    )

    participation_score: float = Field(
        ge=1, le=10, allow_inf_nan=False
    )

    sleep_hours_per_day: float = Field(
        ge=4, le=10, allow_inf_nan=False
    )

    extracurricular_hours_per_week: float = Field(
        ge=0, le=20, allow_inf_nan=False
    )

    failed_subjects: int = Field(
        ge=0, le=5
    )


# ==================================================
# 9. PUBLIC ENDPOINTS
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Student Performance Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==================================================
# 10. PROTECTED PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
@limiter.limit("30/minute")
def predict(
    request: Request,
    data: StudentData,
    _: None = Depends(verify_api_key)
):

    try:

        # Convert validated input into a DataFrame.
        # Preserve the exact training feature names and order.

        features = pd.DataFrame(
            [data.model_dump()],
            columns=FEATURE_COLUMNS
        )

        # Apply the previously fitted scaler

        scaled_features = scaler.transform(
            features
        )

        # Generate ANN prediction probabilities

        probabilities = model.predict(
            scaled_features,
            verbose=0
        )[0]

        # Verify prediction output

        if len(probabilities) != len(encoder.classes_):
            raise RuntimeError(
                "Unexpected number of prediction classes."
            )

        # Find predicted class

        predicted_index = int(
            probabilities.argmax()
        )

        predicted_category = encoder.inverse_transform(
            [predicted_index]
        )[0]

        # Format response

        result = {
            "predicted_performance": str(
                predicted_category
            ),

            "probabilities": {
                str(category): round(
                    float(probability), 4
                )

                for category, probability in zip(
                    encoder.classes_,
                    probabilities
                )
            }
        }

        return result

    except Exception:

        logger.exception(
            "Prediction processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed."
        )
