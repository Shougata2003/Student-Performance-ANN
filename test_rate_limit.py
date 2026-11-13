
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

URL = "http://127.0.0.1:8000/predict"

headers = {
    "X-API-Key": API_KEY
}

student_data = {
    "age": 20,
    "attendance_pct": 85,
    "study_hours_per_week": 20,
    "previous_score": 75,
    "assignment_completion_pct": 90,
    "internal_assessment_score": 75,
    "participation_score": 8,
    "sleep_hours_per_day": 7,
    "extracurricular_hours_per_week": 5,
    "failed_subjects": 0
}

print("Starting rate-limit test...")

for i in range(1, 36):

    response = requests.post(
        URL,
        json=student_data,
        headers=headers,
        timeout=30
    )

    print(
        f"Request {i}: HTTP {response.status_code}"
    )

    if response.status_code == 429:
        print("\nRate limit successfully triggered!")
        break

else:
    print("\nRate limit was not triggered.")
