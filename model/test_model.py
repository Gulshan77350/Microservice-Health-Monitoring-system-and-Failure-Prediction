import joblib
import pandas as pd

model = joblib.load(
    "failure_predictor.pkl"
)

sample = pd.DataFrame([
    {
        "cpu": 40,
        "memory": 55,
        "latency": 200,
        "requests": 2000,
        "error_rate": 1
    }
])
prediction = model.predict(sample)

print(prediction)