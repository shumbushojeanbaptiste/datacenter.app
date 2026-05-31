from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

temp_model = joblib.load("models/temp_model.pkl")
hum_model = joblib.load("models/hum_model.pkl")
risk_model = joblib.load("models/risk_model.pkl")
le = joblib.load("models/risk_encoder.pkl")

class Request(BaseModel):
    month: int
    dayofweek: int
    hour: int
    minute: int


@app.post("/predict")
def predict(data: Request):

    # =====================
    # INPUT
    # =====================
    # please set better lags and means, this is just for testing on standard input

    history = [25, 25, 26]
    hum_history = [60, 61, 62]
    input_base = pd.DataFrame([{
        "month": data.month,
        "dayofweek": data.dayofweek,
        "hour": data.hour,
        "minute": data.minute,

        # fallback lags (use last known default)
        'temp_lag1': history[-1],
        'temp_lag2': history[-2],
        "hum_lag1": hum_history[-1],
        "hum_lag2": hum_history[-2],
        "temp_mean3": history[-3],
        "hum_mean3": hum_history[-3]
    }])

    # =====================
    # PREDICT TEMP
    # =====================
    temp = temp_model.predict(input_base)[0]
        # VERY IMPORTANT
    history.append(temp)
    # =====================
    # PREDICT HUM
    # =====================
    hum = hum_model.predict(input_base)[0]
    hum_history.append(hum)
    # =====================
    # RISK INPUT
    # =====================
    risk_input = pd.DataFrame([{
        "temperature": temp,
        "humidity": hum,
        "month": data.month,
        "dayofweek": data.dayofweek,
        "hour": data.hour,
        "minute": data.minute,
    }])

    risk_pred = risk_model.predict(risk_input)
    risk = le.inverse_transform(risk_pred)[0]

    return {
        "temperature": round(temp, 2),
        "humidity": round(hum, 2),
        "risk": risk
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
