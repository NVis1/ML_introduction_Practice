import sys
from typing import Literal

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel as DanticModel

app = FastAPI()


"""
🔹 3. Обёртка модели в API (FastAPI или Flask)
- Написать API с POST-запросом для предсказания оттока клиента.
- Входные данные → JSON с характеристиками клиента.
- Выходные данные → {"prediction": 1} или {"prediction": 0}.
"""


class UserInfo(DanticModel):
    tenure: int
    TotalCharges: float

    Contract: Literal["Month-to-month", "One year", "Two year"]

    InternetService: Literal["No", "DSL", "Fiber optic"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]

    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No phone service", "No", "Yes"]


model = None


@app.get("/")
async def root():
    return RedirectResponse("/docs/")

    
@app.post("/predict/")
async def root(userinfo: UserInfo):
    print(pd.DataFrame([userinfo.model_dump()]))
    return {"prediction": model.predict(
        pd.DataFrame([ userinfo.model_dump() ])
    ).tolist()[0]}


def throw(err_msg):
    sys.stderr.write(err_msg)
    sys.exit(1)


def get_model(filename):
    try:
        res = joblib.load(filename)
    except FileNotFoundError:
        throw(f"\nCould not load model from a file named \"{filename}\" which was provided. "
              f"Check the provided filename before launching.")
    if not hasattr(res, "predict"):
        throw("Provided model cannot predict. Check if your model is valid.")
    return res


def main(model_filename):
    global model
    model = get_model(model_filename)

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == '__main__':
    main(model_filename="model-newest.joblib")
