import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
import uvicorn

### fast api comes with pydantic for data validation built-in

### to do data profiling on the input data

### for c in categorical:
###    print(df[c].value_counts())
###    print()

### for n in numerical:
###    print(df[n].describe())
###    print()
### paste the output into Chatgpt to get the pydantic model below

class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

# Load model.bin safely. If the file is missing or loading fails, keep pipeline=None
pipeline = None
try:
    with open('model.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
except Exception as e:
    logging.warning(f"Could not load model.bin: {e}")


def predict_single(customer):
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Model is not loaded')
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


@app.get("/")
def root():
    """Simple root endpoint to confirm the service is running."""
    return JSONResponse({
        "message": "customer-churn-prediction service running",
        "predict": "POST /predict",
    })


@app.get("/health")
def health():
    """Health endpoint reporting whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)



### example input for prediction endpoint - open up a new terminal
### paste the following curl command to test the prediction endpoint
### curl -s -X POST -H "Content-Type: application/json" -d '{"gender":"female","seniorcitizen":0,"partner":"yes","dependents":"no","phoneservice":"no","multiplelines":"no_phone_service","internetservice":"dsl","onlinesecurity":"no","onlinebackup":"yes","deviceprotection":"no","techsupport":"no","streamingtv":"no","streamingmovies":"no","contract":"month-to-month","paperlessbilling":"yes","paymentmethod":"electronic_check","tenure":1,"monthlycharges":29.85,"totalcharges":29.85}' http://localhost:9696/predict
