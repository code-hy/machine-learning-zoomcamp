
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn





# Load the model
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

#Create FASTAPI app
app = FastAPI()

class Lead(BaseModel):
    lead_source:str
    number_of_courses_viewed: int
    annual_income: float


@app.post('/predict')
def predict(client: dict):
    # Convert the client data to a dictionary
    client_dict = [client]

    # Make a prediction
    prediction = pipeline.predict_proba(client_dict)[0, 1]

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
