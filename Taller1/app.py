from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import your_model_file  # Importa tu archivo de modelo

app = FastAPI()

# Define la estructura de entrada de la solicitud
class PredictionRequest(BaseModel):
    features: List[float]

# Define la ruta de la API para realizar la predicción
@app.post("/predict")
def predict(request: PredictionRequest):
    # Obtén las características de la solicitud
    features = request.features

    # Realiza la predicción utilizando tu modelo
    prediction = your_model_file.predict(features)

    # Devuelve la predicción como respuesta
    return {"prediction": prediction}