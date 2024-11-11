from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# Instancia de FastAPI
app = FastAPI()

# Cargar el modelo de predicción preentrenado
model_path = "modelo_precio_coche.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en la ruta '{model_path}'")

# Asegúrate de que el modelo esté correctamente cargado
try:
    loaded_model = joblib.load(model_path)  # Cargar el modelo como loaded_model
    print(f"Modelo cargado exitosamente: {type(loaded_model)}")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

# Configuración de plantillas y archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Clase para manejar los datos del formulario
class CarData(BaseModel):
    brand: str
    model: str
    vehicle_age: int
    km_driven: int
    seller_type: str
    fuel_type: str
    transmission_type: str
    mileage: float
    engine: int
    max_power: float
    seats: int

# Ruta para mostrar el formulario inicial
@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Ruta para realizar la predicción
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    brand: str = Form(...),
    model: str = Form(...),
    vehicle_age: int = Form(...),
    km_driven: int = Form(...),
    seller_type: str = Form(...),
    fuel_type: str = Form(...),
    transmission_type: str = Form(...),
    mileage: float = Form(...),
    engine: int = Form(...),
    max_power: float = Form(...),
    seats: int = Form(...)
):
    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }])

    # Ajustar las columnas de input_data al formato esperado por el modelo
    expected_columns = getattr(loaded_model, "feature_names_in_", input_data.columns)
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Realizar la predicción
    try:
        predicted_price = loaded_model.predict(input_data)[0]  # Usa loaded_model en lugar de model
    except Exception as e:
        print("Error en la predicción:", e)
        return templates.TemplateResponse("form.html", {
            "request": request,
            "predicted_price": "Error al hacer la predicción, revise los datos ingresados."
        })

    # Mostrar el precio predicho en la plantilla
    return templates.TemplateResponse("form.html", {
        "request": request,
        "predicted_price": f"El precio estimado del vehículo es: ${predicted_price:,.2f}"
    })
