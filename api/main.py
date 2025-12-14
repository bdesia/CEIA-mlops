# api/main.py
import os
import io
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import mlflow
import boto3
import awswrangler as wr
import joblib

# Importar componentes reutilizables
from src.model import CNN1DRegressor
from src.dataset import DiameterDataset, custom_collate_fn
from torch.utils.data import DataLoader

# =============================================================================
# Configuración de entorno (MinIO local - cambia para producción)
# =============================================================================
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ENDPOINT_URL_S3"] = os.getenv("AWS_ENDPOINT_URL_S3", "http://localhost:9000")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

BUCKET_NAME = "data"
MODEL_NAME = "diameter_cnn1d_prod"
ALIAS = "champion"

# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(
    title="Diameter Deformation Prediction API",
    description="Predice MeanRateOutDiam para tubos de presión CNE usando modelo CNN1D champion",
    version="1.0.0"
)

# =============================================================================
# Cliente S3
# =============================================================================
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# =============================================================================
# Carga global: modelo, scalers, data_by_ch, canales disponibles
# =============================================================================
@app.on_event("startup")
async def load_model_and_data():
    global model, feature_scaler, target_scaler, window_size, max_context, data_by_ch, available_channels

    print("Cargando modelo champion y artefactos...")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run_id = model_version.run_id
    print(f"Modelo champion versión {model_version.version} (run_id: {run_id})")

    # Cargar modelo
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo cargado en {device}")

    # Cargar scalers desde artefactos del run
    def load_scaler(artifact_name):
        obj = s3_client.get_object(
            Bucket=BUCKET_NAME,
            Key=f"mlflow_artifacts/{run_id}/artifacts/{artifact_name}"
        )
        return joblib.load(io.BytesIO(obj['Body'].read()))

    feature_scaler = load_scaler("feature_scaler.pkl")
    target_scaler = load_scaler("target_scaler.pkl")
    print("Scalers cargados")

    # Parámetros del modelo (window_size, max_context)
    run = client.get_run(run_id)
    params = run.data.params
    window_size = int(params.get("best_window_size", 21))
    max_context = int(params.get("best_max_context", 121))
    print(f"Usando window_size={window_size}, max_context={max_context}")

    # Cargar datos de canales
    print("Cargando data_by_ch.pkl...")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key="processed/model_compatible/data_by_ch.pkl")
    data_by_ch = pickle.load(io.BytesIO(obj['Body'].read()))

    print("Cargando lista de canales disponibles...")
    channel_df = wr.s3.read_csv(f"s3://{BUCKET_NAME}/processed/model_compatible/channel_list.csv")
    available_channels = set(channel_df["channel"].tolist())

    print(f"API lista. {len(available_channels)} canales disponibles.")

# =============================================================================
# Modelos Pydantic
# =============================================================================
class PredictionRequest(BaseModel):
    channel: str = Field(..., description="Código del canal, ej: E06")

class PredictionResponse(BaseModel):
    channel: str
    axial_positions: List[float]
    real_values: List[float]
    predicted_values: List[float]
    rmse: float
    mae: float
    message: str = "Predicción completada"

# =============================================================================
# Endpoint de salud
# =============================================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, "alias": ALIAS}

# =============================================================================
# Endpoint principal de predicción
# =============================================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    channel = request.channel.upper()

    if channel not in available_channels:
        raise HTTPException(status_code=404, detail=f"Canal {channel} no encontrado. Usa /channels para ver disponibles.")

    if channel not in data_by_ch:
        raise HTTPException(status_code=500, detail=f"Datos del canal {channel} no cargados correctamente.")

    device = next(model.parameters()).device

    # Dataset para inferencia (contexto fijo)
    infer_dataset = DiameterDataset(
        data_dict=data_by_ch,
        ch_list=[channel],
        window_size=window_size,
        min_context=window_size,
        max_context=max_context,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scaler=False,
        data_stride=1,
        seed=42
    )

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    predictions_scaled = []
    with torch.no_grad():
        for X, _, _ in infer_loader:
            X = X.to(device)
            out = model(X)
            predictions_scaled.extend(out.cpu().numpy())

    predictions_scaled = np.array(predictions_scaled)
    predicted_values = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Valores reales y posiciones
    df = data_by_ch[channel]
    axial_positions = df['Axial_m'].values.tolist()
    real_values = df['MeanRateOutDiam'].values.tolist()

    # Métricas
    rmse = float(np.sqrt(np.mean((np.array(real_values) - predicted_values)**2)))
    mae = float(np.mean(np.abs(np.array(real_values) - predicted_values)))

    return PredictionResponse(
        channel=channel,
        axial_positions=axial_positions,
        real_values=real_values,
        predicted_values=predicted_values.tolist(),
        rmse=rmse,
        mae=mae
    )

# =============================================================================
# Endpoint: lista de canales disponibles
# =============================================================================
@app.get("/channels")
async def get_channels():
    return {"available_channels": sorted(list(available_channels))}