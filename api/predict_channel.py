# predict_channel.py
# Script para predecir MeanRateOutDiam en un canal específico usando el modelo champion

import os
import io
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow
import matplotlib.pyplot as plt
import boto3
import awswrangler as wr

# Importar componentes reutilizables
from src.model import CNN1DRegressor
from src.dataset import DiameterDataset, custom_collate_fn

# =============================================================================
# Configuración de entorno (MinIO local - ajusta si usas AWS real)
# =============================================================================
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"

MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

BUCKET_NAME = "data"
MODEL_NAME = "diameter_cnn1d_prod"
ALIAS = "champion"

# =============================================================================
# Cliente S3
# =============================================================================
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def load_artifact_from_run(run_id: str, artifact_path: str):
    """Descarga un artefacto de un run específico"""
    obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"mlflow_artifacts/{run_id}/artifacts/{artifact_path}"
    )
    return io.BytesIO(obj['Body'].read())

# =============================================================================
# Cargar datos y lista de canales
# =============================================================================
def load_data_and_scalers():
    print("Cargando data_by_ch.pkl...")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key="processed/model_compatible/data_by_ch.pkl")
    data_by_ch = pickle.load(io.BytesIO(obj['Body'].read()))

    print("Cargando lista de canales...")
    channel_list = wr.s3.read_csv(f"s3://{BUCKET_NAME}/processed/model_compatible/channel_list.csv")["channel"].tolist()

    return data_by_ch, channel_list

# =============================================================================
# Cargar modelo champion y sus artefactos (scalers)
# =============================================================================
def load_champion_model():
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    print(f"Buscando modelo registrado '{MODEL_NAME}' con alias '{ALIAS}'...")
    model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run_id = model_version.run_id
    print(f"Versión {model_version.version} (run_id: {run_id}) encontrada.")

    # Cargar modelo PyTorch
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    print("Cargando modelo PyTorch...")
    model = mlflow.pytorch.load_model(model_uri)

    # Cargar scalers (guardados como artefactos en el run final)
    print("Cargando scalers...")
    feature_scaler_bytes = load_artifact_from_run(run_id, "feature_scaler.pkl")
    target_scaler_bytes = load_artifact_from_run(run_id, "target_scaler.pkl")
    
    import joblib
    feature_scaler = joblib.load(feature_scaler_bytes)
    target_scaler = joblib.load(target_scaler_bytes)

    # Obtener hiperparámetros del run (para window_size y max_context)
    run = client.get_run(run_id)
    params = run.data.params
    window_size = int(params.get("best_window_size", 21))   # fallback razonable
    max_context = int(params.get("best_max_context", 121))

    return model, feature_scaler, target_scaler, window_size, max_context, run_id

# =============================================================================
# Predicción para un canal
# =============================================================================
def predict_channel(channel: str, data_by_ch, model, feature_scaler, target_scaler, window_size, max_context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    print(f"Generando predicciones para el canal {channel}...")

    # Dataset solo para inferencia (min_context = max_context = window_size para usar siempre el mismo contexto fijo)
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
    predictions_original = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Obtener posiciones axiales reales
    df_channel = data_by_ch[channel]
    axial_positions = df_channel['Axial_m'].values

    real_values = df_channel['MeanRateOutDiam'].values

    return axial_positions, real_values, predictions_original

# =============================================================================
# Graficar y guardar resultados
# =============================================================================
def plot_and_save(axial, real, pred, channel, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(axial, real, label="Real", linewidth=2)
    plt.plot(axial, pred, label="Predicción", linewidth=2, alpha=0.9)
    plt.title(f"MeanRateOutDiam - Canal {channel}")
    plt.xlabel("Posición axial (m)")
    plt.ylabel("MeanRateOutDiam")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"{channel}_prediction.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado en: {plot_path}")

    # Guardar CSV con resultados
    results_df = pd.DataFrame({
        "Axial_m": axial,
        "Real": real,
        "Predicción": pred,
        "Error_abs": np.abs(real - pred)
    })
    csv_path = os.path.join(output_dir, f"{channel}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir MeanRateOutDiam para un canal usando el modelo champion")
    parser.add_argument("--channel", type=str, required=True, help="Código del canal (ej: E06)")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Carpeta para guardar gráficos y CSV")
    
    args = parser.parse_args()

    # Carga de datos
    data_by_ch, channel_list = load_data_and_scalers()

    if args.channel not in channel_list:
        raise ValueError(f"Canal {args.channel} no encontrado. Canales disponibles: {channel_list[:10]}... (total {len(channel_list)})")

    # Carga de modelo y scalers
    model, feature_scaler, target_scaler, window_size, max_context, run_id = load_champion_model()
    print(f"Modelo cargado (window_size={window_size}, max_context={max_context})")

    # Predicción
    axial, real, pred = predict_channel(
        channel=args.channel,
        data_by_ch=data_by_ch,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window_size=window_size,
        max_context=max_context
    )

    # Resultados
    plot_and_save(axial, real, pred, args.channel, args.output_dir)

    # Métricas rápidas
    rmse = np.sqrt(np.mean((real - pred)**2))
    mae = np.mean(np.abs(real - pred))
    print(f"\n=== Métricas para canal {args.channel} ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")