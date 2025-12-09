# upload_data_to_s3.py
# Versión corregida para MinIO

import os
import awswrangler as wr
import boto3
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------------------
# Cargar variables de entorno desde .env
# ----------------------------------------------------------------
load_dotenv()

# Verificar credenciales
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    raise EnvironmentError(
        "No se encontraron credenciales en .env. "
        "Define AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY."
    )

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT_URL")
if not MINIO_ENDPOINT:
    raise EnvironmentError("Define MINIO_ENDPOINT_URL en .env (ej: http://localhost:9000)")

print(f"Credenciales cargadas desde .env")
print(f"Usando endpoint MinIO: {MINIO_ENDPOINT}")

# ----------------------------------------------------------------
# Crear sesión boto3 PERSONALIZADA con endpoint para MinIO
# ----------------------------------------------------------------
boto3_session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

# Cliente S3 personalizado con el endpoint de MinIO
s3_client = boto3_session.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,           # <-- Aquí se define el endpoint
    config=boto3.session.Config(signature_version='s3v4')
)

# ----------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------
LOCAL_DATA_PATH = "./data"
S3_BASE_PATH = "s3://data/raw/"

# ----------------------------------------------------------------
# Verificaciones iniciales
# ----------------------------------------------------------------
if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"La carpeta local no existe: {LOCAL_DATA_PATH}")

local_path = os.path.abspath(LOCAL_DATA_PATH)
s3_path = S3_BASE_PATH.rstrip("/") + "/"

print(f"Preparando subida...")
print(f"   Desde: {local_path}")
print(f"   Hacia: {s3_path}")

# ----------------------------------------------------------------
# Subida recursiva de todos los archivos
# ----------------------------------------------------------------
try:
    print("Iniciando subida recursiva...")

    uploaded_count = 0

    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_file_path = s3_path + relative_path.replace(os.sep, "/")

            print(f"Subiendo: {local_file} → {s3_file_path}")
            
            # Usar la sesión personalizada (con endpoint ya configurado)
            wr.s3.upload(
                local_file=local_file,
                path=s3_file_path,
                boto3_session=boto3_session   # <-- Solo esto, sin endpoint_url aquí
            )
            uploaded_count += 1

    print(f"¡Subida completada con éxito! {uploaded_count} archivos subidos.")

    # Listar objetos (también con la sesión personalizada)
    objects = wr.s3.list_objects(
        path=s3_path,
        boto3_session=boto3_session
    )
    print(f"Total de objetos ahora en {s3_path}: {len(objects)}")

except Exception as e:
    print(f"Error durante la subida: {e}")
    raise