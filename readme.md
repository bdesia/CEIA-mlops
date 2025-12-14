Markdown## Machine Learning Operations 1
## CEIA - FIUBA
## 5º Bimestre 2025

## Trabajo Práctico Final

### Grupo

Autor: Juan Nervi

---
# **DIAMETRAL DEFORMATION PREDICTION IN CNE NPP PRESSURE TUBES**: an AI solution for outer diameter rate forecasting

## Descripción

El presente servicio representa una solución basada en IA para la predicción confiable de la tasa de deformación diametral (**MeanRateOutDiam**) en tubos de presión de la Central Nuclear Embalse (CNE), Argentina, durante la Inspección en Servicio ISI 2024.

El modelo utiliza mediciones de diámetro interno, espesor, flujo neutrónico extendido, presión y temperatura para predecir la evolución de la deformación. El usuario selecciona un canal (ej: E06) y el servicio devuelve la predicción completa a lo largo del tubo, junto con valores reales y métricas.

## Fuente

Los datos provienen de mediciones reales ISI 2024 (diámetro y espesor por canal), bases de flujo neutrónico, perfiles de presión/temperatura BOL y datos de diseño de canales.

El pipeline ETL procesa ~380 canales, genera flujo extendido, unifica mediciones y enriquece con variables operativas.

## Componentes del proyecto

1. **Apache Airflow**  
   - Orquestación del ETL completo: descarga de datos, procesamiento, enriquecimiento y generación de datasets.

2. **MLflow**  
   - Tracking de experimentos, búsqueda de hiperparámetros (Optuna), registro de modelo con alias "champion".

3. **MinIO**  
   - Almacenamiento S3-compatibile de datos crudos, intermedios, procesados y artefactos MLflow.

4. **FastAPI**  
   - Servicio de inferencia REST que carga el modelo champion y predice por canal.



## Flujo de interacción

1. ETL Process


- Flujo de trabajo :
upload_raw_data_zip >> generate_extended_flux >> process_and_enrich_all >> split_train_val_test >> generate_ml_datasets + generate_model_compatible_data
- upload_raw_data_zip: Descarga única como ZIP desde GitHub y subida a S3 raw/.
- generate_extended_flux: Ajuste de curvas para flujo neutrónico extendido por canal (~380 archivos .flx).
- process_and_enrich_all: Unificación diámetro/espesor, interpolación, enriquecimiento con P/T/Flux, cálculo de tasas (MeanRateOutDiam, etc.).
- split_train_val_test: División por canal (70% train, 10% val, 30% test).
- generate_model_compatible_data: Genera data_by_ch.pkl, channel_list.csv y parquet por canal para inferencia rápida.

2. Model Experimentation Process
- Modelo base: CNN1D Regressor con contexto variable y downsampling.
- Búsqueda de hiperparámetros con Optuna (60 trials), early stopping y ReduceLROnPlateau.
- Entrenamiento final con mejores parámetros, evaluación en test y registro como "champion".

3. Production Process
- El usuario consulta la API FastAPI seleccionando un canal.
- La API carga el modelo champion + scalers desde MLflow/MinIO.
- Genera predicciones para todo el canal y devuelve resultados (JSON + gráficos opcionales).

## Corriendo el servicio

### Pre-requisitos de instalación

- Git
- Docker y Docker Compose

Para notebooks:
- Python 3.11+
- Poetry o pip
- Jupyter

### Step 0: Clonar repositorio

    ```bash
    git clone https://github.com/jern10/CEIA-mlops.git
    cd CEIA-mlops
    ```

###  Step 1: Inicializar el entorno (recomendado) y el servicio
El proyecto incluye un script setup.sh que configura todo el entorno automáticamente:

    ```bash
    ./setup.sh
    ```

En la carpeta raíz de este repositorio, correr el siguiente comando para inicializar el servicio completo utilizando Docker Compose:

```bash
docker compose up postgres -d
```

Importante para Windows: Asegurarse de tener Docker Desktop ejecutándose.

Para asegurarte de que todos los servicios estén en estado *healthy*, revisa en Docker Desktop o escribe el comando:

```bash
docker ps -a
```


###  Step 2: Ejecutar ETL y entrenamiento

1. Accede a Airflow: http://localhost:8080 (user: airflow / pass: airflow)
1. Trigger el DAG diametral_deformation_etl_optimized
1. Una vez finalizado, ejecuta el notebook notebooks/HP_tuning.ipynb para entrenar y registrar el modelo

###  Step 3: Levantar la API de predicción

    ```bash
    compose up diameter-predictor -d
    ```

Detalles de acceso

1. Airflow
- URL: http://localhost:8080
- Credenciales: airflow / airflow

1. MLflow
- URL: http://localhost:5001

1. MinIO
- URL: http://localhost:9001
- Credenciales: Access Key minio / Secret Key minio123

1. FastAPI: Diameter Deformation Predictor
- URL: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs


###  Step 4: Usar el predictor
En http://localhost:8000/docs:

- GET /channels → lista de canales disponibles
- POST /predict → POST con {"channel": "E06"} → predicción completa (posiciones, real vs predicho, RMSE/MAE)

## Corriendo experimentos
###  Step 1: ETL process
En Airflow UI, ejecutar el DAG diametral_deformation_etl_optimized.

###  Step 2: Hyper-parameter tuning
Desde la carpeta /notebooks:

    ```bash
    poetry run jupyter notebook
    ```

## Pendientes a futuro

- Trigger automático de entrenamiento desde Airflow tras ETL.
- Monitoreo de data/model drift.
- Frontend web interactivo con gráficos.
- Batch prediction para múltiples canales.
