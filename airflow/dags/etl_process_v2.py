# ----------------------------------------------------------------
# DAG: Diametral Deformation of CNE NPP fuel channels - ETL process (OPTIMIZED v2)
# ----------------------------------------------------------------
import os
from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# ----------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------
default_args = {
    'owner': 'Juan Nervi',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=60),
}

md_text = """
# ETL - Diametral deformation of CNE NPP pressure tubes (ISI 2024) - OPTIMIZED
Full Pipeline:
- **Download raw data as ZIP from GitHub (single request)**
- Augment neutronic flux database for PT.
- Process inner diameter and thickness measurements.
- Enrich with P, T, Flux and calculate rates
- Train/Val/Test split by channel
- Generate model-compatible data_by_ch.pkl
- Generate channel_list.csv for easy channel selection in DataLoader
"""

S3_BASE_PATH = "s3://data/"
RAW_DATA_PATH = os.path.join(S3_BASE_PATH, "raw/")
INTERIM_PATH = os.path.join(S3_BASE_PATH, "interim/")
PROCESSED_PATH = os.path.join(S3_BASE_PATH, "processed/")

@dag(
    dag_id="diametral_deformation_etl_optimized",
    description="ETL optimizado para deformación diametral de tubos de presión CNE",
    doc_md=md_text,
    tags=["deformacion", "diametral", "isi2024", "nuclear", "etl", "optimized"],
    default_args=default_args,
    catchup=False,
    schedule_interval=None,
    start_date=days_ago(1),
)
def etl_pipeline():
    # ------------------------------------------------------------
    # 0. Download raw data as ZIP from GitHub and upload to S3 raw/
    # ------------------------------------------------------------
    @task.virtualenv(
        task_id="upload_raw_data_zip",
        requirements=["awswrangler==3.6.0", "requests", "pandas"],
        system_site_packages=False,
    )
    def upload_raw_data_zip(s3_raw_path: str):
        import requests
        import zipfile
        import io
        import awswrangler as wr
        import os
        from pathlib import Path

        ZIP_URL = "https://github.com/jern10/CEIA-mlops/archive/amq2-main.zip"
        
        print("Descargando ZIP completo del repositorio...")
        r = requests.get(ZIP_URL, stream=True, timeout=600)
        r.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            prefix = "CEIA-mlops-amq2-main/data/"
            data_files = [f for f in z.namelist() if f.startswith(prefix) and not f.endswith('/')]
            
            uploaded = 0
            for file_path in data_files:
                rel_path = file_path[len(prefix):]
                if rel_path:
                    with z.open(file_path) as source:
                        s3_key = f"{s3_raw_path}{rel_path.replace(os.sep, '/')}"
                        wr.s3.upload(local_file=source, path=s3_key)
                        uploaded += 1
                        if uploaded % 100 == 0:
                            print(f"Subidos {uploaded} archivos...")
        
        print(f"Upload completado: {uploaded} archivos subidos a {s3_raw_path}")
        return uploaded

    @task.virtualenv(
        task_id="get_evaluation_time",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=False,
    )
    def get_evaluation_time(raw_data_path: str) -> float:
        import awswrangler as wr
        df = wr.s3.read_csv(f"{raw_data_path}evaluation_time.csv")
        time_efph = float(df.iloc[0, 0])
        print(f"Evaluation time loaded: {time_efph} EFPH")
        return time_efph

    # ----------------------------------------------------------------
    # 1. Augment neutronic flux db for PT
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="generate_extended_flux",
        requirements=["pandas>=1.5", "numpy>=1.21", "scipy>=1.9", "awswrangler==3.6.0"],
        system_site_packages=False,
    )
    def generate_extended_flux(raw_data_path: str, interim_path: str):
        import numpy as np
        import pandas as pd
        import awswrangler as wr
        from scipy.optimize import curve_fit
        from scipy.interpolate import interp1d
        import warnings
        import os

        class FluxCurve:
            def __init__(self, kparam=5, nbundles=12):
                self.kparam = kparam
                self.nbundles = nbundles
                self.A = None
                self.B = None
                self.fitted = False

            def _model(self, x, A, B):
                global_func = A * np.sin(np.pi * x)
                local_func = B * (np.abs(np.sin(self.nbundles * np.pi * x))) ** (1 / self.kparam)
                return global_func + local_func

            def fit(self, xdata, ydata):
                p0 = np.max(ydata) * np.array([1.03, 0.4])
                bounds = ([0.9 * p0[0], 0.2 * p0[1]], [1.5 * p0[0], 1.1 * p0[1]])
                try:
                    best_params, _ = curve_fit(self._model, xdata, ydata, p0=p0, bounds=bounds, maxfev=10000)
                    self.A, self.B = best_params
                    self.fitted = True
                except Exception as e:
                    warnings.warn(f"Curve fit failed: {e}")
                    self.fitted = False

            def predict(self, xdata):
                if not self.fitted:
                    raise RuntimeError("Model not fitted")
                return self._model(xdata, self.A, self.B)

        df_all = wr.s3.read_csv(f"{raw_data_path}flux/pt_flux_1MeV.csv")
        os.makedirs("/tmp/flux_extended", exist_ok=True)
        
        for _, row in df_all.iterrows():
            canal = row['Canal']
            ntrozos = len(row) - 1
            coord_rel = [0] + [(i - 0.5) / ntrozos for i in range(1, ntrozos + 1)] + [1]
            flux_1mev = [0] + [row[f'Trozo_{i}'] for i in range(1, ntrozos + 1)] + [0]
            xdata = np.array(coord_rel)
            ydata = np.array(flux_1mev)
            
            model = FluxCurve()
            model.fit(xdata, ydata)
            ext_coord = np.linspace(0, 1, 1200)
            
            try:
                if model.fitted:
                    ext_flux = model.predict(ext_coord)
                else:
                    raise RuntimeError()
            except:
                try:
                    f = interp1d(xdata, ydata, kind='cubic', fill_value='extrapolate')
                    ext_flux = f(ext_coord)
                except:
                    f = interp1d(xdata, ydata, kind='linear', fill_value='extrapolate')
                    ext_flux = f(ext_coord)
            
            out_df = pd.DataFrame({'coord_rel': ext_coord, 'flux_1mev': ext_flux})
            local_path = f"/tmp/flux_extended/{canal}.flx"
            out_df.to_csv(local_path, index=False, sep='\t', header=False)
            s3_path = f"{interim_path}flux/pressure_tubes_extended/{canal}.flx"
            wr.s3.upload(local_path, s3_path)
        
        print("Flux extendido generado para todos los canales")

    # ----------------------------------------------------------------
    # 2-3. Process measurements + enrich and calculate rates
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="process_and_enrich_all",
        requirements=["pandas", "numpy", "awswrangler==3.6.0"],
        system_site_packages=False,
    )
    def process_and_enrich_all(raw_data_path: str, interim_path: str, processed_path: str, evaluation_time_efph: float):
        import pandas as pd
        import numpy as np
        import awswrangler as wr
        import os

        years = evaluation_time_efph / (24 * 365.25)
        channel_data = wr.s3.read_csv(f"{raw_data_path}channel_data.csv")

        folder_diam = f"{raw_data_path}isi2024/diameter/"
        folder_thick = f"{raw_data_path}isi2024/thickness/"
        diam_files = wr.s3.list_objects(folder_diam)
        thick_files = wr.s3.list_objects(folder_thick)

        idiam_by_ch = {}
        thick_by_ch = {}
        for file in diam_files:
            if not file.endswith('.csv'): continue
            basename = os.path.basename(file)
            ch = next((p.upper() for p in basename.replace('.csv','').split('_') if len(p)==3 and p.isalnum()), None)
            if ch: 
                df = wr.s3.read_csv(file, skiprows=15)
                idiam_by_ch[ch] = df
        for file in thick_files:
            if not file.endswith('.csv'): continue
            basename = os.path.basename(file)
            ch = next((p.upper() for p in basename.replace('.csv','').split('_') if len(p)==3 and p.isalnum()), None)
            if ch: 
                df = wr.s3.read_csv(file, skiprows=15)
                thick_by_ch[ch] = df

        processed_channels = []
        full_dfs = []

        for ch in idiam_by_ch:
            if ch not in thick_by_ch: continue
            row = channel_data[channel_data['channel'] == ch].iloc[0]
            bm = row['bm_near(mm)']
            ch_length_mm = row['length(mm)']
            PT_idiam_0 = row['inner_diam(mm)']
            PT_thick_0 = row['thick(mm)']
            PT_odiam_0 = PT_idiam_0 + 2 * PT_thick_0
            aface = row['a-face']

            df_d = idiam_by_ch[ch]
            df_t = thick_by_ch[ch]
            interpolated_thick = np.interp(df_d['Axial'], df_t['Axial'], df_t['Mean'])
            df = pd.DataFrame({
                'Axial': df_d['Axial'],
                'MeanDiam': df_d['Mean'],
                'MeanThick': interpolated_thick
            })
            df['Axial_m'] = (df['Axial'] - bm) / 1000.0

            pt_path = f"{raw_data_path}pressure_temperature/canales_BOL/{ch}.pt"
            flux_path = f"{interim_path}flux/pressure_tubes_extended/{ch}.flx"
            pt_df = wr.s3.read_csv(pt_path, sep='\t')
            flux_df = wr.s3.read_csv(flux_path, sep='\t', header=None, names=['coord_rel', 'flux_1mev'])
            
            ch_length_m = ch_length_mm / 1000.0
            ct_length = pt_df['Length(m)'].max()
            dl = ch_length_m - ct_length
            pt_df['Length(m)'] += dl / 2
            flux_df['coord_real'] = flux_df['coord_rel'] * ch_length_m
            
            if aface == 'outlet':
                pt_df['Pressure (Pa)'] = pt_df['Pressure (Pa)'][::-1].values
                pt_df['Temperature (ºC)'] = pt_df['Temperature (ºC)'][::-1].values
                flux_df = flux_df[::-1].reset_index(drop=True)

            P = np.interp(df['Axial_m'], pt_df['Length(m)'], pt_df['Pressure (Pa)'])
            T = np.interp(df['Axial_m'], pt_df['Length(m)'], pt_df['Temperature (ºC)'])
            F = np.interp(df['Axial_m'], flux_df['coord_real'], flux_df['flux_1mev'])

            df['MeanRateDiam'] = (df['MeanDiam'] - PT_idiam_0) / years
            df['MeanRateThick'] = (df['MeanThick'] - PT_thick_0) / years
            df['MeanOutDiam'] = df['MeanDiam'] + 2 * df['MeanThick']
            df['MeanRateOutDiam'] = (df['MeanOutDiam'] - PT_odiam_0) / years
            df['MeanRateElon'] = - (df['MeanRateDiam'] / PT_idiam_0) - (df['MeanRateThick'] / PT_thick_0)
            df['Pressure'] = P
            df['Temperature'] = T
            df['Flux'] = F
            df['Channel'] = ch

            wr.s3.to_csv(df, f"{processed_path}by_channel/{ch}.csv", index=False)
            full_dfs.append(df)
            processed_channels.append(ch)

        if full_dfs:
            full_df = pd.concat(full_dfs, ignore_index=True)
            wr.s3.to_csv(full_df, f"{processed_path}full_dataset.csv", index=False)

        return processed_channels

    # ----------------------------------------------------------------
    # 4. Train / val / test split
    # ----------------------------------------------------------------
    @task
    def split_train_val_test(channel_list):
        import random
        import pandas as pd
        import awswrangler as wr
        random.seed(42)
        n = len(channel_list)
        n_test = int(n * 0.30)
        n_val = int(n * 0.10)
        test_channels = random.sample(channel_list, n_test)
        remaining = [c for c in channel_list if c not in test_channels]
        val_channels = random.sample(remaining, n_val)
        train_channels = [c for c in remaining if c not in val_channels]

        split_info = {"train": train_channels, "val": val_channels, "test": test_channels}
        for name, chs in split_info.items():
            pd.DataFrame(chs, columns=["channel"]).to_csv(f"{PROCESSED_PATH}splits/{name}_channels.csv", index=False)
        return split_info

    # ----------------------------------------------------------------
    # 5. Generate final datasets for ML
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="generate_ml_datasets",
        requirements=["pandas", "awswrangler==3.6.0"],
        system_site_packages=False,
    )
    def generate_ml_datasets(split_info, processed_path: str):
        import pandas as pd
        import awswrangler as wr
        full_df = wr.s3.read_csv(f"{processed_path}full_dataset.csv")
        features = ['Axial_m', 'Pressure', 'Temperature', 'Flux', 'MeanDiam', 'MeanThick', 'MeanOutDiam']
        target = 'MeanRateOutDiam'
        X = full_df[features]
        y = full_df[target]
        channels = full_df['Channel']

        for split_name, ch_list in split_info.items():
            mask = channels.isin(ch_list)
            X_split = X[mask].reset_index(drop=True)
            y_split = pd.DataFrame(y[mask]).reset_index(drop=True)
            wr.s3.to_csv(X_split, f"{processed_path}{split_name}/X_{split_name}.csv", index=False)
            wr.s3.to_csv(y_split, f"{processed_path}{split_name}/y_{split_name}.csv", index=False)
        print("Datasets ML generados: train/val/test")

    # ----------------------------------------------------------------
    # 6. Generate model-compatible data + channel selector
    # ----------------------------------------------------------------
    @task.virtualenv(
        task_id="generate_model_compatible_data",
        requirements=["pandas", "awswrangler==3.6.0", "pyarrow"],
        system_site_packages=False,
    )
    def generate_model_compatible_data(channel_list: list, processed_path: str):
        import pandas as pd
        import awswrangler as wr
        import pickle
        import os

        data_by_ch = {}
        for ch in channel_list:
            df = wr.s3.read_csv(f"{processed_path}by_channel/{ch}.csv")
            data_by_ch[ch] = df

        pickle_path = "/tmp/data_by_ch.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(data_by_ch, f)
        wr.s3.upload(local_file=pickle_path, path=f"{processed_path}model_compatible/data_by_ch.pkl")

        channel_df = pd.DataFrame(sorted(channel_list), columns=["channel"])
        wr.s3.to_csv(channel_df, f"{processed_path}model_compatible/channel_list.csv", index=False)

        os.makedirs("/tmp/parquet_by_channel", exist_ok=True)
        for ch, df in data_by_ch.items():
            df.to_parquet(f"/tmp/parquet_by_channel/{ch}.parquet", index=False)
        wr.s3.upload(local_dir="/tmp/parquet_by_channel",
                     path=f"{processed_path}model_compatible/parquet_by_channel/")

        print(f"Model-compatible data generado para {len(channel_list)} canales")
        return f"{processed_path}model_compatible/data_by_ch.pkl"

    @task
    def save_run_metadata(evaluation_time_efph: float, split_info: dict):
        import json
        from datetime import datetime
        import awswrangler as wr
        metadata = {
            "run_date": datetime.utcnow().isoformat(),
            "evaluation_time_efph": evaluation_time_efph,
            "total_channels": len(split_info["train"]) + len(split_info["val"]) + len(split_info["test"]),
            "n_train": len(split_info["train"]),
            "n_val": len(split_info["val"]),
            "n_test": len(split_info["test"]),
            "dag_version": "optimized_v2_2025",
        }
        path = f"{PROCESSED_PATH}run_metadata/{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        wr.s3.upload(json.dumps(metadata, indent=2), path)

    # ----------------------------------------------------------------
    # Workflow
    # ----------------------------------------------------------------
    upload_task = upload_raw_data_zip(RAW_DATA_PATH)
    evaluation_time = get_evaluation_time(RAW_DATA_PATH)
    flux_task = generate_extended_flux(RAW_DATA_PATH, INTERIM_PATH)
    
    processed_channels = process_and_enrich_all(RAW_DATA_PATH, INTERIM_PATH, PROCESSED_PATH, evaluation_time)
    
    split = split_train_val_test(processed_channels)
    ml_datasets = generate_ml_datasets(split, PROCESSED_PATH)
    notebook_data = generate_model_compatible_data(processed_channels, PROCESSED_PATH)
    metadata = save_run_metadata(evaluation_time, split)

    upload_task >> [flux_task, evaluation_time] >> processed_channels
    flux_task >> processed_channels
    evaluation_time >> processed_channels
    processed_channels >> split >> [ml_datasets, notebook_data] >> metadata

dag = etl_pipeline()
