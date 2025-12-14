# src/dataset.py
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class DiameterDataset(Dataset):
    def __init__(
        self,
        data_dict,
        ch_list,
        window_size,
        min_context,
        max_context,
        feature_scaler=None,
        target_scaler=None,
        fit_scaler=False,
        target="center",
        data_stride=1,
        seed=42
    ):
        if not (window_size <= min_context <= max_context):
            raise ValueError("window_size <= min_context <= max_context")
        for val, name in [(window_size, "window_size"), (min_context, "min_context"), (max_context, "max_context")]:
            if val % 2 == 0 or val < 3:
                raise ValueError(f"{name} debe ser impar y >= 3")

        self.data_dict = data_dict
        self.ch_list = ch_list
        self.window_size = window_size
        self.min_context = min_context
        self.max_context = max_context
        self.data_stride = data_stride
        self.seed = seed
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        self.feature_cols = ['Axial_m', 'Temperature', 'Flux']
        self.target_col = 'MeanRateOutDiam'

        self.feats_padded_list = []
        self.targets_list = []
        self.sample_indices = []
        self.lengths = []

        max_half = max_context // 2

        if fit_scaler:
            all_feats = []
            all_targs = []
            for ch in ch_list:
                df = data_dict[ch]
                feats = df[self.feature_cols].values.astype(np.float32)
                targs = df[self.target_col].values.astype(np.float32)
                all_feats.append(feats)
                all_targs.append(targs)
            all_feats = np.vstack(all_feats)
            all_targs = np.hstack(all_targs).reshape(-1, 1)
            self.feature_scaler = MinMaxScaler().fit(all_feats)
            self.target_scaler = MinMaxScaler().fit(all_targs)

        for ch_idx, ch in enumerate(ch_list):
            df = data_dict[ch]
            feats = df[self.feature_cols].values.astype(np.float32)
            targs = df[self.target_col].values.astype(np.float32)

            if self.feature_scaler:
                feats = self.feature_scaler.transform(feats)
            if self.target_scaler:
                targs = self.target_scaler.transform(targs.reshape(-1, 1)).flatten()

            feats_padded_np = np.pad(feats, ((max_half, max_half), (0, 0)), mode='edge')
            feats_padded = torch.from_numpy(feats_padded_np)
            targs_tensor = torch.from_numpy(targs)

            self.feats_padded_list.append(feats_padded)
            self.targets_list.append(targs_tensor)
            self.lengths.append(len(df))

            L = len(df)
            for i in range(0, L, data_stride):
                self.sample_indices.append((ch_idx, i))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        ch_idx, center_idx = self.sample_indices[idx]
        feats_padded = self.feats_padded_list[ch_idx]
        targets = self.targets_list[ch_idx]

        rng = random.Random(self.seed + idx)
        context_size = rng.randint(self.min_context, self.max_context)
        if context_size % 2 == 0:
            context_size -= 1
        half = context_size // 2

        offset = self.max_context // 2
        center_padded = center_idx + offset
        start = center_padded - half
        end = center_padded + half + 1

        window_real = feats_padded[start:end]

        if context_size > self.window_size:
            step = context_size / self.window_size
            ds_idx = torch.round(torch.arange(0, context_size, step)).long()[:self.window_size]
            window_final = window_real[ds_idx]
        else:
            window_final = window_real

        axial_vals = window_final[:, 0]
        axial_center = axial_vals[self.window_size // 2]
        delta_left = axial_center - axial_vals[0] if self.window_size > 1 else 0
        delta_right = axial_vals[-1] - axial_center if self.window_size > 1 else 0

        sample_info = (
            self.ch_list[ch_idx],
            center_idx,
            context_size,
            (axial_vals[0].item(), axial_vals[-1].item()),
            float(delta_left),
            float(delta_right)
        )

        return window_final, targets[center_idx], sample_info

def custom_collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    sample_info = [item[2] for item in batch]
    return features, targets, sample_info