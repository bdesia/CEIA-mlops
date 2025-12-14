# src/utils.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate_on_test(model, loader, target_scaler, device):
    model.eval()
    preds_scaled, targets_scaled = [], []
    preds_orig, targets_orig = [], []

    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds_scaled.extend(out.cpu().numpy())
            targets_scaled.extend(y.cpu().numpy())

            if target_scaler:
                p_orig = target_scaler.inverse_transform(np.array(out.cpu()).reshape(-1, 1)).flatten()
                t_orig = target_scaler.inverse_transform(np.array(y.cpu()).reshape(-1, 1)).flatten()
                preds_orig.extend(p_orig)
                targets_orig.extend(t_orig)

    # Métricas
    mse_scaled = mean_squared_error(targets_scaled, preds_scaled)
    rmse_scaled = np.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(targets_scaled, preds_scaled)
    r2_scaled = r2_score(targets_scaled, preds_scaled)

    mse = mean_squared_error(targets_orig, preds_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, preds_orig)
    r2 = r2_score(targets_orig, preds_orig)

    return {
        "scaled": {"mse": mse_scaled, "rmse": rmse_scaled, "mae": mae_scaled, "r2": r2_scaled},
        "original": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    }

def plot_loss(train_losses, val_losses, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    if save_path:
        plt.savefig(save_path)
    return fig