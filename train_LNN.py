"""
Train the Physics-Informed LNN model.
Usage: python scripts/train_lnn.py --config config/lnn_config.yaml
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.append('.')
from models.lnn import PhysicsInformedLNN
from utils.data_loader import load_and_preprocess
from utils.metrics import mean_absolute_percentage_error

def train(config):
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, _ = \
        load_and_preprocess(config['data_path'])

    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Model
    input_dim = X_train.shape[1]  # 4 continuous + 3 one-hot = 7
    model = PhysicsInformedLNN(input_dim=input_dim,
                               hidden_dim=config['hidden_dim'],
                               dt=config['dt'],
                               lambda1=config['lambda1'],
                               lambda2=config['lambda2'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    patience = config.get('patience', 20)
    counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            y_pred, T_pred, sigma_pred = model(batch_X)
            # Data loss
            loss_data = mse_loss(y_pred, batch_y)
            # Physics loss (thermal stress not used here, set to None)
            loss_physics = model.physics_loss(T_pred, sigma_pred, batch_X)
            loss_total = loss_data + loss_physics
            loss_total.backward()
            optimizer.step()
            train_loss += loss_total.item() * len(batch_X)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                y_pred, T_pred, sigma_pred = model(batch_X)
                loss_data = mse_loss(y_pred, batch_y)
                loss_physics = model.physics_loss(T_pred, sigma_pred, batch_X)
                loss_total = loss_data + loss_physics
                val_loss += loss_total.item() * len(batch_X)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}: train loss {train_loss:.6f}, val loss {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['save_path'])
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping.')
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    with torch.no_grad():
        y_pred, _, _ = model(X_test_t)
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()
        # Inverse transform to original scale
        y_pred_orig = scaler_y.inverse_transform(y_pred_np)
        y_test_orig = scaler_y.inverse_transform(y_test_np)

    # Compute metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    for i, name in enumerate(['Dilution', 'Hardness', 'Roughness', 'CUI']):
        r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
        mae = mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i])
        mape = mean_absolute_percentage_error(y_test_orig[:, i], y_pred_orig[:, i])
        print(f'{name}: R²={r2:.3f}, MAE={mae:.3f}, MAPE={mape:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/lnn_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)