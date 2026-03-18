"""
Run ablation study comparing LNN variants with/without physics constraints.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('.')
from models.lnn import PhysicsInformedLNN
from utils.data_loader import load_and_preprocess
from utils.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score, mean_absolute_error

def train_variant(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss()
    best_val_loss = float('inf')
    patience = config.get('patience', 20)
    counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            y_pred, T_pred, sigma_pred = model(bx)
            loss_data = mse(y_pred, by)
            # For variants without physics, set physics loss to zero
            if hasattr(model, 'physics_loss') and config['use_physics']:
                loss_physics = model.physics_loss(T_pred, sigma_pred, bx)
            else:
                loss_physics = 0.0
            loss_total = loss_data + loss_physics
            loss_total.backward()
            optimizer.step()
            train_loss += loss_total.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                y_pred, T_pred, sigma_pred = model(bx)
                loss_data = mse(y_pred, by)
                if hasattr(model, 'physics_loss') and config['use_physics']:
                    loss_physics = model.physics_loss(T_pred, sigma_pred, bx)
                else:
                    loss_physics = 0.0
                val_loss += (loss_data + loss_physics).item() * len(bx)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    return best_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/lnn_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, _ = \
        load_and_preprocess(config['data_path'])

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    variants = {
        'LNN-base': {'lambda1': 0.0, 'lambda2': 0.0},
        'LNN-Fourier': {'lambda1': 0.1, 'lambda2': 0.0},
        'LNN-thermo': {'lambda1': 0.0, 'lambda2': 0.1},
        'LNN-full': {'lambda1': 0.1, 'lambda2': 0.1}
    }

    results = {}
    for name, params in variants.items():
        print(f'\n--- Training {name} ---')
        model = PhysicsInformedLNN(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            dt=config['dt'],
            lambda1=params['lambda1'],
            lambda2=params['lambda2']
        ).to(device)
        val_loss = train_variant(model, train_loader, val_loader, config, device)
        results[name] = val_loss

        # Evaluate on test set
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)
        model.eval()
        with torch.no_grad():
            y_pred, _, _ = model(X_test_t)
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        y_pred_orig = scaler_y.inverse_transform(y_pred_np)
        y_test_orig = scaler_y.inverse_transform(y_test_np)

        r2_scores = []
        for i in range(4):
            r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
            r2_scores.append(r2)
        results[name+'_R2'] = r2_scores
        print(f'{name} test R2: {r2_scores}')

    print('\n=== Ablation Study Summary ===')
    for name in variants.keys():
        print(f'{name}: val_loss = {results[name]:.6f}, R2 = {results[name+"_R2"]}')

if __name__ == '__main__':
    main()