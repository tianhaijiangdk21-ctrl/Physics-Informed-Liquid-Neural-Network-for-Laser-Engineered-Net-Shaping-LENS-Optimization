"""
Monte Carlo uncertainty quantification.
Adds Gaussian noise to inputs and observes output variability.
"""
import argparse
import torch
import numpy as np
import sys
sys.path.append('.')
from models.lnn import PhysicsInformedLNN
from utils.data_loader import load_and_preprocess

def monte_carlo(lnn_checkpoint, n_iter=1000, noise_std=0.05):
    # Load data to get scalers
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, _ = \
        load_and_preprocess('data/experimental_data.csv')

    # Load model
    input_dim = X_train.shape[1]
    model = PhysicsInformedLNN(input_dim)
    model.load_state_dict(torch.load(lnn_checkpoint))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Use test set as base samples
    X_base = X_test
    y_base = y_test

    # Storage for coefficients of variation
    cv_list = []

    for i in range(len(X_base)):
        x = X_base[i:i+1].copy()
        y_true = y_base[i]
        preds = []
        for _ in range(n_iter):
            # Add Gaussian noise
            x_noisy = x + np.random.normal(0, noise_std, size=x.shape)
            x_tensor = torch.FloatTensor(x_noisy).to(device)
            with torch.no_grad():
                y_pred, _, _ = model(x_tensor)
            y_pred_np = y_pred.cpu().numpy()
            # Inverse transform
            y_orig = scaler_y.inverse_transform(y_pred_np)
            preds.append(y_orig.flatten())
        preds = np.array(preds)
        # Coefficient of variation for each output
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0, ddof=1)
        cv = std_pred / (np.abs(mean_pred) + 1e-6)
        cv_list.append(cv)
        print(f'Sample {i+1}: CVs = {cv}')

    cv_array = np.array(cv_list)
    print(f'Mean CV across samples: {np.mean(cv_array, axis=0)}')
    print(f'Max CV across samples: {np.max(cv_array, axis=0)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to LNN checkpoint')
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--noise_std', type=float, default=0.05)
    args = parser.parse_args()
    monte_carlo(args.model, args.n_iter, args.noise_std)