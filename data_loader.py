"""
Data loading, preprocessing, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load dataset, one-hot encode material, normalize inputs and outputs.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
    """
    df = pd.read_csv(csv_path)

    # Separate features and targets
    feature_cols = ['P,kW', 'V,mm/s', 'F,g/min', 'η,%', 'Material']
    target_cols = ['D,%', 'HV', 'Ra,μm', 'CUI']

    X = df[feature_cols]
    y = df[target_cols]

    # One-hot encode material
    material_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    material_encoded = material_encoder.fit_transform(X[['Material']])
    material_df = pd.DataFrame(material_encoded, columns=material_encoder.get_feature_names_out(['Material']))

    # Drop original material column and concatenate
    X = X.drop('Material', axis=1)
    X = pd.concat([X, material_df], axis=1)

    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Further split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    # Normalize inputs and outputs
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_y, material_encoder)