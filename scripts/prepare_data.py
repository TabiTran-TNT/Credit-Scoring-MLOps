import pandas as pd
import numpy as np
from credit_scoring.features import prepare_features

def load_and_prepare_data(file_path='data/application_train.csv'):
    df = pd.read_csv(file_path)
    X = prepare_features(df)
    y = df['TARGET'].copy()
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_prepare_data()
    print(f"Features: {X.shape}, Target: {y.shape}")
    print(f"Default rate: {y.mean():.2%}")