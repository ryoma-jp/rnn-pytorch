
import os
import argparse
import numpy as np
import torch
from dataloader import DataLoader_Statsmodels_CO2
from rnn import train as rnn_train
from rnn import predict as rnn_predict
from rnn import evaluate as rnn_evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Impremenation of RNN for Time Series Analysis")

    # --- Output directory path ---
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory path')

    return parser.parse_args()

def main():
    """
    Main function.
    """
    
    # --- Fix random seed ---
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Parse command line arguments ---
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Atomosphic CO2 Data ---
    dataloader_ = DataLoader_Statsmodels_CO2()
    train_data, test_data = dataloader_.split_data()
    X_train, y_train = dataloader_.preprocessing(train_data)
    X_test, y_test = dataloader_.preprocessing(test_data)

    # --- Training RNN ---
    model = rnn_train(X_train, y_train, save_path=args.output_dir)
    train_pred = rnn_predict(model, X_train)
    train_mse = rnn_evaluate(y_train, train_pred)
    print(f'Train MSE: {train_mse}')

    # --- Test RNN ---
    test_pred = rnn_predict(model, X_test)
    test_mse = rnn_evaluate(y_test, test_pred)
    print(f'Test MSE: {test_mse}')

    # --- Visualize the results ---
    df_train_results = pd.DataFrame({'original data(interpolated)': dataloader_.inverse_transform(dataloader_.df_co2.values)}, index=dataloader_.df_co2.index)
    df_train_results['train_pred'] = pd.DataFrame(dataloader_.inverse_transform(train_pred).reshape(-1), index=train_data.index[-len(train_pred):])
    sns.lineplot(data=df_train_results)
    plt.savefig(os.path.join(args.output_dir, 'train_results.png'))
    plt.close()

    df_test_results = pd.DataFrame({'original data(interpolated)': dataloader_.inverse_transform(dataloader_.df_co2.values)}, index=dataloader_.df_co2.index)
    df_test_results['test_pred'] = pd.DataFrame(dataloader_.inverse_transform(test_pred).reshape(-1), index=test_data.index[-len(test_pred):])
    sns.lineplot(data=df_test_results)
    plt.savefig(os.path.join(args.output_dir, 'test_results.png'))
    plt.close()

if __name__ == "__main__":
    main()
