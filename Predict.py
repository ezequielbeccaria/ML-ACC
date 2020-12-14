import sys
import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import motec_preprocess as motec
from Training import LapPredictor, init_hidden

def predict(input, predictor, hidden):
    predictor.eval()

    with torch.no_grad():
        return predictor(input, hidden)


def load_predictor(path):
    hidden_dim = 1024
    layers = 1
    input_dim = 26
    seq_len = 10

    predictor = LapPredictor(input_dim, hidden_dim, layers, seq_len)
    predictor.to('cpu')

    # If model is stored, continue the training
    if os.path.isfile(path + 'model.pt'):
        predictor.load_state_dict(torch.load(path + 'model.pt')['model_state_dict'])
        print("Pre-trained model loaded")
    else:
        print("No pre-trained model found")

    return predictor


def load_scaler(path):
    scaler = StandardScaler()
    scaler_params = np.load(path + "scaler.npy", allow_pickle=True)
    scaler.mean_ = scaler_params[0]
    scaler.scale_ = scaler_params[1]
    return scaler


if __name__ == '__main__':
    run_id = '11'
    # run_path = '/home/ezequiel/experiments/ML-ACC/' + run_id + '/'
    run_path = sys.argv[1]

    scaler = load_scaler(run_path)

    print("Processing Motec CSVs:")
    dfs, _ = motec.read_all_CSV(sys.argv[2])
    # dfs, _ = motec.read_all_CSV('./motec_files/')

    dfs_scaled = [pd.DataFrame(scaler.transform(df.values), index=df.index, columns=df.columns) for df in dfs]

    device = 'cpu'

    predictor = load_predictor(run_path)

    for df in dfs_scaled:
        hidden = init_hidden(1, 1024, device)

        input = torch.tensor(df.iloc[0:10].values, dtype=torch.float)
        input = torch.reshape(input, (1, input.shape[0], input.shape[1]))
        input.to(device)

        out1, hidden = predict(input, predictor, hidden)
        input = torch.reshape(out1, (1, out1.shape[0], out1.shape[1]))
        out2, hidden = predict(input, predictor, hidden)

        target = scaler.inverse_transform(df.iloc[0:30].values)
        plt.plot(target[:, -1], label='original')

        out = torch.cat((out1, out2), dim=0)
        out_np = out.cpu().numpy()
        out_np = scaler.inverse_transform(out_np)
        out_np = np.concatenate((target[0:10, :], out_np), axis=0)
        # out_np = np.concatenate((target[0:10, :], out_np[10:, :]), axis=0)
        plt.plot(out_np[:, -1], label='prediction')

        plt.legend()
        plt.show()