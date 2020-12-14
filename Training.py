import sys
import os.path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import motec_preprocess as motec


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class MotecDataSet(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]


class LapPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, layers, seq_len) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.hidden_to_outputs = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, h):
        x, h = self.lstm(x, h)
        x = x.reshape(x.shape[1], self.hidden_dim)
        x = self.hidden_to_outputs(x)
        return x, h


def init_hidden(layers, hidden_dim, device):
    hidden_state = torch.zeros(layers, 1, hidden_dim).to(device)
    cell_state = torch.zeros(layers, 1, hidden_dim).to(device)
    hidden = (hidden_state, cell_state)
    return hidden


def generate_training_test_sets(dfs):
    inputs = []

    for df in dfs:
        inputs.append(torch.tensor(df.values, dtype=torch.float))

    # slip training-test
    train, test = train_test_split(inputs, test_size=0.1)

    training_set = MotecDataSet(train)
    test_set = MotecDataSet(test)

    return training_set, test_set


def scale_data(dfs):
    # Standarize dataframe
    scaler = StandardScaler()
    for df in dfs:
        scaler.partial_fit(df.values)

    np.save(run_path+"scaler.npy", np.array([scaler.mean_, scaler.scale_]))
    print("Standar Scaler's parameters saved")

    return [pd.DataFrame(scaler.transform(df.values), index=df.index, columns=df.columns) for df in dfs]


def train(train_dataset, test_dataset, epochs, optimizer, seq_len):
    dataset_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    iteration = 0
    for e in range(epochs):
        train_loss = []
        for i, tensor in enumerate(dataset_loader):
            hidden = init_hidden(layers, hidden_dim, device)
            seq_len = 10
            input, target = tensor[0, 0:seq_len], tensor[0, :30]
            input = torch.reshape(input, (1, input.size()[0], input.size()[1]))

            optimizer.zero_grad()
            out1, hidden = predictor(input.to(device), hidden)
            input2 = torch.reshape(out1.detach(), (1, out1.shape[0], out1.shape[1]))
            out2, hidden = predictor(input2.to(device), hidden)
            out = torch.cat((out1, out2), dim=0)
            target_loss = target[10:30].squeeze().to(device)
            loss = loss_fn(out[:target_loss.size()[0]], target_loss)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            writer.add_scalar('Training/Iteration Loss', float(loss.item()), iteration)
            iteration += 1

        print('Epoch: {} - Training Loss Mean: {}'.format(e, np.mean(train_loss)))
        writer.add_scalar('Training/Epoch Mean Loss', float(np.mean(train_loss)), e)

        # Test predictor
        test(test_dataset, e, predictor)

        # Store model every 100 epochs
        if e % 10 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': predictor.state_dict()
            }, run_path + '/model.pt')


def test(dataset, epoch, predictor):
    predictor.eval()

    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    test_loss = []
    for i, tensor in enumerate(dataset_loader):
        with torch.no_grad():
            hidden = init_hidden(layers, hidden_dim, device)
            seq_len = tensor.size()[1]
            input, target = tensor[0, 0:seq_len], tensor[0, 0: seq_len + 1]
            input = torch.reshape(input, (1, input.size()[0], input.size()[1]))
            out, hidden = predictor(input.to(device), hidden)
            loss = loss_fn(out, target.squeeze().to(device))
            test_loss.append(loss.item())

    print('Epoch: {} - Test Loss Mean: {}'.format(epoch, np.mean(test_loss)))
    writer.add_scalar('Testing/Epoch Mean Loss', float(np.mean(test_loss)), epoch)
    predictor.train()


if __name__ == '__main__':
    # run_id = '12'
    # run_path = '/home/ezequiel/experiments/ML-ACC/'+run_id+'/'
    run_path = sys.argv[1]
    print('Working path: {}'.format(run_path))
    if not os.path.isdir(run_path):
        os.mkdir(run_path)

    print("Processing Motec CSVs:")
    # dfs, _ = motec.read_all_CSV('./motec_files/')
    print('Dataset path: {}'.format(sys.argv[2]))
    dfs, _ = motec.read_all_CSV(sys.argv[2])
    dfs = scale_data(dfs)

    writer = SummaryWriter(run_path, flush_secs=30)

    hidden_dim = 1024
    layers = 1
    input_dim = dfs[0].shape[1]
    epochs = 10000
    seq_len = 10

    device = get_device()
    predictor = LapPredictor(input_dim, hidden_dim, layers, seq_len)

    # If model is stored, continue the training
    if os.path.isfile(run_path+'model.pt'):
        predictor.load_state_dict(torch.load(run_path + '/model.pt')['model_state_dict'])
        print("Pre-trained model loaded")
    else:
        print("No pre-trained model found")

    predictor.train()
    predictor.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-6)

    training_dataset, testing_dataset = generate_training_test_sets(dfs)

    train(training_dataset, testing_dataset, epochs, optimizer, seq_len)
    print("Training finished")
