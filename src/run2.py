import numpy as np
from model.ASU import ASU, LiteTCN
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

stocks_data = np.load('/Users/pujanmaharjan/uni adelaide/research project/DeepTrader/src/data/DJIA/stocks_data.npy')

print('torch version ', torch.__version__)

class CustomStockDataset(Dataset):
    def __init__(self, stocks_data):
        self.stocks_data = stocks_data

    def __len__(self):
        return len(self.stocks_data)

    def __getitem__(self, idx):
        # print('stocks data shape ', self.stocks_data.shape)
        row_data = self.stocks_data[idx]
        # print('row_data shape ', row_data.shape)
        # print('row_data', row_data)
        features = row_data[0][:,1:]
        # print('features ', features)
        targets = row_data[0][:,0]
        # print('targets ', targets)
        return features, targets


stock_dataset = CustomStockDataset(stocks_data)
stock_dataloader = DataLoader(stock_dataset, batch_size=3, shuffle=False)

# num_records (time_ids), window_length (batch), stocks, features
num_stocks = stocks_data.shape[2]
num_features = stocks_data.shape[3]
# asu = ASU(num_nodes=num_stocks,
#                   in_features=num_features, # -1 for the target
#                   hidden_dim=128,
#                   window_len=13)
# asu = asu.float()
stocks_data_torch = torch.from_numpy(stocks_data).float()

tcn = LiteTCN(in_features=8,
              hidden_size=10,
              num_layers=1).float()

for x,y in stock_dataloader:
    if len(x) == 1:
        print('x has single row')
        continue
    p = tcn.forward(x.float())
    print('p ', p)

print('finished')