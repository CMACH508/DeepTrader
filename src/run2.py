# import numpy as np
# from model.ASU import ASU, LiteTCN
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# stocks_data = np.load('/Users/pujanmaharjan/uni adelaide/research project/DeepTrader/src/data/DJIA/stocks_data.npy')

# # num_records (time_ids) / batch, window_length, stocks, features
# print('stock_data dimension ', stocks_data.shape)

# print('torch version ', torch.__version__)

# class CustomStockDataset(Dataset):
#     def __init__(self, stocks_data):
#         self.stocks_data = stocks_data

#     def __len__(self):
#         return len(self.stocks_data)

#     def __getitem__(self, idx):
#         # print('stocks data shape ', self.stocks_data.shape)
#         row_data = self.stocks_data[idx]
#         # print('row_data shape ', row_data.shape)
#         # print('row_data', row_data)
#         features = row_data[0][:,1:]
#         # print('features ', features)
#         targets = row_data[0][:,0]
#         # print('targets ', targets)
#         return features, targets


# stock_dataset = CustomStockDataset(stocks_data)
# stock_dataloader = DataLoader(stock_dataset, batch_size=2, shuffle=False)

# # num_records (time_ids), window_length (batch), stocks, features
# num_stocks = stocks_data.shape[2]
# num_features = stocks_data.shape[3]
# stocks_data_torch = torch.from_numpy(stocks_data).float()

# print('stocks_data_torch ', stocks_data_torch.shape)
# # print(' 957 time_ids, window length = 4, stocks = 2, features = 8, shape is 9 because of 1 target value')

# tcn = LiteTCN(in_features=8,
#               hidden_size=10,
#               num_layers=1,
#               output_size=num_stocks*2).float()

# all_output = []
# counter = 0
# for x, y in stock_dataloader:
#     counter += 1
#     print('shape of x ', x.shape)
#     print('batch / time_ids, stocks, features')
#     if len(x) == 1:
#         print('x has single row')
#         continue
#     p = tcn.forward(x.float())
#     print('tcn output ', p.shape)
#     p_numpy = p.detach().numpy()
#     for i in range(len(p_numpy)):
#         first_dim = p_numpy[i]
#         # print('first_dim ', first_dim)
#         for j in range(len(first_dim)):
#             second_dim = first_dim[j]
#             # print('second_dim ', second_dim)
#     # for i in range(p_numpy.shape[0]):
#     #     all_output.append(p_numpy[i])

# all_output_df = pd.DataFrame(all_output)
# all_output_df.to_csv('/Users/pujanmaharjan/uni adelaide/research project/Informer/dataset/stock_data_tcn.csv', index=False)

# print('counter ', counter)
# print('finished')