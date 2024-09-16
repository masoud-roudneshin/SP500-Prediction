import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web

class StockDataset(Dataset):
    def __init__(self, data, sequence_length=60):
      """
      Args:
          file_path (str): Path to the CSV file containing stock data.
          sequence_length (int): Number of days to use as input sequence.
          transform (callable, optional): Optional transform to be applied on a sample.
      """

      # Normalize the daily returns

      self.scaler = MinMaxScaler(feature_range=(0, 1))

      sp500_scaled = self.scaler.fit_transform(data.values.reshape(-1,1))

      # Prepare sequences and labels
      self.sequence_length = sequence_length
      self.data, self.labels = self.create_univariate_rnn_data(sp500_scaled)

    def create_univariate_rnn_data(self, data):
      x_data, y_data = [], []
      for i in range(len(data) - self.sequence_length):

        x = data[i:i + self.sequence_length, 0][::-1]
        y = data[i + self.sequence_length, 0]

        x_data.append(x)
        y_data.append(y)

      return np.array(x_data), np.array(y_data)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):

      return self.data[idx],  self.labels[idx]

    def inverse_transform(self, data):
      """Inverse transforms normalized data back to original scale."""
      return self.scaler.inverse_transform(data)

# Function to create DataLoader
def create_dataloader( data, batch_size=16, sequence_length=60, shuffle=False):
    dataset = StockDataset(data, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last= False)
    return dataloader, dataset
