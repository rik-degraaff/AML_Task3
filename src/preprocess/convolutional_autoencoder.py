import torch
from torch import sigmoid
from torch.nn import (
    Conv1d, ConvTranspose1d, Module, Conv2d, MaxPool1d, Flatten, 
    Linear, Sequential, ReLU, SmoothL1Loss
)

import pandas as pd
import numpy as np

class Unflatten(Module):
    def __init__(self, out_shape):
        super(Unflatten, self).__init__()
        self.out_shape = (-1,) + out_shape

    def forward(self, x):
        return x.reshape(self.out_shape)

class ToFloat(Module):
    def __init__(self):
        super(ToFloat, self).__init__()

    def forward(self, x):
        return x.float()

class ConvAutoEncoder(Module):
    def __init__(self, channels, n_features, n_non_conv):
        super(ConvAutoEncoder, self).__init__()

        self.n_non_conv = n_non_conv

        self.conv = Sequential(
            Unflatten((channels, 180)),
            ToFloat(),
            Conv1d(channels, 16, 5, padding=2, padding_mode='circular'), # 180 x  16
            ToFloat(),
            ReLU(),
            ToFloat(),
            MaxPool1d(3, 3),                                             #  60 x  16
            ToFloat(),
            Conv1d(16, 32, 5, padding=2, padding_mode='circular'),       #  60 x  32
            ToFloat(),
            ReLU(),
            ToFloat(),
            MaxPool1d(3, 3),                                             #  20 x  32
            ToFloat(),
            Conv1d(32, 64, 3, padding=1, padding_mode='circular'),       #  20 x  64
            ToFloat(),
            ReLU(),
            ToFloat(),
            MaxPool1d(2, 2),                                             #  10 x  64
            ToFloat(),
            Conv1d(64, 128, 3, padding=1, padding_mode='circular'),      #  10 x 128
            ToFloat(),
            ReLU(),
            ToFloat(),
            MaxPool1d(2, 2),                                             #   5 x 128
            ToFloat(),
            Flatten()                                                    #       640
        )

        self.encode = Linear(640 + n_non_conv, n_features)

        # latent features

        self.decode = Linear(n_features, 640 + n_non_conv)

        self.deconv = Sequential(
            Unflatten((128, 5)),
            ToFloat(),
            ConvTranspose1d(128, 64, 2, stride=2),
            ToFloat(),
            ReLU(),
            ToFloat(),
            ConvTranspose1d(64, 32, 2, stride=2),
            ToFloat(),
            ToFloat(),
            ReLU(),
            ToFloat(),
            ConvTranspose1d(32, 16, 3, stride=3),
            ToFloat(),
            ReLU(),
            ToFloat(),
            ConvTranspose1d(16, channels, 3, stride=3),
            Flatten()
        )

    def forward(self, x):
        x = self.forward_encode(x)
        x = self.forward_decode(x)
        #print('forward:', x.size())
        return x
    
    def forward_encode(self, x):
        non_conv_x, conv_x = torch.tensor_split(x, (self.n_non_conv,), dim=1)
        #print('non conv', non_conv_x.size())
        #print('conv', conv_x.size())
        conv_x = self.conv(conv_x).float()
        x = torch.cat([non_conv_x, conv_x], dim=1).float()
        return self.encode(x).float()
    
    def forward_decode(self, x):
        x = self.decode(x).float()
        non_conv_x , conv_x = torch.tensor_split(x, (self.n_non_conv,), dim=1)
        conv_x = self.deconv(conv_x).float()
        return sigmoid(torch.cat([non_conv_x, conv_x], dim=1)).float()

class ConvAutoEncodePreprocessor:
    def __init__(self, channels, n_features, n_non_conv, n_epochs):
        self.model = ConvAutoEncoder(channels, n_features, n_non_conv)
        self.criterion = SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.n_epochs = n_epochs
    
    def __call__(self, X_train, y_train, X_test):
        X_train_tensor = torch.tensor(X_train.values).float()
        X_test_tensor = torch.tensor(X_test.values).float()
        X_all = torch.cat([X_train_tensor, X_test_tensor])

        print('min:', np.min(X_all.detach().numpy()))
        print('max:', np.max(X_all.detach().numpy()))

        train_loader = torch.utils.data.DataLoader(X_all, batch_size=128, shuffle=True)

        for epoch in range(1, self.n_epochs + 1):
            train_loss = 0.0
            for data in train_loader:
                #print(data)
                #print('data:', data.size())
                data.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.criterion(pred, data)
                train_loss += loss.item() * data.size(0)
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / X_all.size(0)
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')
        
        X_train_latent = self.model.forward_encode(X_train_tensor)
        X_test_latent = self.model.forward_encode(X_test_tensor)

        return pd.DataFrame(X_train_latent.detach().numpy()), y_train, pd.DataFrame(X_test_latent.detach().numpy())
        