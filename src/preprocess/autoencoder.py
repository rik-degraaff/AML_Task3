import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

def autoencoder(n_latent):
    def preprocess(x_train, y_train, x_test):
        n_encoder = (x_train.shape[0] + n_latent) // 2
        reg = MLPRegressor(hidden_layer_sizes = (n_encoder, n_latent, n_encoder), 
            activation = 'tanh', 
            solver = 'adam', 
            learning_rate_init = 0.0001, 
            max_iter = 200, 
            tol = 0.0000001, 
            verbose = True)
        
        smote = SMOTE(random_state=0)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        temp = np.vstack((x_train_resampled, x_test))
        reg.fit(temp, temp)

        def encode(data):
            data = np.asmatrix(data)
            
            encoder = data*reg.coefs_[0] + reg.intercepts_[0]
            encoder = (np.exp(encoder) - np.exp(-encoder))/(np.exp(encoder) + np.exp(-encoder))
            
            latent = encoder*reg.coefs_[1] + reg.intercepts_[1]
            latent = (np.exp(latent) - np.exp(-latent))/(np.exp(latent) + np.exp(-latent))
    
            return np.asarray(latent)

        return encode(x_train), y_train, encode(x_test)
    
    return preprocess