from sklearn.decomposition import PCA
import numpy as np

def pca(n_components):
    def preprocess(x_train, y_train, x_test):
        pca = PCA()
        temp = np.vstack((x_train, x_test))
        pca.fit(temp)
        return pca.transform(x_train)[:, :n_components], y_train, pca.transform(x_test)[:, :n_components]
    
    return preprocess
