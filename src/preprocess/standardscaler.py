from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_standardscaler(X_train, y_train, X_test):
    scaler = StandardScaler()
    scaler.fit(pd.concat([X_train, X_test], axis=0))
    X_train_feature = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_feature = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    return X_train_feature, y_train, X_test_feature