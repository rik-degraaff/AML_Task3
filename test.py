from src.preprocess.heartbeat_processor import preprocess_heartbeats
import pandas as pd
from src.boilerplate import import_data
X_train, y_train, X_test = import_data()

def test1(out):
    return [1, 2]

def test2(out):
     return [3, 4, 5]


X_train_processed, y_train_processed, X_test_processed = preprocess_heartbeats([test1, test2])(X_train.iloc[:2,:], y_train, X_test.iloc[:2,:])
print(X_train_processed)
print(X_test_processed)