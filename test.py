import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.feature_extraction.heartbeat_feature_extractors import extract_all_features
from src.boilerplate import import_data

X_train, y_train, X_test = import_data()

def test1(out):
    return [1, 2]

def test2(out):
     return [3, 4, 5]

def fft(heartbeat):
    start, end = heartbeat.rpeaks[0], heartbeat.rpeaks[-1]
    length = end - start
    sp = np.fft.rfft(heartbeat.filtered[start:end])
    freq = np.fft.rfftfreq(heartbeat.ts[start:end].shape[-1])
    print("real")
    print(sp.real)
    print(len(sp.real))
    print("imag")
    print(sp.imag)
    print(len(sp.imag))
    plt.plot(freq, sp.real/length, freq, sp.imag/length)
    plt.show()

X_train_processed, y_train_processed, X_test_processed = extract_all_features([test1, test2, fft])(X_train, y_train, X_test)

print(X_train_processed)
print()
print(X_test_processed)