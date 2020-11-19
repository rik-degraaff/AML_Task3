import os
import pandas as pd
import biosppy.signals.ecg as ecg
from utils import path_project
from boilerplate import import_data


X_train, y_train, X_test = import_data(None)

#print(X_train.shape)
#print("----")
#print(y_train.shape)
#print("----")
#print(X_test.shape)

#print(X_train.head())
#print((X_train.iloc[3029,:]))

print("test")
ts=X_train.iloc[4448,:]
print(ts.shape)
print(ts.index)
ts_new=ts[ts.notna()]
print(ts_new.shape)
print(ts_new.index)


ts,f_signal,rpeaks,templates_ts,templates,heart_rate_ts,heart_rate=ecg.ecg(signal=ts_new, sampling_rate=300.0, show=True)
print("ts")
print(ts)
print("----")
print(ts.shape)
print("----")
print(ts.size)
print("----")
print(ts.dtype)
print("----")
print("----")
print("----")

print("f_signal")
print(f_signal)
print("----")
print(f_signal.shape)
print("----")
print(f_signal.size)
print("----")
print(f_signal.dtype)
print("----")
print("----")
print("----")

print("rpeaks")
print(rpeaks)
print("----")
print(rpeaks.shape)
print("----")
print(rpeaks.size)
print("----")
print(rpeaks.dtype)
print("----")
print("----")
print("----")

print("templates_ts")
print(templates_ts)
print("----")
print(templates_ts.shape)
print("----")
print(templates_ts.size)
print("----")
print(templates_ts.dtype)
print("----")
print("----")
print("----")

print("templates")
print(templates)
print("----")
print(templates.shape)
print("----")
print(templates.size)
print("----")
print(templates.dtype)
print("----")
print("----")
print("----")

print("heart_rate_ts")
print(heart_rate_ts)
print("----")
print(heart_rate_ts.shape)
print("----")
print(heart_rate_ts.size)
print("----")
print(heart_rate_ts.dtype)
print("----")
print("----")
print("----")

print("heart_rate")
print(heart_rate)
print("----")
print(heart_rate.shape)
print("----")
print(heart_rate.size)
print("----")
print(heart_rate.dtype)
print("----")
print("----")
print("----")


