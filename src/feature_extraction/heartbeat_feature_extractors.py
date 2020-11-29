import pandas as pd
import numpy as np
import pickle
import os
import sys

from sklearn.preprocessing import MinMaxScaler

from itertools import cycle

from biosppy.signals import ecg

from ..utils import path_project
from .feature_extractor import extract_features

class Heartbeat:
    def __init__(self, out):
        self.ts = out['ts']
        self.filtered = out['filtered']
        self.rpeaks = out['rpeaks']
        self.templates_ts = out['templates_ts']
        self.templates = out['templates']
        self.heart_rate_ts = out['heart_rate_ts']
        self.heart_rate = out['heart_rate']

def extract_all_features(feature_extractors, ts_feature_extractors):
    extract_heartbeats = extract_heartbeat_features(feature_extractors, ts_feature_extractors)
    return extract_features([extract_heartbeats])

def min_max_scale(train_df, test_df, columns=True):
    all_features = pd.concat([train_df, train_df], axis=0)
    if columns:
        scaler = MinMaxScaler()
        scaler.fit(all_features)
        return pd.DataFrame(scaler.transform(train_df)), pd.DataFrame(scaler.transform(test_df))
    
    min_val = all_features.min()
    max_val = all_features.max()
    return (train_df - min_val)/(max_val - min_val), (test_df - min_val)/(max_val - min_val)

def extract_heartbeat_features(feature_extractors, ts_feature_extractors):
    def aux(X_train, X_test):
        train_heartbeats, train_df, test_heartbeats, test_df = get_heartbeats(X_train, X_test)

        train_df, test_df = min_max_scale(train_df, test_df)

        for extractor in feature_extractors:
            train_features, test_features = min_max_scale(extract(train_heartbeats, extractor), extract(test_heartbeats, extractor))

            train_df = pd.concat([train_df, train_features], axis=1, ignore_index=True)
            test_df = pd.concat([test_df, test_features], axis=1, ignore_index=True)

        for extractor in ts_feature_extractors:
            train_features, test_features = min_max_scale(extract(train_heartbeats, extractor), extract(test_heartbeats, extractor), columns=False)

            train_df = pd.concat([train_df, train_features], axis=1, ignore_index=True)
            test_df = pd.concat([test_df, test_features], axis=1, ignore_index=True)
        
        return train_df, test_df

    return aux

def get_heartbeats(X_train, X_test):
    path = path_project + "data/ecg_data/"
    os.makedirs(path, exist_ok=True)

    train_path = path + "train.sav"
    if os.path.isfile(train_path) and os.access(train_path, os.R_OK):
        train_heartbeats, train_bounds = pickle.load(open(train_path, "rb"))
    else:
        train_heartbeats, train_bounds = calc_heartbeats(X_train)
        pickle.dump((train_heartbeats, train_bounds), open(train_path, "wb"))
    
    test_path = path + "test.sav"
    if os.path.isfile(test_path) and os.access(test_path, os.R_OK):
        test_heartbeats, test_bounds = pickle.load(open(test_path, "rb"))
    else:
        test_heartbeats, test_bounds = calc_heartbeats(X_test)
        pickle.dump((test_heartbeats, test_bounds), open(test_path, "wb"))
    
    return train_heartbeats, train_bounds, test_heartbeats, test_bounds

def calc_heartbeats(df):
    heartbeats = []
    bounds = pd.DataFrame()
    for _, row in df.iterrows():
        heartbeat = row[row.notna()]
        bounds = bounds.append(pd.DataFrame([[np.min(heartbeat), np.max(heartbeat)]]), ignore_index=True)
        # heartbeat = (heartbeat - np.min(heartbeat))/(np.max(heartbeat) - np.min(heartbeat))
        heartbeats.append(Heartbeat(ecg.ecg(heartbeat, sampling_rate=300., show=False)))
    return heartbeats, bounds

def extract(heartbeats, feature_extractor):
    df = pd.DataFrame()
    for heartbeat in heartbeats:
        features = feature_extractor(heartbeat)
        df = pd.concat([df, pd.DataFrame([features])], axis=0, ignore_index=True)
    print(df)
    return df

