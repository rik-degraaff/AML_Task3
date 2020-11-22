import pandas as pd
import pickle
import os

from biosppy.signals import ecg

from ..utils import path_project

def preprocess_heartbeats(feature_extractors):
    def aux(X_train, y_train, X_test):
        train_heartbeats, test_heartbeats = get_heartbeats(X_train, X_test)
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for extractor in feature_extractors:
            train_df = pd.concat([train_df, extract(train_heartbeats, extractor)], axis=1)
            test_df = pd.concat([test_df, extract(test_heartbeats, extractor)], axis=1)
        
        return train_df, y_train, test_df

    return aux

def get_heartbeats(X_train, X_test):
    path = path_project + "data/raw_data/heartbeats/"
    os.makedirs(path, exist_ok=True)
    
    train_path = path + "train.sav"
    if os.path.isfile(train_path) and os.access(train_path, os.R_OK):
        train_heartbeats = pickle.load(open(train_path, "rb"))
    else:
        train_heartbeats = calc_heartbeats(X_train)
        pickle.dump(train_heartbeats, open(train_path, "wb"))
    
    test_path = path + "test.sav"
    if os.path.isfile(test_path) and os.access(test_path, os.R_OK):
        test_heartbeats = pickle.load(open(test_path, "rb"))
    else:
        test_heartbeats = calc_heartbeats(X_test)
        pickle.dump(test_heartbeats, open(test_path, "wb"))
    
    return train_heartbeats, test_heartbeats

def calc_heartbeats(df):
    heartbeats = []
    for index, row in df.iterrows():
        heartbeat = row[row.notna()]
        heartbeats.append(ecg.ecg(heartbeat, sampling_rate=300., show=False))
    return heartbeats

def extract(heartbeats, feature_extractor):
    df = pd.DataFrame()
    for heartbeat in heartbeats:
        features = feature_extractor(heartbeat)
        df = df.append(pd.DataFrame([features]), ignore_index=True)

    return df

