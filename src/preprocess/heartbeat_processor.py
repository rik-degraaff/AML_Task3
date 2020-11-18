import pandas as pd
from biosppy.signals import ecg

def preprocess_heartbeats(feature_extractors):
    def aux(X_train, y_train, X_test):
        feature_extractor = extract(feature_extractors)
        return X_train.apply(feature_extractor), y_train, X_test.apply(feature_extractor)

    return aux

def extract(feature_extractors):
    def aux(raw_heartbeat):
        out = ecg.ecg(raw_heartbeat, sampling_rate=1000., show=False)
        res = []
        for extractor in feature_extractors:
            res += extractor(out)
        return res

    return aux