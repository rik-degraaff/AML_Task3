import os

from src.boilerplate import main

import src.feature_extraction.functions_for_features as f
from src.feature_extraction.heartbeat_feature_extractors import extract_all_features
from src.preprocess.convolutional_autoencoder import ConvAutoEncodePreprocessor
from src.model.kernel_svm import give_model

model = give_model()

def preprocess_ecg_data_convolutional_autoencoder(X_train, y_train, X_test):
    feature_extractors = [
        f.heart_rate_derived_from_peaks,
        f.extract_heart_rate_variability_peaks
    ]
    ts_feature_extractors = [
        f.extract_template_stats_median,
        f.extract_template_stats_std,
        f.extract_template_stats_skew,
        f.extract_template_stats_kurtosis
    ]
    extractor = extract_all_features(feature_extractors, ts_feature_extractors)
    X_train, y_train, X_test = extractor(X_train, y_train, X_test)
    print(X_train)
    print(X_test)
    autoencode = ConvAutoEncodePreprocessor(4, 40, len(X_train.columns) - 4*180, 250)
    return autoencode(X_train, y_train, X_test)

main(name=os.path.basename(__file__), model=model, preprocess=preprocess_ecg_data_convolutional_autoencoder, get_score=True, save_model=True)