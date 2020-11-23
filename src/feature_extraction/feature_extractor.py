import pandas as pd

def extract_features(feature_extractors):
    def aux(X_train, y_train, X_test):
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for extract in feature_extractors:
            train, test = extract(X_train, X_test)
            train_df = pd.concat([train_df, train], axis=1)
            test_df = pd.concat([test_df, test], axis=1)
        
        return train_df, y_train, test_df

    return aux