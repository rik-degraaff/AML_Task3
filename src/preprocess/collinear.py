import numpy as np
import pandas as pd

def feature_select_collinear(X_train, y_train, X_test, threshold=1000):
    
    def conditional_number(corr):
        eigValues, eigVectors = np.linalg.eig(corr)
        return abs(max(eigValues)/min(eigValues))
    
    X_concat = pd.concat([X_train, X_test])
    cols = X_concat.columns
    
    corr = X_concat.corr()
    corr_abs_unstack = corr.abs().unstack()
    
    pairs_to_drop = set()
    for i in range(0, X_concat.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    corr_abs_unstack = corr_abs_unstack.drop(labels=pairs_to_drop)
    
    cond_num = conditional_number(corr)
    print(f"Conditional Number {cond_num}")
    
    removed_features = []
    while (cond_num>threshold and X_concat.shape[1]>1):
        
        idxmax_corr = corr_abs_unstack.idxmax()
        
        removed_feature = idxmax_corr[0]
        corr_abs_unstack = corr_abs_unstack.drop(labels=removed_feature, level=0)
        corr_abs_unstack = corr_abs_unstack.drop(labels=removed_feature, level=1)
        removed_features.append(removed_feature)
        X_concat = X_concat.drop(removed_feature, axis=1)
        
        corr = X_concat.corr()
        cond_num = conditional_number(corr)
        
        print(f"Removed {removed_feature}, Conditonal Number: {cond_num}")
    
    print(f"Removed {len(removed_features)} features. Number of features left after selection: {X_concat.shape[1]}")
    return X_train.drop(removed_features, axis=1), y_train, X_test.drop(removed_features, axis=1) 