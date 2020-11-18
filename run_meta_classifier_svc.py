import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from src.utils import path_project
from src.boilerplate import main
from src.model.kernel_svm import give_model

model = give_model()

def preprocess(x_train, y_train, x_test):
    input_folder = path_project + "input_for_metaclassifier/"

    x_train_svc = pickle.load(open(input_folder + "y_train_pred_proba_ae_rbf_svc_10fold.sav", "rb"))
    x_test_svc = pickle.load(open(input_folder + "y_test_pred_proba_ae_rbf_svc_10fold.sav", "rb"))

    #x_train_xgb = pickle.load(open(input_folder + "y_train_proba_pca_smote_xgb.sav", "rb"))
    #x_test_xgb = pickle.load(open(input_folder + "y_test_proba_pca_smote_xgb.sav", "rb"))

    x_train_logreg = pickle.load(open(input_folder + "y_train_pred_proba_pca_logreg_2_10fold.sav", "rb"))
    x_test_logreg = pickle.load(open(input_folder + "y_test_pred_proba_pca_logreg_2_10fold.sav", "rb"))

    kf = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
    x_train_ae = pickle.load(open(path_project + "processed_data/run_ae_50_rbf_svc.py_X_train.sav", "rb"))
    y_train = y_train.to_numpy()
    x_train_ae_rearranged = None
    y_train_rearranged = None
    for train_idx, test_idx in kf.split(x_train_ae, y_train):
        x_train_ae_cv = x_train_ae[test_idx, :]
        y_train_cv = y_train[test_idx, :]
        if x_train_ae_rearranged is None:
            x_train_ae_rearranged = x_train_ae_cv
            y_train_rearranged = y_train_cv
        else:
            x_train_ae_rearranged = np.vstack([x_train_ae_rearranged, x_train_ae_cv])
            y_train_rearranged = np.vstack([y_train_rearranged, y_train_cv])
    
    x_test_ae = pickle.load(open(path_project + "processed_data/run_ae_50_rbf_svc.py_X_test.sav", "rb"))

    return np.hstack([x_train_svc, x_train_logreg, x_train_ae_rearranged]), pd.DataFrame(y_train_rearranged), np.hstack([x_test_svc, x_test_logreg, x_test_ae])

main(name=os.path.basename(__file__), model=model, preprocess=preprocess, get_score=False, save_model=True)