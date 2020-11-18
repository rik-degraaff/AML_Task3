import os
import pickle
import numpy as np

from src.utils import path_project

from src.boilerplate import create_csv

def highest(x):
    return (x == x.max(axis=1)[:,None]).astype(int)

def softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(x - mx)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator/denominator

input_folder = path_project + "input_for_metaclassifier/"

#x_train_svc = pickle.load(open(input_folder + "y_train_df_svc.sav", "rb"))
x_test_svc = highest(pickle.load(open(input_folder + "y_test_df_svc.sav", "rb")))

#x_train_logreg = pickle.load(open(input_folder + "y_train_proba_pca_logreg_2.sav", "rb"))
x_test_logreg = highest(pickle.load(open(input_folder + "y_test_proba_pca_logreg_2.sav", "rb")))

#x_train_xgb = pickle.load(open(input_folder + "y_train_proba_smote_xgb_3.sav", "rb"))
x_test_xgb = highest(pickle.load(open(input_folder + "y_test_proba_smote_xgb_3.sav", "rb")))

weighted_sums = 0.715327818482 * x_test_svc + 0.696462625431 * x_test_logreg + 0.655322666413 * x_test_xgb
y_pred = np.argmax(weighted_sums, axis=1)

create_csv(y_pred, os.path.basename(__file__))