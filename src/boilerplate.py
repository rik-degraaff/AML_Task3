import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline

from utils import path_project


def import_data(subfolder=None):
    # extended paths to the .csv files
    if subfolder is not None:
        path_X_train = path_project + 'data/' + subfolder + '/X_train.csv'
        path_X_test = path_project + 'data/' + subfolder + '/X_test.csv'
        path_y_train = path_project + 'data/' + subfolder + '/y_train.csv'
    else:
        path_X_train = path_project + 'data/raw_data/X_train.csv'
        path_X_test = path_project + 'data/raw_data/X_test.csv'
        path_y_train = path_project + 'data/raw_data/y_train.csv'

    # import the files as panda data frames
    X_train = pd.read_csv(path_X_train, sep=',', index_col=0, low_memory=False)
    X_test = pd.read_csv(path_X_test, sep=',', index_col=0, low_memory=False)
    y_train = pd.read_csv(path_y_train, sep=',', index_col=0)

    return X_train, y_train, X_test

def create_csv(prediction, name):
    y_test_table = np.zeros((prediction.shape[0], 2))

    for i in range(0, prediction.shape[0]):
        y_test_table[i, 0] = i
        y_test_table[i, 1] = prediction[i]

    # dump into a csv file
    os.makedirs(path_project + 'predictions/', exist_ok=True)
    np.savetxt(path_project + "predictions/" + name + ".csv", y_test_table,
                   header='id,y', comments='', delimiter=",", fmt=['%1.1f', '%1.10f'])
    
    
def get_predict_proba(name, model, X_train, y_train, X_test, n_splits=5, get_decision_function=False):
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=0)
    
    y_train_pred_proba = None
    
    for train_idx, test_idx in kf.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train.iloc[train_idx, :], X_train.iloc[test_idx, :]
        y_train_cv, y_test_cv = y_train.iloc[train_idx, :], y_train.iloc[test_idx, :]
        
        model.fit(X_train_cv, y_train_cv.values.ravel())
        
        if get_decision_function:
            temp = model.decision_function(X_test_cv)
        else:            
            temp = model.predict_proba(X_test_cv)
            
        if y_train_pred_proba is None:
            y_train_pred_proba = temp
        else:
            y_train_pred_proba = np.vstack([y_train_pred_proba, temp])
        

    pickle.dump(y_train_pred_proba, open(path_project+"input_for_metaclassifier/y_train_pred_proba_"+name+".sav", "wb"))
    
    model.fit(X_train, y_train.values.ravel())
    
    if get_decision_function:
        y_test_pred_proba = model.decision_function(X_test)
    else:            
        y_test_pred_proba = model.predict_proba(X_test)
    
    pickle.dump(y_test_pred_proba, open(path_project+"input_for_metaclassifier/y_test_pred_proba_"+name+".sav", "wb"))
        
        

def main(name, model, preprocess=None, get_pred=True, get_score=True, cv_splits=None, save_model=True, import_subfolder=None, save_preprocessed_data=False, get_proba=False, get_decision_function=False):
    
    X_train, y_train, X_test = import_data(import_subfolder)
        
    result_entry = {}
    result_entry["name"] = name
    
    if isinstance(model, GridSearchCV):
        result_entry["is_GridSearchCV"] = True
        result_entry["model/pipeline"] = model.estimator
        result_entry["param_grid"] = model.param_grid
    else:
        result_entry["is_GridSearchCV"] = False
        result_entry["model/pipeline"] = str(model)
        result_entry["param_grid"] = None
        result_entry["best_params_"] = model.get_params()
    
    if preprocess==None:
        result_entry["preprocess"] = None
    else:
        X_train, y_train, X_test = preprocess(X_train, y_train, X_test)
        result_entry["preprocess"] = preprocess.__name__

    if save_preprocessed_data:
        os.makedirs(path_project + 'data/' + preprocess.__name__ + '/', exist_ok=True)
        X_train.to_csv(path_project + 'data/data_' + preprocess.__name__ + "_" + name + '/X_train.csv')
        y_train.to_csv(path_project + 'data/data_' + preprocess.__name__ + "_" + name + '/y_train.csv')
        X_test.to_csv(path_project + 'data/data_' + preprocess.__name__ + "_" + name + '/X_test.csv')
    
    if get_score:
        if isinstance(cv_splits, int):
            
            print("Getting cv-scores...")

            result_entry["cv_splits"] = cv_splits
            result_entry["test_score"] = None

            cv = StratifiedKFold(n_splits=cv_splits, shuffle=False)
            cv_scores = []

            for train_idx, test_idx in cv.split(X_train, y_train):

                X_train_cv, X_test_cv = X_train.iloc[train_idx, :], X_train.iloc[test_idx, :]
                y_train_cv, y_test_cv = y_train.iloc[train_idx, :], y_train.iloc[test_idx, :]

                model.fit(X_train_cv, y_train_cv.values.ravel())
                if isinstance(model, GridSearchCV):
                    print(model.best_params_)
                y_pred_cv = model.predict(X_test_cv)
                cv_score = balanced_accuracy_score(y_test_cv, y_pred_cv)
                cv_scores.append(cv_score)
                print(f"cv_scores: {cv_scores}")

            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            print(f"mean_cv_score={mean_cv_score:.3f} ({std_cv_score:.3f})")

            result_entry["cv_scores"] = cv_scores
            result_entry["mean_cv_score"] = mean_cv_score
            result_entry["std_cv_score"] = std_cv_score

        else:
            
            print("Getting test score...")

            result_entry["cv_splits"] = None
            result_entry["cv_scores"] = None
            result_entry["mean_cv_score"] = None
            result_entry["std_cv_score"] = None

            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2)

            model.fit(X_train_split, y_train_split.values.ravel())
            y_pred_split = model.predict(X_test_split)
            score = balanced_accuracy_score(y_test_split, y_pred_split)

            result_entry["test_score"] = score
            
    if get_pred:
        
        print("Refitting model on all data and getting predictions...")

        model.fit(X_train, y_train.values.ravel())
        
        if isinstance(model, GridSearchCV):
            result_entry["cv_results_"] = model.cv_results_
            
        y_pred = model.predict(X_test)
        create_csv(y_pred, "y_pred_" + name)
        

        
        if save_model:
            print("Saving model...")
            os.makedirs(path_project + 'trained_model/', exist_ok=True)
            model_dump_filename = path_project + 'trained_model/' + name + '.sav'
            pickle.dump(model, open(model_dump_filename, 'wb'))
    
    print("Saving results...")
    if os.path.isfile("results.csv"):
        result = pd.read_csv("results.csv", index_col=None, header=0)
    else:
        result = pd.DataFrame(columns=["name", "preprocess_step", "is_GridSearchCV", "model/pipeline", "param_grid", "cv_splits", "test_score", "cv_scores", "mean_cv_score", "std_cv_score", "best_params_", "cv_results_"])
    
    result = result.append(result_entry, ignore_index=True)
    result.to_csv("results.csv", index=False, header=True)
    
    if get_proba:
        print("Predicting probabilities...")
        get_predict_proba(name, model, X_train, y_train, X_test, n_splits=10, get_decision_function=get_decision_function) 
        
    print("Completed.")