
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn import metrics

def give_model():
    steps = [("smote", SMOTE()), ("undersampling", TomekLinks()), ("featureselect", SelectFromModel(RandomForestClassifier())), ("randomforest", RandomForestClassifier())]
    param_grid = {'smote__k_neighbors': [3,5,7,9],
                  'smote__random_state': [0],
                  'undersampling__sampling_strategy': ['auto', 'all', 'majority', 'not minority'],
                  'featureselect__estimator__class_weight': ['balanced', 'balanced_subsample'],
                  'featureselect__estimator__max_features': ['auto'],
                  'randomforest__n_estimators': [100],
                  'randomforest__random_state': [0],
                  'randomforest__class_weight': ['balanced', 'balanced_subsample'],
                  'randomforest__max_features': ['auto']
                  }

    model = GridSearchCV(Pipeline(steps=steps), param_grid=param_grid, scoring='f1_micro', return_train_score=True, n_jobs=-1, verbose=1)
    return model
