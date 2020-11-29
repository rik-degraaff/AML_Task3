from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline

def give_model():
    n_splits = 5
    param_grid = {
        'kernel': ['rbf'],
        'C': [500, 700, 1000, 1400, 2000],
        'gamma': [0.0003, 0.0006, 0.001, 0.002, 0.003,], #use scale instead
        'class_weight': ['balanced'],
        'random_state': [0]
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    model = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring='f1_micro', return_train_score=True, n_jobs=-1, verbose=1)
    
    return model