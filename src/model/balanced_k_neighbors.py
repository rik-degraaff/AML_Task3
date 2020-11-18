import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import KNeighborsMixin

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline

class BalancedKNeighborsClassifier(KNeighborsClassifier):
    def fit(self, X, y):
        def count(elem):
            counts = dict()
            if counts.get(elem) is None:
                counts[elem] = np.count_nonzero(y == elem)
            return counts[elem]
        self._weights = [1 / count(elem) for elem in y]
        return KNeighborsClassifier.fit(self, X, y)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        result = KNeighborsMixin.kneighbors(self, X=X, n_neighbors=n_neighbors, return_distance=return_distance)
        if return_distance:
            distances, indexes = result            
            distances = distances * np.vectorize(lambda i: self._weights[i])(indexes)
            return distances, indexes
        return result


def give_model():
    n_splits = 5
    param_grid = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [6, 12, 18, 24]
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    model = GridSearchCV(BalancedKNeighborsClassifier(), param_grid=param_grid, cv=cv, scoring='balanced_accuracy', return_train_score=True, n_jobs=-1, verbose=1)
    
    return model