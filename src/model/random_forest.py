

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics

def give_model():
    model = RandomForestClassifier(n_estimators=150, random_state=0, class_weight='balanced')
    return model