import os

from src.boilerplate import main

from src.preprocess.standardscaler import preprocess_standardscaler
from src.preprocess.autoencoder import autoencoder
from src.model.pipeline_pca_logreg_simple import give_model

model = give_model()

def preprocess(x_train, y_train, x_test):
    x_train, y_train, x_test = preprocess_standardscaler(x_train, y_train, x_test)
    return autoencoder(100)(x_train, y_train, x_test)

main(name=os.path.basename(__file__), model=model, preprocess=preprocess, cv_splits=None, save_model=True)