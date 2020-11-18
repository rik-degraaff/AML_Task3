import os

from src.boilerplate import main

from src.preprocess.standardscaler import preprocess_standardscaler
from src.preprocess.autoencoder import autoencoder
from src.model.kernel_svm import give_model

model = give_model()

def autoencoder_35(x_train, y_train, x_test):
    x_train, y_train, x_test = preprocess_standardscaler(x_train, y_train, x_test)
    return autoencoder(35)(x_train, y_train, x_test)

main(name=os.path.basename(__file__), model=model, preprocess=autoencoder_35, get_score=False, save_model=True)