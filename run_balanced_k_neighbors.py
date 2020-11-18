from src.boilerplate import main

from src.preprocess.standardscaler import preprocess_standardscaler
from src.model.balanced_k_neighbors import give_model

model = give_model()

main(name="balanced_k_neighbors", model=model, preprocess=preprocess_standardscaler, cv_splits=None, save_model=True)