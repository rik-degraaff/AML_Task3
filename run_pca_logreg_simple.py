import os

from src.boilerplate import main

from src.preprocess.standardscaler import preprocess_standardscaler
from src.model.pipeline_pca_logreg_simple import give_model

model = give_model()

main(name=os.path.basename(__file__), model=model, preprocess=preprocess_standardscaler, get_score=False, save_model=True)