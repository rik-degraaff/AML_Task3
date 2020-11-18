from src.boilerplate import main

from src.preprocess.standardscaler import preprocess_standardscaler
from src.model.pipeline_pca_logreg import give_model

model = give_model()

main(name="smote_pca_log_reg", model=model, preprocess=preprocess_standardscaler, cv_splits=None, save_model=True)