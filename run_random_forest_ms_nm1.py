

from src.boilerplate import main

from src.preprocess.standardscaler_separate import preprocess_standardscaler_separate
from src.preprocess.standardscaler import preprocess_standardscaler
from src.model.random_forest_ms_nm1 import give_model

model = give_model()

main(name="random_forest_smote__tomek_ms", model=model, preprocess=None, cv_splits=None, save_model=True, import_subfolder="processed")