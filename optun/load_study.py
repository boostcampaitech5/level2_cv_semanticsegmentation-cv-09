import optuna
import joblib


# LOAD & USE
load_study = joblib.load("optuna_tuning_model.pkl")
print("Best trial until now:")
print(" Value: ", load_study.best_trial.value)
print(" Params: ")
for key, value in load_study.best_trial.params.items():
    print(f"    {key}: {value}")

optuna.visualization.plot_param_importances(load_study)

