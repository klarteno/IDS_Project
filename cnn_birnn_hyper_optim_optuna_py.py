import optuna


import models_hyper_optim_optuna
import models

args = {
    "max_epochs": 15,
    "no_trials": 15,
    "trial_parameter": "accuracy",
    "USE_AUTOMATIC_MIXED_PRECISION": True,
}
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
)


args_model = {
    "input_features": 1,
    "number_of_classes": models_hyper_optim_optuna.number_of_classes,
    "input_shape": (32, models_hyper_optim_optuna.input_size, 78),
    "use_gru_instead_of_lstm": False,
}


cnn_birnn_model = models.CnnBirnnModel(args_model)
models_trainning = models_hyper_optim_optuna.neural_net_model_train.ModelsTrainning()

models_hyper_optim_optuna.start_optimize_objective(
    study, args, cnn_birnn_model, models_trainning
)

import time

start = time.time()
stop = time.time()
time_run = stop - start
print(time_run)


import plotly

optuna.visualization.plot_intermediate_values(study)


optuna.visualization.plot_optimization_history(study)


optuna.visualization.plot_parallel_coordinate(study)


optuna.visualization.plot_param_importances(study)
