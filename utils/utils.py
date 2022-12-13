import torchmetrics
import torch

import optuna

import os
import joblib

from neural_net_models.models_params import ModelsNames


def create_or_empty_folder(path: str):
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)

    else:
        # empty files when a new tunning starts
        for file in os.scandir(path):
            os.remove(file.path)


def create_foder(path):
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)


def saveModel(checkpoint: dict, path_models_id: str, number: int):
    torch.save(checkpoint, path_models_id + str(number))


# save study object to file for later use
def save_optimization_study(study, model_name: ModelsNames, evaluation_function):
    model_name_str = (model_name.name).lower()
    path_study = (
        "models_generated"
        + "//"
        + model_name_str
        + "_study_"
        + evaluation_function
        + ".pkl"
    )

    # path_study= "models_generated/cnn/study" +  "_gru_" + args_trial['evaluation_function']  +".pkl"
    joblib.dump(study, path_study)

    # load study object from file
    # study123 = joblib.load("path_study")


# load study object for use
def load_optimization_study(path_study: str):
    study = joblib.load(path_study)

    return study


def computeMean(values, mean_metric=torchmetrics.MeanMetric(nan_strategy="warn")):
    mean_metric.update(values)
    mean_value = mean_metric.compute()
    mean_metric.reset()

    return mean_value


def print_optimization_results(study: optuna.study.study.Study, evaluation_function):
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print("number of the best trial: ", study.best_trial.number)
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Evaluation function: {}".format(evaluation_function))


# https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-capture
# https://github.com/jupyter/notebook/issues/5269
def capture_notebook_cell(model_name, evaluation_function):
    # add at the top of thw cell:  %%capture vvvvv

    with open(
        (model_name.name).lower() + "_study_" + evaluation_function + "_output.txt", "w"
    ) as f:
        f.write(vvvvv.stdout)
