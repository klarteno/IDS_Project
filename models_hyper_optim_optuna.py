""" 
¤ in colab:

!pip install optuna
!pip install import_ipynb
!pip install torchmetrics

"""

""" 
¤ in azure:

conda install -c conda-forge --y optuna 
conda install -c conda-forge --y torchmetrics
"""

""" 
¤ in colab:

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd '/content/drive/My Drive/Colab Notebooks/IDS_Project'
"""

import optuna
import torch.optim as optim

import statistics


import utils_data_load

train_loader = utils_data_load.train_loader
test_loader = utils_data_load.test_loader

input_size = (utils_data_load.X_train).shape[1]
number_of_classes = len(utils_data_load.labels_dict)

import models
import neural_net_model_train

import os


def createFolder(path:str):
    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
        
    else:
        #empty files when a new tunning starts
        for file in os.scandir(path):
            os.remove(file.path)   


def start_optimize_objective(
    study: optuna.Study,
    args,
    path_models_id,
    models_interface:models.CnnBirnnModel,
    models_trainning:neural_net_model_train.ModelsTrainning
):

    max_epochs = args["max_epochs"]
    evaluation_function = args["evaluation_function"]
    no_trials = args["no_trials"]
    USE_AUTOMATIC_MIXED_PRECISION = args["USE_AUTOMATIC_MIXED_PRECISION"]
    
    # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
    def objective(trial: optuna.Trial):
        accuracies, losses, f1_scores, auroc_scores = [0], [0], [0], [0]

        net_model = models_interface.define_model(trial)

        print(repr(net_model))
        # print(type(mlp_model))
        # print(dir(mlp_model))

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
            "optimizer": trial.suggest_categorical(
                "optimizer", ["Adam", "RMSprop", "SGD"]
            ),
        }

        # Generate the optimizers.
        optimizer = getattr(optim, params["optimizer"])(
            net_model.parameters(), lr=params["learning_rate"]
        )

        models_trainning.setParameters(
            net_model,
            optimizer,
            train_loader,
            test_loader,
            max_epochs,
            trial_optimisation=trial,
            evaluation_function=evaluation_function,
        )

        # use if Tensor Core is present
        if USE_AUTOMATIC_MIXED_PRECISION:
            prunned_trial = False

            try:
                accuracies, losses, f1_scores, auroc_scores = models_trainning.trainNetModel_AmpOptimized()
            except ValueError as ve:
                prunned_trial = True

            # check if trainning finished and thus it was not prunned
            if not prunned_trial:
                # save trained model and continue trainning if network model is fine
                checkpoint = models_trainning.getModelCheckpoint_AmpOptimized()
                models_interface.saveModel(checkpoint, path_models_id, trial.number)

            prunned_trial = False

        else:
            prunned_trial = False

            try:
                accuracies, losses, f1_scores, auroc_scores = models_trainning.trainNetModel()
            except ValueError as ve:
                prunned_trial = True
                

            # check if trainning finished and thus it was not prunned
            if not prunned_trial:
                # save trained model and continue trainning if network model is fine
                checkpoint = models_trainning.getModelCheckpoint()
                models_interface.saveModel(checkpoint, path_models_id, trial.number)

            prunned_trial = False

        print(
            "\nTrain set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t Average f1_score: {:.4f}, Average Area Under ROC: {:.4f} \n".format(
                statistics.fmean(accuracies),
                statistics.fmean(losses),
                statistics.fmean(f1_scores),
                statistics.fmean(auroc_scores),
            )
        )

        test_evaluation_function = models_trainning.testNetModel()

        return test_evaluation_function

    models_interface.empty_models(evaluation_function)

    study.optimize(objective, n_trials=no_trials)
    # study.optimize(objective, timeout=200)
