
import optuna
import torchmetrics

import torch
import torch.optim as optim

import models
import neural_net_model_train

import os


def createFolder(path: str):
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)

    else:
        # empty files when a new tunning starts
        for file in os.scandir(path):
            os.remove(file.path)

def computeMean(values, mean_metric=torchmetrics.MeanMetric(nan_strategy="warn")):
    mean_metric.update(values)
    mean_value = mean_metric.compute()
    mean_metric.reset()

    return mean_value

def print_optimization_results(study:optuna.study.study.Study, args_trial):
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print('number of the best trial: ',study.best_trial.number)
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
        
    print("  Evaluation function: {}".format(args_trial['evaluation_function']))



def start_optimize_objective(
    study: optuna.Study,
    args,
    path_models_id,
    models_interface: models.ModelsInterface,
    models_trainning: neural_net_model_train.ModelsTrainning,
):

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
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
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
            trial_optimisation=trial,
            evaluation_function=evaluation_function,
        )

        # use if Tensor Core is present
        if USE_AUTOMATIC_MIXED_PRECISION:
            prunned_trial = False

            try:
                (
                    accuracies,
                    losses,
                    f1_scores,
                    auroc_scores,
                ) = models_trainning.trainNetModel_AmpOptimized()
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
                (
                    accuracies,
                    losses,
                    f1_scores,
                    auroc_scores,
                ) = models_trainning.trainNetModel()
            except ValueError as ve:
                prunned_trial = True

            # check if trainning finished and thus it was not prunned
            if not prunned_trial:
                # save trained model and continue trainning if network model is fine
                checkpoint = models_trainning.getModelCheckpoint()
                models_interface.saveModel(checkpoint, path_models_id, trial.number)

            prunned_trial = False

        # the evaluation are scaled between 0 and 1
        print(
            "\nTrain set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t Average f1_score: {:.4f}, Average Area Under ROC: {:.4f} \n".format(
                computeMean(losses),
                computeMean(accuracies),
                computeMean(f1_scores),
                computeMean(auroc_scores),
            )
        )

        # turn off because of RNN :https://github.com/pytorch/captum/issues/564 , https://github.com/pytorch/captum/pull/576
        torch.backends.cudnn.enabled = False
        test_evaluation_function = models_trainning.testNetModel()
        torch.backends.cudnn.enabled = True
        
        net_model=None

        return test_evaluation_function

    study.optimize(objective, n_trials=no_trials)
    #when is ruuning on cpu: n_jobs is 2 or more the training fails with errors
    #study.optimize(objective ,timeout = 5 * 60 * 60)
    
    


    
