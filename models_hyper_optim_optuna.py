from pprint import pprint
import optuna

import torch
import torch.nn as nn
import torch.optim as optim


import models_trainning_utils.neural_net_model_train as neural_net_model_train
import neural_net_models.models_params as models_params


import utils.utils as utils


def start_optimize_objective(
    study: optuna.Study,
    models_ops ,#:ModelsOps,
    path_models_id,
    models_trainning: neural_net_model_train.ModelsTrainning,
    number_of_trials
):

    # Define a set of hyperparameter values, build the model, train the model,
    # and evaluate the accuracy
    def objective(trial: optuna.Trial):
        #accuracies, losses, f1_scores, auroc_scores = [0], [0], [0], [0]
        losses =[0]

        net_model:nn.Module = models_ops.define_model(trial)

        print(repr(net_model))
        # print(type(mlp_model))
        # print(dir(mlp_model))
        # torch.Cuda.max_memory_allocated()
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            
            "weight_decay" : trial.suggest_float("weight_decay", 0.01, 1.0, log=True),

            "optimizer": trial.suggest_categorical(
                "optimizer", ["RAdam", "AdamW", "NAdam"]
            ),
            
            "scheduler_iterations_restart": trial.suggest_int(
                "scheduler_iterations_restart",low=models_ops.dataInputParams.batch_size*20, high=models_ops.dataInputParams.batch_size*30,log=True
            ),
            
            "scheduler_minimum_learning_rate": trial.suggest_float("scheduler_minimum_learning_rate", 1e-5, 9e-1, log=True),
            
        }    
        # Generate the optimizers.
        optimizer = getattr(optim, params["optimizer"])(
            net_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
        )

        scheduler_learning:torch.optim.lr_scheduler._LRScheduler = getattr(torch.optim.lr_scheduler, "CosineAnnealingWarmRestarts")(
            optimizer=optimizer, T_0 = params["scheduler_iterations_restart"], eta_min = params["scheduler_minimum_learning_rate"]
        )
        
        models_trainning.setParameters(
            net_model,
            optimizer,
            scheduler_learning,
            trial_optimisation=trial
        )

        # use if Tensor Core is present
        prunned_trial = False
        try:
            ( accuracies,
            losses,
            f1_scores,
            auroc_scores,
            ) = models_trainning.train_net_model()
            
        
        except ValueError as ve:
                prunned_trial = True
        
        if not prunned_trial:
            # save trained model and continue trainning if network model is
            # fine
            checkpoint = models_trainning.getModelCheckpoint()
            utils.saveModel(checkpoint, path_models_id, trial.number)
    
        multi_class_accuracies = models_trainning.get_multi_class_accuracies()
        accuracy_trainning,f1_score_trainning,auroc_trainning = models_trainning.get_evaluations_score()
        
        print("\nTrain set multi class accuracies: ", multi_class_accuracies)        
        print(
            "Train set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t  Average f1_score: {:.4f}%, Average Area Under ROC: {:.4f} \n".format(
                utils.computeMean(losses),
                accuracy_trainning,
                f1_score_trainning,
                auroc_trainning,
            )
        )

        # turn off because of RNN :https://github.com/pytorch/captum/issues/564
        # , https://github.com/pytorch/captum/pull/576
        torch.backends.cudnn.enabled = False
        accuracies_scores, losses_scores, f1_scores, auroc_scores = models_trainning.testNetModel()
        torch.backends.cudnn.enabled = True


        multi_class_accuracies = models_trainning.get_multi_class_accuracies()
        accuracy_trainning,f1_score_trainning,auroc_trainning = models_trainning.get_evaluations_score()
        loss_training=utils.computeMean(losses_scores)
        
        print("Test set multi class accuracies: ", multi_class_accuracies)        
        
        print(
            "\nTest set: Average loss: {:.4f}, Average accuracy: {:.4f}%, \n \t  Average f1_score: {:.4f}%, Average Area Under ROC: {:.4f} \n".format(
                loss_training,
                accuracy_trainning,
                f1_score_trainning,
                auroc_trainning,
            )
        )
            
        prunned_trial = False
        
        return models_trainning.get_evaluation_result(  accuracy_trainning,
                                                        loss_training,
                                                        f1_score_trainning)
        
        

    study.optimize(objective, n_trials=number_of_trials)
    # when is ruuning on cpu: n_jobs is 2 or more the training fails with errors
    # study.optimize(objective ,timeout = 5 * 60 * 60)
