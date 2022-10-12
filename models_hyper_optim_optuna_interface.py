import optuna
import torch



import models_hyper_optim_optuna
from models_trainning_utils import neural_net_model_train

import neural_net_models.models as models
import neural_net_models.models_params as models_params
from neural_net_models.models_params import ModelsNames

from data_preps_utils import utils_data_load

import utils.utils as utils

from pprint import pprint

train_loader = utils_data_load.train_loader
test_loader = utils_data_load.test_loader

input_size = (utils_data_load.X_train).shape[1]
number_of_classes = len(utils_data_load.labels_dict)


#set folders where models will be saved
def get_path_from_folder(model_name=ModelsNames.MLP, evaluation_function='accuracy'):
    model_name_str = (model_name.name).lower()
    
    #set folders where models will be saved
    mlp_path_root_model = "models_generated" + "//" +  model_name_str + "//"+ evaluation_function

    utils.create_or_empty_folder(mlp_path_root_model)  
    
    return mlp_path_root_model +  "//" + model_name_str + "_"




CHANNELS = 1

def define_model(batch_size, max_epochs, evaluation_function ='accuracy', USE_AUTOMATIC_MIXED_PRECISION=False):

    dataInputParams: models_params.DataInputParams = models_params.DataInputParams()

    dataInputParams.number_of_classes = number_of_classes
    dataInputParams.input_shape = (batch_size, CHANNELS, input_size)
    dataInputParams.input_features = input_size
    dataInputParams.batch_size = batch_size

    dataInputParams.evaluation_function=evaluation_function
    dataInputParams.USE_AUTOMATIC_MIXED_PRECISION=USE_AUTOMATIC_MIXED_PRECISION

    #set models that will be used for tunning
    models_trainning = neural_net_model_train.ModelsTrainning(
                        train_loader,
                        test_loader,
                        max_epochs,
                        evaluation_function,
                        USE_AUTOMATIC_MIXED_PRECISION)
    
    return dataInputParams, models_trainning




def create_study_optimization(dataInputParams:models_params.DataInputParams, models_trainning:neural_net_model_train.ModelsTrainning, number_of_trials:int,model_name = ModelsNames.MLP):
    
    models_ops = models.ModelsOps()
    models_ops.set_model_names(dataInputParams, model_name = model_name)
    
    path_models_id = get_path_from_folder(model_name=model_name, evaluation_function=dataInputParams.evaluation_function)
    
    study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.TPESampler(seed=1121218),
                                pruner=optuna.pruners.HyperbandPruner(min_resource=3))

    models_hyper_optim_optuna.start_optimize_objective(study, models_ops,
                                                    path_models_id,
                                                    models_trainning, number_of_trials)
    
    return study

import torch.optim as optim
from utils.utils_plotting import plot_float_values,plotEvaluationResults

def train_best_model(best_trial: optuna.Trial, dataInputParams, model_name, models_trainning:neural_net_model_train.ModelsTrainning,path_checkpoint_save):

    pprint(best_trial.params)
    print('-----------------------------------------------------------------------')

    # model_name = ModelsNames.MLP
    models_ops = models.ModelsOps()
    models_ops.set_model_names(dataInputParams, model_name = model_name)
    
    mlp_trial_params, mlp_net_model = models_ops.set_model_params(model_name, best_trial.params)
    pprint(mlp_trial_params)
    print('-----------------------------------------------------------------------')
    pprint(mlp_net_model)
    
    
    

    # Generate the optimizers.
    optimizer = getattr(optim, best_trial.params["optimizer"])(
        mlp_net_model.parameters(), lr=best_trial.params["learning_rate"]
    )

    scheduler_learning:torch.optim.lr_scheduler._LRScheduler = getattr(torch.optim.lr_scheduler, "CosineAnnealingWarmRestarts")(
                optimizer=optimizer, T_0 = best_trial.params["scheduler_iterations_restart"], eta_min = best_trial.params["scheduler_minimum_learning_rate"]
            )

    models_trainning.setParameters(mlp_net_model, optimizer, scheduler_learning)

    (   accuracies,
        losses,
        f1_scores,
        auroc_scores,
    ) = models_trainning.train_net_model()

    accuracy_trainning, f1_score_trainning, auroc_trainning = models_trainning.get_evaluations_score()

    print("Train set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t Average f1_score: {:.4f}, Average Area Under ROC: {:.4f} \n".format(
                    utils.computeMean(losses),
                    accuracy_trainning,
                    f1_score_trainning,
                    auroc_trainning,
                )
            )

    checkpoint = models_trainning.getModelCheckpoint()
    torch.save(checkpoint, path_checkpoint_save)   
    



    multi_class_accuracies = models_trainning.get_multi_class_accuracies()    
    print("Train multi class accuracies: ", multi_class_accuracies)        
    plot_float_values(multi_class_accuracies, label_y = 'Multi class accuracies')

    plot_float_values(losses, label_y = 'Losses testset')

    scheduler_learning_history = models_trainning.get_scheduler_learning_history()
    plot_float_values(scheduler_learning_history, label_y = 'Scheduler learning history')
    
    plotEvaluationResults(accuracies,losses,f1_scores,auroc_scores)
    
    
    
def test_best_model(models_trainning:neural_net_model_train.ModelsTrainning):
    accuracies_scores, losses_scores, f1_scores, auroc_scores = models_trainning.testNetModel()

    multi_class_accuracies = models_trainning.get_multi_class_accuracies()        
    accuracy_test, f1_score_test, auroc_test = models_trainning.get_evaluations_score()
    loss_test = utils.computeMean(losses_scores)

    print(
        "\nTest set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t Average f1_score: {:.4f}%, Average Area Under ROC: {:.4f} \n".format(
            loss_test,
            accuracy_test,
            f1_score_test,
            auroc_test,
        )
    )

    print("Test multi class accuracies: ", multi_class_accuracies) 
    
    from utils.utils_plotting import plot_trainning_eval

    plot_float_values(multi_class_accuracies, label_y = 'Multi class accuracies')
    plot_float_values(losses_scores, label_y = 'Losses scores')

