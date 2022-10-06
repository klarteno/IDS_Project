import optuna
import torch



import models_hyper_optim_optuna
from models_trainning_utils import neural_net_model_train

import neural_net_models.models as models
import neural_net_models.models_params as models_params
from neural_net_models.models_params import ModelsNames

from data_preps_utils import utils_data_load

train_loader = utils_data_load.train_loader
test_loader = utils_data_load.test_loader

input_size = (utils_data_load.X_train).shape[1]
number_of_classes = len(utils_data_load.labels_dict)

import utils.utils as utils


#set folders where models will be saved
def get_path_from_folder(model_name=ModelsNames.MLP, evaluation_function='accuracy'):
    model_name_str = (model_name.name).lower()
    
    #set folders where models will be saved
    mlp_path_root_model = "models_generated" + "//" +  model_name_str + "//"+ evaluation_function

    utils.createFolder(mlp_path_root_model)  
    
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
