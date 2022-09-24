''' 
¤ in colab:

!pip install optuna
!pip install import_ipynb
!pip install torchmetrics

'''

''' 
¤ in azure:

conda install -c conda-forge --y optuna 
conda install -c conda-forge --y torchmetrics
'''

''' 
¤ in colab:

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
%cd '/content/drive/My Drive/Colab Notebooks/IDS_Project'
'''


import neural_net_model_train 

DEVICE = neural_net_model_train.DEVICE



import utils_data_prep_py

train_loader = utils_data_prep_py.train_loader
test_loader = utils_data_prep_py.test_loader

input_size = (utils_data_prep_py.X_train).shape[1]
number_of_classes = len(utils_data_prep_py.labels_dict)


import os, glob

from models import ModelsInterface
from neural_net_model_train import ModelsTrainning

import optuna
import torch.optim as optim

import os, glob
import pprint
import statistics

def start_optimize_objective(study:optuna.Study, args, models_interface:ModelsInterface, models_trainning:ModelsTrainning):
    
    max_epochs = args['max_epochs']
    trial_parameter = args['trial_parameter']    
    no_trials = args['no_trials']
    USE_AUTOMATIC_MIXED_PRECISION = args['USE_AUTOMATIC_MIXED_PRECISION']

    # Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
    def objective(trial:optuna.Trial):
        
        #input_features = input_size
        #numClasses = number_of_classes
        #trial controls de parameters of the model
        # mlp_model = define_model(trial, input_features, numClasses)
        net_model = models_interface.define_model(trial)

        print(repr(net_model))
        # print(type(mlp_model))
        # print(dir(mlp_model))

        params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                }
        
        # Generate the optimizers.
        optimizer = getattr(optim, params['optimizer'])(net_model.parameters(), lr = params['learning_rate'])

        models_trainning.setParameters(net_model, optimizer, train_loader, test_loader, max_epochs, trial_optimisation = trial, trial_parameter = trial_parameter)
        
        # use if Tensor Core is present
        if USE_AUTOMATIC_MIXED_PRECISION:
            prunned_trial=False

            try:
                losses, accuracies = models_trainning.trainNetModel_AmpOptimized()
            except ValueError as ve:
                print('prunned_trialprunned_trialprunned_trialprunned_trialprunned_trialprunned_trial')
                prunned_trial=True
                
            # check if trainning finished and thus it was not prunned
            if not prunned_trial:
                #save trained model and continue trainning if network model is fine
                
                checkpoint = models_trainning.getModelCheckpoint_AmpOptimized()
                models_interface.saveModel(checkpoint, trial.number)        

            prunned_trial=False
            
        else:
            prunned_trial=False
            
            try:
                losses, accuracies = models_trainning.trainNetModel()
            except ValueError as ve:
                print('prunned_trialprunned_trialprunned_trialprunned_trialprunned_trialprunned_trial')
                prunned_trial=True
                
            # check if trainning finished and thus it was not prunned
            if not prunned_trial:
                #save trained model and continue trainning if network model is fine
                checkpoint = models_trainning.getModelCheckpoint()
                models_interface.saveModel(checkpoint, trial.number)  
                    
            prunned_trial=False

        accuracy, test_loss = models_trainning.testNetModel()
        
        print('train loss: ', statistics.fmean(losses))
        print('train accuracy: ', accuracies.average)
        
        print('test loss: ',test_loss)
        print('test accuracy: ',accuracy)
        
        return accuracy
    
    models_interface.empty_models()
    
    study.optimize(objective, n_trials = no_trials)