import optuna

from models_params import CnnBiRnnTrialParams, MlpTrialParams, ModelsNames


class TrialModelSettings:
    def __init__(self):
        self.n_layers:str = "n_layers"
        self.num_units_l:str = "num_units_l"
        self.dropout_l:str = "dropout_l"
        
        self.no_strides_l:str = "no_strides_l"
        
        self.rnn_drop_procentages:str = "rnn_drop_procentages"
        
    
    def set_model_params_from(self, trial: optuna.Trial, model_name:ModelsNames):
        
        #check instance of net_model and set params accordingly  
        if model_name.value == ModelsNames.MLP.value:
            mlp_trial_params=MlpTrialParams()
            
            mlp_trial_params.number_layers = trial.suggest_int(self.n_layers, 2, 6)
            mlp_trial_params.stack_layers=[]
            
            for i in range(mlp_trial_params.number_layers):
                num_units_l = trial.suggest_int(self.num_units_l+"{}".format(i), 25, 256, step=3)
                dropout_l = trial.suggest_float(
                    self.dropout_l+"{}".format(i), 0.1, 0.5, log=True
                )
                
                mlp_trial_params.stack_layers.append((num_units_l,dropout_l))
                
            return mlp_trial_params
        
        elif (model_name.value is ModelsNames.CNN_BI_LSTM.value) or (model_name.value is ModelsNames.CNN_BI_GRU.value):
            cnn_birnn_trial_params=CnnBiRnnTrialParams()
            
            cnn_birnn_trial_params.number_layers= no_layers = trial.suggest_int(self.n_layers, 2, 4)
            cnn_birnn_trial_params.stack_layers=[]
            
            for i in range(no_layers):
                num_units_l = trial.suggest_int(self.num_units_l+"{}".format(i), 25, 88, step=3)
                no_strides = trial.suggest_int(self.no_strides_l+"{}".format(i), 1, 7, step=2)
                
                cnn_birnn_trial_params.stack_layers.append((num_units_l, no_strides))
             
            rnn_drop_procentages = trial.suggest_float(self.rnn_drop_procentages+"{}".format(i+1), 0.1, 0.5, log=True)
            cnn_birnn_trial_params.rnn_drop_procentages=rnn_drop_procentages 
                
            return cnn_birnn_trial_params
        
        else:
            raise NotImplementedError("Not implemented for other models")
        
    def set_model_params(self, model_name:ModelsNames, args: dict):
        if model_name.value == ModelsNames.MLP.value:
            mlp_trial_params=MlpTrialParams()
            
            mlp_trial_params.number_layers = args[self.n_layers]

            for i in range(mlp_trial_params.number_layers):
                num_units_l = self.num_units_l+"{}".format(i)
                dropout_l = self.dropout_l+"{}".format(i)
                
                mlp_trial_params.stack_layers.append((args[num_units_l], args[dropout_l]))
        
            return mlp_trial_params
        
        
        elif (model_name.value is ModelsNames.CNN_BI_LSTM.value) or (model_name.value is ModelsNames.CNN_BI_GRU.value):
            cnn_birnn_trial_params=CnnBiRnnTrialParams()
            
            cnn_birnn_trial_params.number_layers = args[self.n_layers]
            
            for i in range(cnn_birnn_trial_params.number_layers):
                num_unit = self.num_units_l+"{}".format(i)
                no_stride = self.no_strides_l+"{}".format(i)
                
                cnn_birnn_trial_params.stack_layers.append((args[num_unit], args[no_stride]))

            cnn_birnn_trial_params.rnn_drop_procentages = args[self.rnn_drop_procentages+"{}".format(i+1)]      
            
            return cnn_birnn_trial_params
            
        else:
            raise NotImplementedError("Not implemented for other models")
        