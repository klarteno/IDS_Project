from typing import Type
import optuna
import torch
import torch.nn as nn

from torchinfo import summary

import models_params

from  models_params import ModelsNames

import trial_model_settings

if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelsOps:
    def __init__(self):
        self.trialModelSettings = trial_model_settings.TrialModelSettings()


    def set_model_names(self, dataInputParams: models_params.DataInputParams, model_name:models_params.ModelsNames):
        self.model_name = model_name
        self.dataInputParams = dataInputParams
        
        
    def define_model(self, trial: optuna.Trial) -> nn.Module:
        if self.model_name.value == models_params.ModelsNames.MLP.value:
            mlp_trial_params = self.trialModelSettings.set_model_params_from(trial, self.model_name)
            
            return MlpModel(self.dataInputParams).define_model(mlp_trial_params)
        
        elif (self.model_name.value == models_params.ModelsNames.CNN_BI_LSTM.value) or (self.model_name.value == models_params.ModelsNames.CNN_BI_GRU.value):
            cnn_birnn_trial_params = self.trialModelSettings.set_model_params_from(trial, self.model_name)
            
            return CnnBirnnModel(self.dataInputParams).define_model(cnn_birnn_trial_params, self.model_name)
        else:
            raise NotImplementedError("Not implemented for other models")


    def get_zero_rate_classifier_model(self) -> nn.Module:
        return ZeroRateClassifier()


    def set_model_params(self, model_name:ModelsNames, args: dict):
        if self.model_name.value == models_params.ModelsNames.MLP.value:
            mlp_trial_params=self.trialModelSettings.set_model_params(model_name, args)
            return  mlp_trial_params, MlpModel(self.dataInputParams).define_model(mlp_trial_params)  
              
        elif (self.model_name.value == models_params.ModelsNames.CNN_BI_LSTM.value) or (self.model_name.value == models_params.ModelsNames.CNN_BI_GRU.value):
            cnn_birnn_trial_params = self.trialModelSettings.set_model_params(model_name, args)
            return cnn_birnn_trial_params, CnnBirnnModel(self.dataInputParams).define_model(cnn_birnn_trial_params, model_name)    
        else:
            raise NotImplementedError("Not implemented for other models")


# TO DO: delete this class
class ModelsInterface2:
    def __init__(self, dataInputParams: models_params.DataInputParams):
        self.number_of_classes:int = dataInputParams.number_of_classes
        self.input_shape = dataInputParams.input_shape
        self.input_features = dataInputParams.input_features  # input_shape[2]
        self.batch_size:int = dataInputParams.batch_size

    def define_model(self, trialParams:None) -> nn.Module:
        print("ModelsInterface define_model")
        
        return None    

def _check_layer_output(layer, input_shape):
    _summary = summary(layer, input_size=input_shape, device=DEVICE)
    input_shape = _summary.summary_list[0].output_size

    return input_shape, input_shape[-1]


def _check_nans(outputs):
    if torch.isnan(outputs).any():
        print("NANs prunned")
        raise optuna.exceptions.TrialPruned()



class  ZeroRateClassifier(nn.Module):
    def __init__(self, most_frequent = torch.full((32, 15, 1), 0.0)):
        super(ZeroRateClassifier, self).__init__()
        
        #code to use when weightedsampler is not used
        '''
        y_dtset = train_loader.dataset.Y_train
        type(y_dtset)
        max_elements, max_idxs = torch.min(y_dtset, dim=0)
        print(max_elements.item(), max_idxs.item())
        print(torch.mode(y_dtset, dim=-1, keepdim=True, out=None))

        print(torch.bincount(y_dtset).size())
        print(torch.bincount(y_dtset))


        lst_bincount=torch.bincount(y_dtset)

        max_elements, max_idxs = torch.max(lst_bincount, dim=0)
        print(max_elements.item(), max_idxs.item())
        sumsi = torch.sum(lst_bincount,dim=0,keepdim=True)
        '''
        
        self.most_frequent=most_frequent.to(DEVICE)
        
    def forward(self, x_inputs) -> int:
        return self.most_frequent
        
        
class MlpModel():
    def __init__(self, dataInputParams: models_params.DataInputParams):
        self.dataInputParams=dataInputParams
        
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    # Build a model by providing model parameters
    def define_model(self, mlp_trial_params:models_params.MlpTrialParams):
        no_layers = mlp_trial_params.number_layers 
        
        def init_func(self):
            super(NetModel, self).__init__()
           
        dataInputParams=self.dataInputParams
        # @torch.cuda.amp.autocast()
        def forward_func(self, inputs, no_layers=no_layers):
            outputs = getattr(self, f"flat")(inputs)

            for index in range(1, no_layers + 1):
                outputs = getattr(self, f"lin" + str(index))(outputs)
                _check_nans(outputs)

                outputs = getattr(self, f"bn" + str(index))(outputs)
                outputs = getattr(self, f"lrel" + str(index))(outputs)
                outputs = getattr(self, f"drop" + str(index))(outputs)

            _check_nans(outputs)

            outputs = getattr(self, f"fc_output")(outputs)
            _check_nans(outputs)
            
            outputs = outputs.view(dataInputParams.batch_size, -1)
            linear_output = (nn.Linear(outputs.shape[1], dataInputParams.number_of_classes)).to(
                DEVICE
            )

            # apply Linear to get shape of: (batch_size, number_of_classes)
            # the Linear is generated in this way to make the output fit the
            # required shape and to let RNN layer be as wide as possible
            outputs = linear_output(outputs)

            # reshaped because  has to match the labels target
            outputs = outputs.unsqueeze(2)

            return outputs

        NetModel = type(
            "NetModel", (nn.Module,),
            {
                "__init__": init_func, "forward": forward_func,
            },
        )
        mlp_model: NetModel = NetModel()

        setattr(mlp_model, "flat", nn.Flatten())

        input_features = self.dataInputParams.input_features
        input_shape     = self.dataInputParams.input_shape
        
        for i in range(mlp_trial_params.number_layers):
            out_features, drop_procentages=mlp_trial_params.stack_layers[i]

            lin_layer = nn.Linear(input_features, out_features)
            input_shape, output_size = _check_layer_output(lin_layer, input_shape)

            setattr(mlp_model, "lin" + str(i + 1), lin_layer)
            setattr(mlp_model, "bn" + str(i + 1), nn.BatchNorm1d(out_features))
            setattr(mlp_model, "lrel" + str(i + 1), nn.LeakyReLU())
            setattr(mlp_model, "drop" + str(i + 1), nn.Dropout(drop_procentages))

            input_features = out_features
        
        setattr(mlp_model, "fc_output", nn.Linear(input_features, self.dataInputParams.batch_size))
        # mlp_model.apply(self._weights_init)

        return mlp_model.apply(self._weights_init)


class CnnBirnnModel():
    def __init__(self, dataInputParams: models_params.DataInputParams):
        self.dataInputParams=dataInputParams
        
    def _weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    # remove Dropout because is still researched how to appply to Conv , some details here:
    # https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html

    # Build a model by implementing define-by-run design from Optuna
    def define_model(self, cnn_birnn_trial_params:models_params.CnnBiRnnTrialParams, model_name:models_params.ModelsNames):
        
        use_gru_instead_of_lstm = (model_name.value == models_params.ModelsNames.CNN_BI_GRU.value)
        no_layers = cnn_birnn_trial_params.number_layers
        
        # to be optimized by optuna or no because the hidden size is big?
        rnn_stacked_layers = 2

        def init_func(self):
            super(NetModel, self).__init__()

        dataInputParams=self.dataInputParams
        # @torch.cuda.amp.autocast()
        def forward_func(self, inputs, no_layers=no_layers):
            outputs = inputs

            for index in range(1, no_layers + 1):
                outputs = getattr(self, f"cv" + str(index))(outputs)
                _check_nans(outputs)

                outputs = getattr(self, f"avgp" + str(index))(outputs)
                _check_nans(outputs)

                outputs = getattr(self, f"bn" + str(index))(outputs)
                _check_nans(outputs)

                outputs = getattr(self, f"lrel" + str(index))(outputs)

            outputs = torch.transpose(outputs, 1, 2)

            if use_gru_instead_of_lstm:
                h0 = torch.zeros(
                    rnn_stacked_layers * 2, outputs.size(0), self.hidden_size_lstm
                ).to(DEVICE)

                outputs, hidden = getattr(self, f"bi_gru")(outputs, h0)
            else:
                h0 = torch.zeros(
                    rnn_stacked_layers * 2, outputs.size(0), self.hidden_size_lstm
                ).to(DEVICE)
                c0 = torch.zeros(
                    rnn_stacked_layers * 2, outputs.size(0), self.hidden_size_lstm
                ).to(DEVICE)

                outputs, hidden = getattr(self, f"bi_lstm")(outputs, (h0, c0))
            _check_nans(outputs)

            outputs = getattr(self, f"fc_output")(outputs)
            _check_nans(outputs)

            outputs = outputs.view(dataInputParams.batch_size, -1)
            linear_output = (nn.Linear(outputs.shape[1], dataInputParams.number_of_classes)).to(
                DEVICE
            )

            # apply Linear to get shape of: (batch_size,number_of_classes)
            # the Linear is generated in this way to make the output fit the
            # required shape and to let RNN layer be as wide as possible
            outputs = linear_output(outputs)
            _check_nans(outputs)

            # reshaped because CrossEntropyLoss has to match the target
            outputs = outputs.unsqueeze(2)

            return outputs

        NetModel = type(
            "NetModel",
            (nn.Module,),
            {
                "__init__": init_func,
                "forward": forward_func,
            },
        )

        cnn_bilstm_model = NetModel()
        kernel_size = 3

        input_shape = self.dataInputParams.input_shape
        input_features = 1
        for i in range(no_layers):
            
            out_features, no_strides=cnn_birnn_trial_params.stack_layers[i]

            conv_layer = nn.Conv1d(
                #self.dataInputParams.input_features,
                input_features,
                out_features,
                kernel_size=kernel_size,
                stride=no_strides,
                bias=False,
            )

            input_shape, output_size = _check_layer_output(conv_layer, input_shape)

            if output_size < 3:
                kernel_size = 1

            setattr(cnn_bilstm_model, "cv" + str(i + 1), conv_layer)

            avgp_layer = nn.AvgPool1d(kernel_size=kernel_size, stride=no_strides)
            input_shape, output_size = _check_layer_output(avgp_layer, input_shape)
            if output_size < 3:
                kernel_size = 1

            setattr(cnn_bilstm_model, "avgp" + str(i + 1), avgp_layer)

            # setattr(cnn_bilstm_model,'maxp'+str(i+1), nn.MaxPool1d(kernel_size=1))
            setattr(cnn_bilstm_model, "bn" + str(i + 1), nn.BatchNorm1d(out_features))
            setattr(cnn_bilstm_model, "lrel" + str(i + 1), nn.LeakyReLU())

            input_features = out_features

        setattr(cnn_bilstm_model, "hidden_size_lstm", input_features * 2)

        rnn_drop_procentages = cnn_birnn_trial_params.rnn_drop_procentages
       
        if use_gru_instead_of_lstm:
            setattr(
                cnn_bilstm_model,
                "bi_gru",
                nn.GRU(
                    input_size=input_features,
                    hidden_size=input_features * 2,
                    num_layers=rnn_stacked_layers,
                    batch_first=True,
                    dropout=rnn_drop_procentages,
                    bidirectional=True,
                ),
            )
        else:
            setattr(
                cnn_bilstm_model,
                "bi_lstm",
                nn.LSTM(
                    input_size=input_features,
                    hidden_size=input_features * 2,
                    num_layers=rnn_stacked_layers,
                    batch_first=True,
                    dropout=rnn_drop_procentages,
                    bidirectional=True,
                ),
            )

        setattr(
            cnn_bilstm_model,
            "fc_output",
            nn.Linear(input_features * 4, self.dataInputParams.batch_size),
        )

        cnn_bilstm_model.apply(self._weights_init)

        return cnn_bilstm_model




                
