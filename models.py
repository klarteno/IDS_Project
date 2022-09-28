import optuna
import torch
import torch.nn as nn
import os


from torchinfo import summary


if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ModelsInterface:
    def define_model(self, trial: optuna.Trial):
        print("ModelsInterface define_model")

    def saveModel(self, checkpoint: dict, path_models_id:str, number: int):
        torch.save(checkpoint, path_models_id + str(number))      


def _check_layer_output(layer, input_shape):
    _summary = summary(layer, input_size=input_shape)
    input_shape = _summary.summary_list[0].output_size

    return input_shape, input_shape[-1]


def _check_nans(outputs):
    if torch.isnan(outputs).any():
        print("NANs prunned")
        raise optuna.exceptions.TrialPruned()


class MlpModel(ModelsInterface):
    def __init__(self, args):
        self.args = args

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    # Build a model by implementing define-by-run design from Optuna
    def define_model(self, trial: optuna.Trial):
        number_of_classes = self.args["number_of_classes"]
        input_shape = self.args["input_shape"]
        input_features=input_shape[2]
        
        batch_size = input_shape[0]

        no_layers = trial.suggest_int("n_layers", 2, 9)

        def init_func(self):
            super(NetModel, self).__init__()
            
        @torch.cuda.amp.autocast()
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
            
            outputs = outputs.view(batch_size, -1)
            linear_output = (nn.Linear(outputs.shape[1], number_of_classes)).to(DEVICE)
            
            # apply Linear to get shape of: (batch_size, number_of_classes)
            # the Linear is generated in this way to make the output fit the required shape and to let RNN layer be as wide as possible
            outputs = linear_output(outputs)
            
            # reshaped because  has to match the labels target
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

        mlp_model: NetModel() = NetModel()

        setattr(mlp_model, "flat", nn.Flatten())

        for i in range(no_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 25, 256, step=3)

            lin_layer = nn.Linear(input_features, out_features)
            input_shape, output_size = _check_layer_output(lin_layer, input_shape)

            setattr(mlp_model, "lin" + str(i + 1), lin_layer)
            setattr(mlp_model, "bn" + str(i + 1), nn.BatchNorm1d(out_features))
            setattr(mlp_model, "lrel" + str(i + 1), nn.LeakyReLU())

            drop_procentages = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5, log=True)
            setattr(mlp_model, "drop" + str(i + 1), nn.Dropout(drop_procentages))

            input_features = out_features

        setattr(mlp_model, "fc_output", nn.Linear(input_features, batch_size))

        mlp_model.apply(self._weights_init)

        return mlp_model


class CnnBirnnModel(ModelsInterface):
    def __init__(self, args):
        self.args = args

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
    def define_model(self, trial):

        number_of_classes = self.args["number_of_classes"]
        input_shape = self.args["input_shape"]
        use_gru_instead_of_lstm = self.args["use_gru_instead_of_lstm"]

        batch_size = input_shape[0]
        input_features = input_shape[1] 


        no_layers = trial.suggest_int("n_layers", 2, 9)

        # to be optimized by optuna or no because the hidden size is big?
        rnn_stacked_layers = 2

        def init_func(self):
            super(NetModel, self).__init__()

        @torch.cuda.amp.autocast()
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
            
            outputs = outputs.view(batch_size, -1)
            linear_output = (nn.Linear(outputs.shape[1], number_of_classes)).to(DEVICE)
            
            # apply Linear to get shape of: (batch_size,number_of_classes)
            # the Linear is generated in this way to make the output fit the required shape and to let RNN layer be as wide as possible
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

        for i in range(no_layers):

            out_features = trial.suggest_int("n_units_l{}".format(i), 25, 256, step=3)
            no_strides = trial.suggest_int("no_strides_l{}".format(i), 1, 7, step=2)

            conv_layer = nn.Conv1d(
                input_features, out_features, kernel_size=kernel_size, stride=no_strides,bias=False
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

        drop_procentages = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5, log=True)

        if use_gru_instead_of_lstm:
            setattr(
                cnn_bilstm_model,
                "bi_gru",
                nn.GRU(
                    input_size=input_features,
                    hidden_size=input_features * 2,
                    num_layers=rnn_stacked_layers,
                    batch_first=True,
                    dropout=drop_procentages,
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
                    dropout=drop_procentages,
                    bidirectional=True,
                ),
            )

        setattr(
            cnn_bilstm_model,
            "fc_output",
            nn.Linear(input_features * 4, batch_size),
        )

        cnn_bilstm_model.apply(self._weights_init)

        return cnn_bilstm_model