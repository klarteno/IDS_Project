import torch
from torch import nn

# Default constants
# LEARNING_RATE = 1e-2

EVAL_FREQ = 10
BATCH_SIZE = 32

# %reload_ext autoreload
# %autoreload 2

num_groups = 4


class MLP_Model(nn.Module):
    def __init__(self, inputSize, numClasses):
        super(MLP_Model, self).__init__()

        self.fltn = nn.Flatten()
        self.fc1 = nn.Linear(inputSize, 128)
        # self.bc1 = nn.BatchNorm1d(128)
        self.bc1 = nn.GroupNorm(num_groups, 128)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, 128)
        self.bc2 = nn.GroupNorm(num_groups, 128)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc21 = nn.Linear(128, 128)
        self.bc21 = nn.GroupNorm(num_groups, 128)
        self.relu21 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(128, 64)
        self.bc3 = nn.GroupNorm(num_groups, 64)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(64, 64)
        self.bc4 = nn.GroupNorm(num_groups, 64)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc5 = nn.Linear(64, numClasses)

        self.initialize_weights()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.logsm = nn.LogSoftmax(dim=1)

    def forward(self, x):

        # out = self.fc1(self.fltn(x))
        out = self.fc1(x)
        assert not torch.isnan(out).any()
        out = self.bc1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out2 = self.fc2(out)
        assert not torch.isnan(out2).any()
        out = self.bc2(out2)
        out = self.relu2(out)
        out = self.dropout(out)

        out21 = self.fc21(out)
        assert not torch.isnan(out21).any()
        out = self.bc21(out21)
        out = self.relu21(out)
        out = self.dropout(out)

        out3 = self.fc3(out)
        assert not torch.isnan(out3).any()
        out3 = self.bc3(out3)
        out = self.relu3(out3)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.bc4(out)
        out = self.relu4(out)
        out4 = self.dropout(out)
        assert not torch.isnan(out4).any()

        out5 = self.fc5(out4)
        assert not torch.isnan(out5).any()

        # out = self.logsm(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
