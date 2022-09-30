import torch
import numpy as np
import pandas as pd
import os

from torch.utils.data import DataLoader

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)  # , OneHotEncoder

import pickle
from pprint import pprint

print(os.getcwd())

# os.chdir('./Users/bredsoby')

# Load data with 78 
# The data with 78 features takes very long to train on Colab an locally
#load_78_features = True
# Load data with  23 features
load_78_features = False



# loads the whole datasets ,the code will run slowly
def _load_datasets(load_78_features=True, n_rows=None):
    # Load 78 features
    if load_78_features is True:
        # read  500 rows for testing
        X_train__78_features = pd.read_csv(
            "datasets/X_train_78_features_sampled.csv", nrows=n_rows
        )
        if (
            len(X_train__78_features.columns) == 79
        ):  # Check the files loaded because extra columns gets added
            X_train__78_features = X_train__78_features.drop(columns="Unnamed: 0")
        print("X_train__78_features", X_train__78_features.shape)

        X_test__78_features = pd.read_csv(
            "datasets/X_test_78_features.csv", nrows=n_rows
        )
        if len(X_test__78_features.columns) == 79:
            X_test__78_features = X_test__78_features.drop(columns="Unnamed: 0")
        print("X_test__78_features", X_test__78_features.shape)

        # max_rows=500,
        Y_train_binary__78_features = np.loadtxt(
            "datasets/y_train_binary_78_features_sampled.csv",
            max_rows=n_rows,
            delimiter=",",
        )
        print("Y_train_binary__78_features", Y_train_binary__78_features.shape)
        print(pd.DataFrame(Y_train_binary__78_features).value_counts())

        Y_test__binary = np.loadtxt(
            "datasets/y_test_binary.csv", max_rows=n_rows, delimiter=","
        )
        print("Y_test_binary", Y_test__binary.shape)

        return (
            X_train__78_features,
            X_test__78_features,
            Y_train_binary__78_features,
            Y_test__binary,
        )

    # Load 23 features
    if load_78_features is False:

        X_train__23_features = pd.read_csv(
            "datasets/X_train_23_features_sampled.csv", nrows=n_rows
        )
        if len(X_train__23_features.columns) == 24:
            X_train__23_features = X_train__23_features.drop(columns="Unnamed: 0")
        print("X_train__23_features", X_train__23_features.shape)

        X_test__23_features = pd.read_csv(
            "datasets/X_test_23_features.csv", nrows=n_rows
        )
        if len(X_test__23_features.columns) == 24:
            X_test__23_features = X_test__23_features.drop(columns="Unnamed: 0")
        print("X_test__23_features", X_test__23_features.shape)

        Y_train_binary__23_features = np.loadtxt(
            "datasets/y_train_binary_23_features_sampled.csv",
            max_rows=n_rows,
            delimiter=",",
        )

        print("y_train_binary__23_features", Y_train_binary__23_features.shape)
        pprint(pd.DataFrame(Y_train_binary__23_features).value_counts())

        Y_test__binary = np.loadtxt(
            "datasets/y_test_binary.csv", max_rows=n_rows, delimiter=","
        )
        print("Y_test__binary", Y_test__binary.shape)

        return (
            X_train__23_features,
            X_test__23_features,
            Y_train_binary__23_features,
            Y_test__binary,
        )       


def load_datasets(load_for_testing=False, n_rows=None):
    if load_for_testing:

        X_train, X_test, Y_train, Y_test = _load_datasets(
            load_78_features=load_78_features, n_rows=n_rows
        )
        
        # when Y datasets are loaded in small amount we get zeros because zeros are majority
        # print('is zero: ', np.all((Y_train == 0)))
        # to get around this we generate some Y data
        Y_train = np.random.randint(0, 15, size=n_rows)
        Y_test = np.random.randint(0, 15, size=n_rows)

        le = LabelEncoder()  # Encode target labels with value between 0 and n_classes-1
        Y_train_binary = le.fit_transform(Y_train)
        Y_test_binary = le.transform(Y_test)
        labels_dict = dict(zip(le.classes_, range(len(le.classes_))))
    
        return X_train, X_test, Y_train_binary, Y_test_binary, labels_dict

    else:
        X_train, X_test, Y_train, Y_test = _load_datasets(
            load_78_features=load_78_features, n_rows=None
        )
        labels_dict = pickle.load(open("datasets/labels_dict_file.pkl", "rb"))

        return X_train, X_test, Y_train, Y_test, labels_dict


# load_for_testing: load a small part of the dataset for testing,debuging the code
X_train, X_test, Y_train, Y_test, labels_dict = load_datasets(load_for_testing=True, n_rows=1500)

# load the whole datasets available
#X_train, X_test, Y_train, Y_test, labels_dict  = load_datasets()


# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test.to_numpy())


class SecurityDataset(torch.utils.data.Dataset):
    def __init__(
        self, X_train, y_train, transform=torch.tensor, target_transform=torch.tensor
    ):
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)

        self.transform = transform
        self.target_transform = target_transform

        if self.transform:
            self.X_train = self.transform(X_train, dtype=torch.float32)
        if self.target_transform:
            self.Y_train = self.target_transform(y_train, dtype=torch.int64)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, index):
        feature = torch.index_select(self.X_train, 0, torch.tensor([index]))
        label = torch.index_select(self.Y_train, 0, torch.tensor([index]))

        return feature, label


train_dataset = SecurityDataset(X_train, Y_train)
test_dataset = SecurityDataset(X_test, Y_test)


from sklearn.utils import class_weight


classes_y = np.array(list(labels_dict.values()))
print("classes_y: ", classes_y)
# calculate the class weights
class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=classes_y, y=Y_train  # np.unique(y_train_binary),
)
print("class_weights: ", class_weights)
print()
# class_weights.round(decimals=3, out=None)
class_weights = np.around(class_weights, decimals=3)
classes_class_weights = dict(zip(classes_y, class_weights))
print("classes_class_weights: ")
pprint(classes_class_weights)


weights_sampler = 1.0 / class_weights
sample_weights = [0] * len(train_dataset)
# weights_sampler =np.around(weights_sampler, decimals=5)
for idx, (data, label) in enumerate(train_dataset):
    class_weight = class_weights[int(label.item())]
    sample_weights[idx] = class_weight
sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)


batch_size = 32
# use num_workers declared when the code is run on the cloud and it runs faster
'''train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    drop_last=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=os.cpu_count(),
    pin_memory=True
)
'''

# use without num_workers declared when the code is run loccaly(on laptop) because it gives errors
train_loader = DataLoader( dataset=train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)


def get_input_size():
    return X_train.shape[1]


def get_number_of_classes():
    return len(labels_dict)


def get_dataloaders():
    return train_loader, test_loader
