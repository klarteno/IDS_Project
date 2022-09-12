import torch
import numpy as np
import pandas as pd
import sys
import os

from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder #, OneHotEncoder



if torch.cuda.is_available():
   torch.cuda.empty_cache() 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE: ', DEVICE) 

import random, torch, os, numpy as np




def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()




# Store All CSV Filenames in the Dataset in an Array
csv_files = []
for dirname, _, filenames in os.walk('C:/Users/androgo/Documents/Python Scripts/IDS/CICIDS2017_notebooks/IDS_Project/datasets/MachineLearningCSV/MachineLearningCVE'):
  # yelds a 3-tuple (dirpath, (sub)dirnmes, filenames)
    for filename in filenames:
        csv_file = os.path.join(dirname, filename)
        csv_files.append(csv_file)


        
        
week_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)



week_data.columns = week_data.columns.str.strip()
week_data.replace([np.inf, -np.inf],  np.nan, inplace=True)  # replace -infinity and+infinity with NaN
week_data.dropna(inplace=True)    #remove missing values
# print("Length of week_data after droping null values:", len(week_data))



y = week_data.Label
X = week_data.drop(columns='Label')
class_labels = y.unique()
num_classes = y.nunique()     # number of unique values
print("shape of X: ",X.shape)
print("number of labels of y: ", num_classes)
print("Class labels: ", class_labels)
print()




selected_features = ["Bwd Packet Length Min", "Subflow Fwd Bytes", "Total Length of Fwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Std", "Flow IAT Min", "Fwd IAT Min", "Flow IAT Mean", "Flow Duration", "Flow IAT Std", "Init_Win_bytes_forward", "Active Min", "Active Mean", "Bwd Packets/s", "Bwd IAT Mean", "Fwd IAT Mean", "ACK Flag Count", "Fwd PSH Flags", "SYN Flag Count", "Fwd Packets/s", "Init_Win_bytes_backward", "PSH Flag Count", "Average Packet Size"]
X_select = X[np.intersect1d(X.columns, selected_features)]



X_train, X_test, y_train, y_test = train_test_split(X_select, y, random_state=99, stratify=y)
print("\n After spliting the data:")
print("training data shape:", X_train.shape)
print("test data shape:", X_test.shape)
print("training data shape:", y_train.shape)
print("test data shape:", y_test.shape)



le = LabelEncoder()       # Encode target labels with value between 0 and n_classes-1
y_train_binary = le.fit_transform(y_train)
#print("instances per label in test set\n", y_test_binary.value_counts())
# transform -	Transform labels to normalized encoding.
y_test_binary = le.transform(y_test)
#we use fit_transform() on training data but transform() on the test data

# classes_ - ndarray of shape (n_classes,) - Holds the label for each class.
# To create a dictionary from two sequences, use dict(zip(keys, values))
# The zip(fields, values) method returns an iterator that generates two-items tuples 
labels_dict = dict(zip(le.classes_, range(len(le.classes_))))




X_train_sampled = pd.read_csv('datasets/X_train_sampled_23_features.csv')
y_train_binary_sampled = np.loadtxt('datasets/y_train_binary_sampled_23_features.csv', delimiter=',')


# Check the files loaded because extra columns gets added
# if  len(X_train_sampled.columns)==79:  
if  len(X_train_sampled.columns) == 24:
    X_train_sampled['Unnamed: 0']  
    X_train_sampled = X_train_sampled.drop(columns='Unnamed: 0')
    



scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_sampled)
X_test = scaler.transform(X_test)    


class SecurityDataset(torch.utils.data.Dataset):

  def __init__(self, X_train, y_train_binary):
    
    self.X_train = torch.tensor(X_train, dtype = torch.float32).clone().detach()
    self.Y_train = torch.tensor(y_train_binary).clone().detach()

  def __len__(self):
    return len(self.Y_train)
  
  def __getitem__(self, index):
    return self.X_train[index], self.Y_train[index]
  
  
  
train_dataset = SecurityDataset(torch.tensor(X_train_sampled.values), y_train_binary_sampled)
test_dataset = SecurityDataset(X_test, y_test_binary)  


### Class Weights
from sklearn.utils import class_weight

classes_y = np.array(list(labels_dict.values()))
#calculate the class weights
class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = classes_y, # np.unique(y_train_binary),
                                                 y = y_train_binary_sampled)
classes_class_weights = dict(zip(classes_y, class_weights))
class_weights = np.around(class_weights, decimals=3)



##### Apply WeightedRandomSampler only to train data and leave test or validate data untouched because is treated as unseen
weights_sampler = 1. / class_weights
sample_weights = [0] * len(train_dataset)
# weights_sampler =np.around(weights_sampler, decimals=5)
for idx, (data, label) in enumerate(train_dataset):
        class_weight = class_weights[ int(label.item()) ]
        sample_weights[idx] = class_weight   
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=
                                    len(sample_weights), replacement=True)



def get_input_size():
    return X_train.shape[1]
  
def get_number_of_classes():
    return num_classes

def get_dataloaders(batch_size = 32):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    
    return train_loader,test_loader


