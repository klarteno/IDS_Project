# Imports
import torch
import torch.nn as nn 

import tqdm
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_groups = 4 

# model that takes long to train: more than 600 minutes
class CNN_BILSTM(nn.Module):
    def __init__(self, input_size, num_classes,kernel_size= 1, n_layers=2):
        super(CNN_BILSTM, self).__init__()
        
        self.input_size = input_size
        self.n_layers = n_layers
        self.num_classes = num_classes

        self.cv1 = nn.Conv1d(input_size, 128, kernel_size= kernel_size)
        self.gn1 = nn.GroupNorm(num_groups, 128)
        self.rel1 = nn.ReLU()
        self.p1 = nn.MaxPool1d(kernel_size= kernel_size)
        
        self.cv2 = nn.Conv1d(128, 128,kernel_size= kernel_size)
        self.gn2 = nn.GroupNorm(num_groups, 128)
        self.rel2 = nn.ReLU()
        self.p2 = nn.MaxPool1d(kernel_size= kernel_size)
        
        self.cv3 = nn.Conv1d(128, 256,kernel_size= kernel_size)
        self.gn3 = nn.GroupNorm(num_groups, 256)
        self.rel3 = nn.ReLU()
        self.p3 = nn.MaxPool1d(kernel_size= kernel_size)
        
        self.cv4 = nn.Conv1d(256, 256,kernel_size= kernel_size)
        self.gn4 = nn.GroupNorm(num_groups, 256)
        self.rel4 = nn.ReLU()
        self.p4 = nn.MaxPool1d(kernel_size= kernel_size)
                
        self.hidden_size_lstm = 256
        
        self.bilstm = nn.LSTM(
            input_size=1, hidden_size = self.hidden_size_lstm, num_layers=n_layers, batch_first=True,  dropout=0.01, bidirectional=True
        )

        self.fc = nn.Linear(self.hidden_size_lstm*2 , num_classes)


        self.initialize_weights()

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, inputs, hidden=None):        
        
        # Run through CNN net: (batch_size x input_size x seq_len) 
        inputs1=inputs.unsqueeze(2)
        
        out1 = self.cv1(inputs1)
        assert not torch.isnan(out1).any()
        out = self.gn1(out1)
        out = self.rel1(out)
        out = self.p1(out)
        
        out2 = self.cv2(out)
        assert not torch.isnan(out2).any()
        out = self.gn2(out2)
        out = self.rel2(out)
        out = self.p2(out)
        
        out3 = self.cv3(out)
        assert not torch.isnan(out3).any()
        out = self.gn3(out3)
        out = self.rel3(out)
        out = self.p3(out)
        
        out4 = self.cv4(out)
        assert not torch.isnan(out4).any()
        out = self.gn4(out4)
        out = self.rel4(out)
        out = self.p4(out)
                
        # make data as (seq_len x batch_size x hidden_size) for RNN
        # out = out.transpose(1, 2).transpose(0, 1)
        
        h0 = torch.zeros(self.n_layers * 2, out.size(0), self.hidden_size_lstm).to(DEVICE)
        c0 = torch.zeros(self.n_layers * 2, out.size(0), self.hidden_size_lstm).to(DEVICE)

        
        out1 = out.squeeze(2)

        output1, hidden = self.bilstm(out, (h0, c0))
        assert not torch.isnan(output1).any()

        output2 = output1[:, -1, :]
        
        output = self.fc(output2)
        assert not torch.isnan(output).any()
        
        return output
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    




import time


losses = []
accuracy = []

def get_stats():
    return losses, accuracy 

def train_CNN_BILSTM_Model(cnn_bilstm_model, optimizer, trainloader,MAX_EPOCHS):    
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 

    cnn_bilstm_model.to(DEVICE)
    cnn_bilstm_model.train()

    for epoch in range(MAX_EPOCHS):
        timestart = time.time()
        correct = 0
        
        tickstep = 0
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):            

            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            outputs = cnn_bilstm_model(batch_x)
            assert not torch.isnan(outputs).any()
            
            loss = criterion(outputs, batch_y.long())  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    
            _loss = np.around(loss.item(), decimals=3)
    
            losses.append(_loss)

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            
            accuracy.append(100.0 * correct / batch_y.size(0))
                    
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
    print('Finished Training')
    return losses, accuracy



def test_CNN_BILSTM_model(model, test_loader):
    
    model.to(DEVICE)
    model.eval()    #torch.nn.Module.train() -> Sets the module in evaluation mode.

    test_loss = 0
    correct = 0
    for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(test_loader),0):
        # send to device
        data, target = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)

        outputs = model(data)

        assert not torch.isnan(outputs).any()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        loss_fn = loss_fn.to(DEVICE)

        loss = loss_fn(outputs, target)
        test_loss += loss.detach().cpu()
        # print(test_loss)
        # test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item() # sum up batch loss                                                               

        pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
#    accuracy_list.append(accuracy)
#    loss_list.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

