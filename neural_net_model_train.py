
import time

import torch
import torch.nn as nn 
 
import torchmetrics

import tqdm
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''  
losses = []
accuracy = []

def get_stats():
    return losses, accuracy 
'''

def train_net_model(net_model, optimizer, trainloader, MAX_EPOCHS,trial_optimisation = None, trial_parameter=None):    
    
    criterion = nn.CrossEntropyLoss().to(DEVICE) 

    losses = []
    losses_epoch= []
    
    accuracies = []
    accuracy = torchmetrics.Accuracy()
    accuracy = accuracy.to(DEVICE)  
   
    net_model.to(DEVICE)
    net_model.train()


    if trial_optimisation :  
        import optuna 

    for epoch in range(MAX_EPOCHS):
        timestart = time.time()
        correct = 0
        
        tickstep = 0
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):            

            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            outputs = net_model(batch_x)
            assert not torch.isnan(outputs).any()
            
            #loss = criterion(outputs, batch_y.long())  
            loss = criterion(outputs, batch_y.long())  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            _loss = np.around(loss.item(), decimals=3)
            losses_epoch.append(_loss)
            
            accuracy.update(outputs, batch_y.to(dtype=torch.int32))
            
        _acc_epoch = accuracy.compute()
        accuracies.append(_acc_epoch)    
        accuracy.reset()          
        
        if trial_optimisation and trial_parameter:  
            # Add prune mechanism
            if trial_parameter == 'accuracy':
                trial_optimisation.report(_acc_epoch, epoch)

                if trial_optimisation.should_prune():
                    raise optuna.exceptions.TrialPruned() 
                
            elif trial_parameter == 'loss':   
                trial_optimisation.report(1.0-max(losses_epoch), epoch)

                if trial_optimisation.should_prune():
                    raise optuna.exceptions.TrialPruned() 
            else:
                print('trial_parameter unknown')     
                  
        losses.extend(losses_epoch) 
        losses_epoch = []     
                    
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
    print('Finished Training')
    
    return losses, accuracy



def test_net_model(net_model, test_loader):
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(DEVICE)

    net_model.to(DEVICE)
    net_model.eval()    

    test_loss = 0
    correct = 0

    accuracy = torchmetrics.Accuracy()
    accuracy=accuracy.to(DEVICE)
    
    for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(test_loader),0):
        # send to device
        data, target = batch_x.to(DEVICE), batch_y.to(DEVICE)

        outputs = net_model(data)
        assert not torch.isnan(outputs).any()

        loss = criterion(outputs, target.long())
        test_loss += loss.detach().cpu()
        
        accuracy.update(outputs, target.to(dtype=torch.int32))

    test_loss /= len(test_loader.dataset)
    accuracy_result = accuracy.compute()


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy_result))
    
    return accuracy_result, test_loss






