import torch
from torch import nn

import time

import tqdm
import torchmetrics
    
# Default constants

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# do not use because is not working on my laptop
def train_MLP_Model_faster(mlp_model, optimizer, trainloader,MAX_EPOCHS,trial=None):    
    
      
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    
    losses = []
    accuracy = []
    
    mlp_model.to(DEVICE)
    mlp_model.train()
    
    # Necessary for FP16
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(MAX_EPOCHS):
        timestart = time.time()
        correct = 0
        
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # batch_x = batch_x.view(-1, input_size)
            # batch_x = batch_x.float()
            
            # print(batch_x.size())
            # assert not torch.isnan(batch_x).any()
            with torch.cuda.amp.autocast():
                outputs = mlp_model(batch_x)
                assert not torch.isnan(outputs).any()
                loss = criterion(outputs, batch_y.long())
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        
            losses.append(loss.item())
        
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            
            accuracy.append(100.0 * correct / batch_y.size(0))
                    
        if trial :
            import optuna 
            # Add prune mechanism
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
    
    
    print('Finished Training')
    
    return losses, accuracy






# train the model for hyperptimization tunning
def train_MLP_Model_HypO(mlp_model, optimizer, trainloader, MAX_EPOCHS,trial=None):    
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    
    losses = []
    accuracies = []
    
    mlp_model.to(DEVICE)
    mlp_model.train()

    
    for epoch in range(MAX_EPOCHS):
        timestart = time.time()
        correct = 0
        
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            # batch_x = batch_x.view(-1, input_size)
            # batch_x = batch_x.float()
            
            # print(batch_x.size())
            # assert not torch.isnan(batch_x).any()
            
            outputs = mlp_model(batch_x)
            assert not torch.isnan(outputs).any()
            
            loss = criterion(outputs, batch_y.long())           
            loss.backward()
            optimizer.step()
    
            losses.append(loss.item())
        
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            
            accuracy = 100.0 * correct / batch_y.size(0)
            accuracies.append(accuracy)
                    
        if trial :
            import optuna 
            # Add prune mechanism
            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()            
                    
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
    
    print('Finished Training')
    
    return losses, accuracies






def train_MLP_Model(mlp_model, optimizer, trainloader, MAX_EPOCHS,trial_optimisation = None, trial_parameter=None):    
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    
    losses = []
    losses_epoch= []
    
    
    accuracies = []
    accuracy=torchmetrics.Accuracy()
    accuracy=accuracy.to(DEVICE)    
    
    mlp_model.to(DEVICE)
    mlp_model.train()

    if trial_optimisation :  
            import optuna 
    
    for epoch in range(MAX_EPOCHS):
        timestart = time.time()
        correct = 0
        
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE,dtype=torch.int32)
            
            # batch_x = batch_x.view(-1, input_size)
            # batch_x = batch_x.float()
            
            # print(batch_x.size())
            # assert not torch.isnan(batch_x).any()
            
            outputs = mlp_model(batch_x)
            assert not torch.isnan(outputs).any()
            
            loss = criterion(outputs, batch_y.long())           
            loss.backward()
            optimizer.step()
    
            losses_epoch.append(loss.item())
            accuracy.update(outputs,batch_y)
        
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
    
    return losses, accuracies





# accuracy_list = []
# loss_list = []

def test_mlp(model, test_loader):
    
    model.to(DEVICE)
    model.eval()    #torch.nn.Module.train() -> Sets the module in evaluation mode.

    test_loss = 0
    correct = 0
    
    accuracy = torchmetrics.Accuracy()
    accuracy=accuracy.to(DEVICE)

    
    for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(test_loader),0):
        # send to device
        data, target = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
        
        outputs = model(data)

        assert not torch.isnan(outputs).any()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        loss_fn = loss_fn.to(DEVICE)

        loss = loss_fn(outputs, target)
        test_loss += loss.detach().cpu()
        
        accuracy.update(outputs,target)

    test_loss /= len(test_loader.dataset)
    accuracy_result = accuracy.compute()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy_result))
    return accuracy_result





def test_mlp_backup(model, test_loader):
    
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

