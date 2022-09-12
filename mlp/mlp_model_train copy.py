import torch
from torch import nn

import time

import tqdm

    
# Default constants

EVAL_FREQ = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# do not use because is not working
def train_MLP_Model_faster(mlp_model, optimizer, trainloader,MAX_EPOCHS):    
    
      
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
                    
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
     
    print('Finished Training')
    
    return losses, accuracy



def train_MLP_Model(mlp_model, optimizer, trainloader,MAX_EPOCHS):    
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    
    losses = []
    accuracy = []
    
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
            
            accuracy.append(100.0 * correct / batch_y.size(0))
                    
        print('epoch %d cost %3f sec' %(epoch, time.time()-timestart))
    print('Finished Training')
    return losses, accuracy





# accuracy_list = []
# loss_list = []

def test_mlp(model, test_loader):
    
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
