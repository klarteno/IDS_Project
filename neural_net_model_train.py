
import time

import torch
import torch.nn as nn 
 
import torchmetrics

import tqdm
import numpy as np


if torch.cuda.is_available():
   torch.cuda.empty_cache() 
   
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE found: ', DEVICE) 





class ModelsTrainning:
    
    def setParameters(self,mlp_model, optimizer, train_loader,test_loader, max_epochs, trial_optimisation= None, trial_parameter=None):
        self.net_model = mlp_model 
        self.optimizer = optimizer
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.max_epochs = max_epochs
        self.trial_optimisation = trial_optimisation
        self.trial_parameter = trial_parameter
        
    def trainNetModel(self):    
        
        net_model = self.net_model
        optimizer = self.optimizer
        trainloader = self.train_loader
        MAX_EPOCHS = self.max_epochs
        trial_optimisation = self.trial_optimisation
        trial_parameter = self.trial_parameter 
        
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
            
            for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):            

                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                outputs_res = net_model(batch_x)
                # assert not torch.isnan(outputs_res).any()
                
                loss = criterion(outputs_res, batch_y.long())  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                _loss = np.around(loss.item(), decimals=3)
                losses_epoch.append(_loss)
                
                accuracy.update(outputs_res, batch_y.to(dtype=torch.int32))
                
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


    def getModelCheckpoint_AmpOptimized(self):
        checkpoint = {"model": self.net_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict()
                    }
        
        return checkpoint    
       
    def getModelCheckpoint(self):
        checkpoint = {"model": self.net_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }
        return checkpoint       
        
    # use if Tensor Core is present
    def trainNetModel_AmpOptimized(self):    
        self.scaler = scaler = torch.cuda.amp.GradScaler()
        
        net_model = self.net_model
        optimizer = self.optimizer
        trainloader = self.train_loader
        MAX_EPOCHS = self.max_epochs
        trial_optimisation = self.trial_optimisation
        trial_parameter = self.trial_parameter 
        
        
        
        criterion = nn.CrossEntropyLoss().to(DEVICE) 

        losses = []
        losses_epoch= []
        
        accuracies = []
        accuracy = torchmetrics.Accuracy()
        accuracy = accuracy.to(DEVICE)  
    
        net_model.to(DEVICE)
        net_model.train()

        # Necessary for FP16
        # scaler = torch.cuda.amp.GradScaler()

        if trial_optimisation :  
            import optuna 

        for epoch in range(MAX_EPOCHS):
            timestart = time.time()
            
            for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(trainloader),0):            

                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)          
                
                with torch.cuda.amp.autocast():
                    outputs_res = net_model(batch_x)
                    # assert not torch.isnan(outputs_res).any()      
                    loss = criterion(outputs_res, batch_y.long())  

                optimizer.zero_grad()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
        
                _loss = np.around(loss.item(), decimals=3)
                losses_epoch.append(_loss)
                
                accuracy.update(outputs_res, batch_y.to(dtype=torch.int32))
                
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


    def testNetModel(self):
        net_model = self.net_model
        test_loader = self.test_loader

        
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

        net_model.to(DEVICE)
        net_model.eval()    

        test_loss = 0
        correct = 0

        accuracy = torchmetrics.Accuracy()
        accuracy=accuracy.to(DEVICE)
        
        for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(test_loader),0):
            # send to device
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)       

            outputs = net_model(batch_x)
            # assert not torch.isnan(outputs).any()
            
            loss = criterion(outputs, batch_y.long())
            test_loss += loss.detach().cpu()
            
            accuracy.update(outputs, batch_y.to(dtype=torch.int32))

        test_loss /= len(test_loader.dataset)
        accuracy_result = accuracy.compute()


        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy_result))
        
        return accuracy_result, test_loss






