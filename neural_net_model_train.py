import torch
import torch.nn as nn

import torchmetrics

#from tqdm import tqdm
from tqdm.auto import tqdm
#from tqdm.notebook import tqdm

import numpy as np

import os, random


if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE found: ", DEVICE)
number_of_classes=15
print("number of classes: ", number_of_classes)



def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything()

class ModelsEvaluations:
    
    def __init__(self):
        self.accuracies_scores = []
        self.accuracy = torchmetrics.Accuracy().to(DEVICE)

        self.f1_scores = []
        self.f1_score = torchmetrics.F1Score(num_classes=number_of_classes, average="weighted").to(DEVICE)
        
        self.auroc_scores = []
        self.auroc = torchmetrics.AUROC(num_classes=number_of_classes).to(DEVICE)
        
        self.mean_metric = torchmetrics.MeanMetric(nan_strategy="warn")

        
    def updateEvaluationEpochStep(self, outputs_res, batch_y):
        self.accuracy.update(outputs_res, batch_y.to(dtype=torch.int32))
        self.f1_score.update(outputs_res, batch_y.to(dtype=torch.int32))
        self.auroc.update(outputs_res, batch_y.to(dtype=torch.int32))
    
    def getEvaluationEpoch(self):
        _accuracy_epoch = self.accuracy.compute()
        self.accuracies_scores.append(_accuracy_epoch)
        self.accuracy.reset()    
        
        _f1_score_epoch = self.f1_score.compute()
        self.f1_scores.append(_f1_score_epoch)
        self.f1_score.reset() 
        
        _auroc_score_epoch = self.auroc.compute()
        self.auroc_scores.append(_auroc_score_epoch)
        self.auroc.reset()
        
        return _accuracy_epoch,  _f1_score_epoch, _auroc_score_epoch
    
    def getEvaluationModelTrainning(self):
        return self.accuracies_scores, self.f1_scores , self.auroc_scores
    
    def reset(self):
        self.accuracies_scores = []    
        self.f1_scores = []
        self.auroc_scores = []

    def computeMean(self, values):
        self.mean_metric.update(values)
        mean_value = self.mean_metric.compute()
        self.mean_metric.reset()

        return mean_value    


class ModelsTrainning:
    
    def __init__(self, train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        max_epochs):
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
    
    def setParameters(
        self,
        net_model: nn.Module,
        optimizer,
        trial_optimisation=None,
        evaluation_function=None,
    ):
        self.net_model = net_model
        self.optimizer = optimizer
        self.trial_optimisation = trial_optimisation
        self.evaluation_function = evaluation_function
        
        self.modelsEvaluations = ModelsEvaluations()
        
        
    def prune_trial(self, epoch, accuracy_epoch, losses_epoch, f1_score_epoch):
        import optuna

        # Add prune mechanism
        if self.trial_optimisation and self.evaluation_function:
            if self.evaluation_function == "accuracy":
                self.trial_optimisation.report(accuracy_epoch, epoch)

                if self.trial_optimisation.should_prune():
                    raise optuna.exceptions.TrialPruned()

            elif self.evaluation_function == "loss":
                self.trial_optimisation.report(1.0 - losses_epoch, epoch)

                if self.trial_optimisation.should_prune():
                    raise optuna.exceptions.TrialPruned()

            elif self.evaluation_function == "f1_score":
                self.trial_optimisation.report(f1_score_epoch, epoch)

                if self.trial_optimisation.should_prune():
                    raise optuna.exceptions.TrialPruned()
            else:
                 print("test_evaluation_function unknown")

    def get_evaluation_result(self,accuracy_score,loss_score,f1_score):
        if self.evaluation_function == "accuracy":
            return accuracy_score

        elif self.evaluation_function == "loss":
            return loss_score

        elif self.evaluation_function == "f1_score":
            return f1_score


    def trainNetModel(self):

        MAX_EPOCHS = self.max_epochs

        criterion = nn.CrossEntropyLoss().to(DEVICE)

        losses = []
        losses_epoch = []

        self.net_model.to(DEVICE)
        self.net_model.train()

        for epoch in range(MAX_EPOCHS):
            for step, (batch_x, batch_y) in enumerate(tqdm(self.train_loader, position=0, leave=True), 0):
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(
                    DEVICE, non_blocking=True
                )

                outputs_res = self.net_model(batch_x)
                # assert not torch.isnan(outputs_res).any()

                loss = criterion(outputs_res, batch_y.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_epoch.append(np.around(loss.item(), decimals=3))
                
                self.modelsEvaluations.updateEvaluationEpochStep(outputs_res, batch_y)

            _accuracy_epoch,  _f1_score_epoch, _auroc_score_epoch = self.modelsEvaluations.getEvaluationEpoch()
           
           
            _losses_epoch = self.modelsEvaluations.computeMean(losses_epoch)
            losses_epoch=[]
            
            # Add prune mechanism
            self.prune_trial(epoch, _accuracy_epoch, _losses_epoch, _f1_score_epoch)

            losses.append(_losses_epoch)
        
        accuracies_scores, f1_scores, auroc_scores = self.modelsEvaluations.getEvaluationModelTrainning()
        self.modelsEvaluations.reset()    
            
        return accuracies_scores, losses, f1_scores, auroc_scores


    def getModelCheckpoint_AmpOptimized(self):
        checkpoint = {
            "model": self.net_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        return checkpoint

    def getModelCheckpoint(self):
        checkpoint = {
            "model": self.net_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return checkpoint

    # use if Tensor Core is present
    def trainNetModel_AmpOptimized(self):
        # Necessary for FP16
        self.scaler = scaler = torch.cuda.amp.GradScaler()

        MAX_EPOCHS = self.max_epochs

        criterion = nn.CrossEntropyLoss().to(DEVICE)

        losses = []
        losses_epoch = []

        self.net_model.to(DEVICE)
        self.net_model.train()

        for epoch in range(MAX_EPOCHS):

            for step, (batch_x, batch_y) in enumerate(tqdm(self.train_loader, position=0, leave=True), 0):

                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(
                    DEVICE, non_blocking=True
                )

                with torch.cuda.amp.autocast():
                    outputs_res = self.net_model(batch_x)
                    # assert not torch.isnan(outputs_res).any()
                    loss = criterion(outputs_res, batch_y.to(dtype=torch.long))

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                losses_epoch.append(np.around(loss.item(), decimals=3))
            
                self.modelsEvaluations.updateEvaluationEpochStep(outputs_res, batch_y)

            _accuracy_epoch,  _f1_score_epoch, _auroc_score_epoch = self.modelsEvaluations.getEvaluationEpoch()
            
            _losses_epoch = self.modelsEvaluations.computeMean(losses_epoch)
            losses_epoch=[]
            
            # Add prune mechanism
            self.prune_trial(epoch, _accuracy_epoch, _losses_epoch, _f1_score_epoch)
            
            losses.append(_losses_epoch)
        
        accuracies_scores, f1_scores, auroc_scores = self.modelsEvaluations.getEvaluationModelTrainning()
        self.modelsEvaluations.reset() 
        
        return accuracies_scores, losses, f1_scores, auroc_scores




    def testNetModel(self):

        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

        self.net_model.to(DEVICE)
        self.net_model.eval()

        losses_epoch = []

        for step, (batch_x, batch_y) in enumerate(tqdm(self.test_loader, position=0, leave=True), 0):

            # send to device
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(
                DEVICE, non_blocking=True
            )

            outputs_res = self.net_model(batch_x)
            # assert not torch.isnan(outputs).any()
            loss = criterion(outputs_res, batch_y.long())

            losses_epoch.append(np.around(loss.item(), decimals=3))
            
            self.modelsEvaluations.updateEvaluationEpochStep(outputs_res, batch_y)

            
        self.modelsEvaluations.getEvaluationEpoch()
        
        loss_score = self.modelsEvaluations.computeMean(losses_epoch)
        losses_epoch=[]
        
        accuracies_scores, f1_scores, auroc_scores = self.modelsEvaluations.getEvaluationModelTrainning()
        self.modelsEvaluations.reset() 
        
        
        print(
            "\nTest set: Average loss: {:.4f}, Average accuracy: {:.4f}%,\n \t  Average f1_score: {:.4f}, Average Area Under ROC: {:.4f} \n".format(
                loss_score.item(), accuracies_scores[0].item(), f1_scores[0].item() , auroc_scores[0].item()
            )
        )
        
        return self.get_evaluation_result(accuracies_scores[0],
                                          loss_score,
                                          f1_scores[0])



def testNetModel(net_model, test_loader, modelsEvaluations = ModelsEvaluations()):

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    net_model.to(DEVICE)
    net_model.eval()

    losses_epoch = []

    for step, (batch_x, batch_y) in enumerate(tqdm(test_loader, position=0, leave=True), 0):

        # send to device
        batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(
            DEVICE, non_blocking=True
        )

        outputs_res = net_model(batch_x)
        # assert not torch.isnan(outputs).any()
        loss = criterion(outputs_res, batch_y.long())

        losses_epoch.append(np.around(loss.item(), decimals=3))
        
        modelsEvaluations.updateEvaluationEpochStep(outputs_res, batch_y)

    modelsEvaluations.getEvaluationEpoch()
    
    loss_score = modelsEvaluations.computeMean(losses_epoch)
    losses_epoch=[]
    
    accuracies_scores, f1_scores, auroc_scores = modelsEvaluations.getEvaluationModelTrainning()
    modelsEvaluations.reset() 

    return loss_score.item(), accuracies_scores[0].item(), f1_scores[0].item() , auroc_scores[0].item()
    
    
