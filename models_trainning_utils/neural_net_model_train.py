from pickle import TRUE
from re import A
import torch
import torch.utils.data

import torch.nn as nn

import torchmetrics
from torchmetrics import classification


import optuna

# from tqdm import tqdm
from tqdm.auto import tqdm

# from tqdm.notebook import tqdm

import numpy as np

import os
import random

import torch


if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE found: ", DEVICE)
number_of_classes = 15
print("number of classes: ", number_of_classes)

# use for crossentropy classification because the output is a vector of probabilities that hardly can be compared with the target vector of labels when one-hot encoded makes these integers from 0 to 14 (15 classes)
LABEL_SMOOTHING = 0.015


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
        # average='weighted' is for handling imbalanced balanced dataset
        self.device_for_metrics = "cpu"
        kwargs = {"compute_on_cpu": True}  # if use_cuda else {}

        self.multi_class_accuracies_trainning = []
        self.multi_accuracy_epoch = classification.MulticlassAccuracy(
            num_classes=number_of_classes,
            average=None,
            multidim_average="global",
            **kwargs
        ).to(self.device_for_metrics)

        self.accuracies_trainning = []
        self.accuracy_epoch = torchmetrics.Accuracy(
            num_classes=number_of_classes,
            average="weighted",
            multidim_average="global",
            **kwargs
        ).to(self.device_for_metrics)

        self.f1_scores_trainning = []
        self.f1_score_epoch = torchmetrics.F1Score(
            num_classes=number_of_classes, average="weighted", **kwargs
        ).to(self.device_for_metrics)

        self.auroc_scores_trainning = []
        self.auroc_epoch = torchmetrics.AUROC(
            num_classes=number_of_classes, average="weighted", **kwargs
        ).to(self.device_for_metrics)

        self.mean_metric = torchmetrics.MeanMetric(nan_strategy="warn", **kwargs).to(
            self.device_for_metrics
        )

        self.scheduler_learning_history = []

        self.accuracies_testset = []
        self.accuracy_testset = torchmetrics.Accuracy(
            num_classes=number_of_classes,
            average="weighted",
            multidim_average="global",
            **kwargs
        ).to(self.device_for_metrics)

    def update_evaluation_epoch(self, outputs_res: torch.Tensor, batch_y: torch.Tensor):
        with torch.no_grad():
            _outputs_res = outputs_res.to(self.device_for_metrics)
            _batch_y = batch_y.to(self.device_for_metrics, dtype=torch.int32)

            self.multi_accuracy_epoch.update(_outputs_res, _batch_y)
            self.accuracy_epoch.update(_outputs_res, _batch_y)
            self.f1_score_epoch.update(_outputs_res, _batch_y)
            self.auroc_epoch.update(_outputs_res, _batch_y)

    def get_evaluation_epoch(self):

        multi_accuracy_epoch = self.multi_accuracy_epoch.compute()
        self.multi_accuracy_epoch.reset()
        self.multi_class_accuracies_trainning.append(multi_accuracy_epoch)

        accuracy_epoch = self.accuracy_epoch.compute()
        accuracy_epoch = accuracy_epoch.item() * 100
        self.accuracy_epoch.reset()
        self.accuracies_trainning.append(accuracy_epoch)

        f1_score_epoch = self.f1_score_epoch.compute()
        f1_score_epoch = f1_score_epoch.item() * 100
        self.f1_score_epoch.reset()
        self.f1_scores_trainning.append(f1_score_epoch)

        auroc_score_epoch = self.auroc_epoch.compute()
        auroc_score_epoch = auroc_score_epoch.item()
        self.auroc_epoch.reset()
        self.auroc_scores_trainning.append(auroc_score_epoch)

        return accuracy_epoch, f1_score_epoch, auroc_score_epoch

    def get_multi_class_accuracies_trainning_score(self):
        with torch.no_grad():

            _multi_class_accuracies = torch.stack(
                self.multi_class_accuracies_trainning, dim=0
            )
            _shape = _multi_class_accuracies.shape
            total_multi_class_accuracies = []

            for i in range(_shape[1]):
                _mean_value = self.compute_mean(_multi_class_accuracies[:, i])
                total_multi_class_accuracies.append(
                    np.around(_mean_value.item() * 100, decimals=2)
                )
            # _multi_class_accuracies=None

        return total_multi_class_accuracies

    def get_evaluations_score(self):
        accuracy_trainning = self.compute_mean(self.accuracies_trainning)
        f1_score_trainning = self.compute_mean(self.f1_scores_trainning)
        auroc_trainning = self.compute_mean(self.auroc_scores_trainning)

        return (
            accuracy_trainning.item(),
            f1_score_trainning.item(),
            auroc_trainning.item(),
        )

    def get_all_evaluations(self):
        return (
            self.accuracies_trainning,
            self.f1_scores_trainning,
            self.auroc_scores_trainning,
        )

    def reset_evaluations(self):
        self.accuracy_epoch.reset()
        self.f1_score_epoch.reset()
        self.auroc_epoch.reset()

        self.accuracies_trainning = []
        self.f1_scores_trainning = []
        self.auroc_scores_trainning = []

        self.mean_metric.reset()

        self.scheduler_learning_history = []

        self.accuracies_testset = []
        self.accuracy_testset.reset()

    def append_learning_rate(self, learning_rate):
        self.scheduler_learning_history.extend(learning_rate)

    def get_scheduler_learning_history(self):
        return self.scheduler_learning_history

    def compute_mean(self, values: float | torch.Tensor):
        self.mean_metric.update(values)
        mean_value = self.mean_metric.compute()
        self.mean_metric.reset()

        return mean_value


class ModelsTrainning:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        max_epochs,
        evaluation_function,
        USE_AUTOMATIC_MIXED_PRECISION=False,
    ):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.evaluation_function = evaluation_function

        self.USE_AUTOMATIC_MIXED_PRECISION = USE_AUTOMATIC_MIXED_PRECISION

        self.__modelsEvaluations = ModelsEvaluations()

    def setParameters(
        self,
        net_model: nn.Module,
        optimizer,
        scheduler_learning: torch.optim.lr_scheduler._LRScheduler,
        trial_optimisation: optuna.Trial = None,
    ):
        self.net_model = net_model
        self.optimizer = optimizer
        self.scheduler_learning = scheduler_learning
        self.trial_optimisation = trial_optimisation

    def set_base_clasifier(self, base_clasifier):
        self.net_model = base_clasifier

    def prune_trial(
        self,
        epoch: int,
        accuracy_epoch: float,
        losses_epoch: float,
        f1_score_epoch: float,
    ):
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

    def get_evaluation_result(self, accuracy_score, loss_score, f1_score):

        if self.evaluation_function == "accuracy":
            return accuracy_score

        elif self.evaluation_function == "loss":
            return loss_score

        elif self.evaluation_function == "f1_score":
            return f1_score

    def getModelCheckpoint(self):
        if self.USE_AUTOMATIC_MIXED_PRECISION:

            checkpoint = {
                "model": self.net_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler_learning": self.scheduler_learning.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
        else:
            checkpoint = {
                "model": self.net_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler_learning": self.scheduler_learning.state_dict(),
            }

        return checkpoint

    # use USE_AUTOMATIC_MIXED_PRECISION=True if Tensor Core is present
    def train_net_model(self):

        if self.USE_AUTOMATIC_MIXED_PRECISION:
            return self._trainNetModel_AmpOptimized()
        else:
            return self._trainNetModel()

    # has to be called after training or testing
    def get_evaluations_score(self):
        return self.__modelsEvaluations.get_evaluations_score()

    def get_scheduler_learning_history(self):
        return self.__modelsEvaluations.get_scheduler_learning_history()

    def get_multi_class_accuracies(self):
        return self.__modelsEvaluations.get_multi_class_accuracies_trainning_score()

    def _trainNetModel(self):

        MAX_EPOCHS = self.max_epochs

        self.__modelsEvaluations.reset_evaluations()
        self.__modelsEvaluations = ModelsEvaluations()

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)

        losses = []
        losses_epoch = []

        self.net_model.to(DEVICE)
        self.net_model.train()

        for epoch in range(MAX_EPOCHS):
            for step, (batch_x, batch_y) in enumerate(
                tqdm(self.train_loader, position=0, leave=True), 0
            ):
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
                self.__modelsEvaluations.update_evaluation_epoch(outputs_res, batch_y)

                self.scheduler_learning.step()
                learning_rate = self.scheduler_learning.get_last_lr()
                self.__modelsEvaluations.append_learning_rate(learning_rate)

            (
                _accuracy_epoch,
                _f1_score_epoch,
                _auroc_score_epoch,
            ) = self.__modelsEvaluations.get_evaluation_epoch()

            _losses_epoch = self.__modelsEvaluations.compute_mean(losses_epoch)
            losses_epoch = []

            # Add prune mechanism
            self.prune_trial(epoch, _accuracy_epoch, _losses_epoch, _f1_score_epoch)

            losses.append(_losses_epoch)

        (
            accuracies_scores,
            f1_scores,
            auroc_scores,
        ) = self.__modelsEvaluations.get_all_evaluations()

        return accuracies_scores, losses, f1_scores, auroc_scores

    # use if Tensor Core is present
    def _trainNetModel_AmpOptimized(self):
        # Necessary for FP16
        self.scaler = scaler = torch.cuda.amp.GradScaler()

        MAX_EPOCHS = self.max_epochs
        self.__modelsEvaluations.reset_evaluations()
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)

        losses = []
        losses_epoch = []

        self.net_model.to(DEVICE)
        self.net_model.train()

        for epoch in range(MAX_EPOCHS):

            for step, (batch_x, batch_y) in enumerate(
                tqdm(self.train_loader, position=0, leave=True), 0
            ):

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

                self.__modelsEvaluations.update_evaluation_epoch(outputs_res, batch_y)

                self.scheduler_learning.step()
                learning_rate = self.scheduler_learning.get_last_lr()
                self.__modelsEvaluations.append_learning_rate(learning_rate)

            (
                _accuracy_epoch,
                _f1_score_epoch,
                _auroc_score_epoch,
            ) = self.__modelsEvaluations.get_evaluation_epoch()

            _losses_epoch = self.__modelsEvaluations.compute_mean(losses_epoch)
            losses_epoch = []

            # Add prune mechanism
            self.prune_trial(epoch, _accuracy_epoch, _losses_epoch, _f1_score_epoch)

            losses.append(_losses_epoch)

        (
            accuracies_scores,
            f1_scores,
            auroc_scores,
        ) = self.__modelsEvaluations.get_all_evaluations()

        return accuracies_scores, losses, f1_scores, auroc_scores

    def testNetModel(self):

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(
            DEVICE
        )
        self.__modelsEvaluations.reset_evaluations()
        self.__modelsEvaluations = ModelsEvaluations()

        self.net_model.to(DEVICE)
        self.net_model.eval()

        losses_epoch = []
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(
                tqdm(self.test_loader, position=0, leave=True), 0
            ):

                # send to device
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(
                    DEVICE, non_blocking=True
                )

                outputs_res = self.net_model(batch_x)

                # assert not torch.isnan(outputs).any()
                loss = criterion(outputs_res, batch_y.long())

                losses_epoch.append(np.around(loss.item(), decimals=3))
                self.__modelsEvaluations.update_evaluation_epoch(outputs_res, batch_y)

            self.__modelsEvaluations.get_evaluation_epoch()
            (
                accuracies_scores,
                f1_scores,
                auroc_scores,
            ) = self.__modelsEvaluations.get_all_evaluations()

            losses_scores = losses_epoch

        return accuracies_scores, losses_scores, f1_scores, auroc_scores


def testNetModel123(net_model, test_loader, modelsEvaluations=ModelsEvaluations()):

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)

    net_model.to(DEVICE)
    net_model.eval()

    losses_epoch = []

    for step, (batch_x, batch_y) in enumerate(
        tqdm(test_loader, position=0, leave=True), 0
    ):

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
    losses_epoch = []

    (
        accuracies_scores,
        f1_scores,
        auroc_scores,
    ) = modelsEvaluations.getEvaluationModelTrainning()

    return (
        loss_score.item(),
        # accuracies_scores[0].item(),
        accuracies_scores,
        f1_scores[0].item(),
        auroc_scores[0].item(),
    )
