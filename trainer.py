import copy
from typing import List, Dict

import wandb
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from model_wrapper import ModelWrapper


class Trainer:
    def __init__(
            self,
            config: dict,
            model_wrapper: ModelWrapper,
            dataloaders: List[DataLoader],
            id2label: List[str],
            wandb_params: Dict[str, str] = None):
        self.config = config
        self.model_wrapper = model_wrapper
        self.num_classes = len(id2label)
        self.dataloaders = dataloaders

        self.metrics = {phase: {'accuracy': [], 'loss': []} for phase in self.dataloaders.keys()}
        self.model_wrapper.init_model(self.num_classes, config['freeze_backbone'])
        self.optimizer = getattr(optim, config['optimizer_name'])(params=self.model_wrapper.learnable_parameters, lr=config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        self.last_record = 0.
        self.best_model = None

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model_wrapper.train()
                else:
                    self.model_wrapper.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.model_wrapper.device)
                    labels = labels.to(self.model_wrapper.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = self.model_wrapper.get_logits(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                self.metrics[phase]['loss'].append(running_loss / len(self.dataloaders[phase].dataset))
                self.metrics[phase]['accuracy'].append(running_corrects / len(self.dataloaders[phase].dataset))

                print('{:>12} {:>12} {:>12.3f} {:>12.3f}'.format(epoch, phase, self.metrics[phase]['loss'][-1],
                                                                 self.metrics[phase]['accuracy'][-1]))

                if phase == 'valid':
                    if self.metrics[phase]['accuracy'][-1] > self.last_record:
                        self.last_record = self.metrics[phase]['accuracy'][-1]
                        self.best_model = copy.deepcopy(self.model_wrapper.model)
                        print('new best model achieved with test accuracy {:.3f}'.format(self.last_record))

                wandb.log({f"accuracy_{phase}": self.metrics[phase]['accuracy'][-1],
                           f"loss_{phase}": self.metrics[phase]['loss'][-1],
                           f"learning rate": self.scheduler._last_lr[0]})

    def eval(self, dataloader, metric_func=None):
        total_metrics = 0
        for inputs, labels in self.dataloaders:
            inputs = inputs.to(self.model_wrapper.device)
            labels = labels.to(self.model_wrapper.device)

            outputs = self.modelWrapper.get_logits(inputs)
            _, preds = torch.max(outputs, 1)
            total_metrics += metric_func(labels.data, preds.data)
        return total_metrics/len(dataloader)
