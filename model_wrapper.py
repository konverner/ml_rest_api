from torch import nn
from torchvision import models


class ModelWrapper:
    def __init__(
            self,
            backbone_name: str,
            device: str,
    ):
        self.learnable_parameters = None
        self.backbone_name = backbone_name
        self.model = getattr(models, backbone_name)()
        self.device = device
        self.trained = False
        self.wandb = None

    def count_parameters(self):
        return sum(p.numel() for p in self.learnable_parameters if p.requires_grad)

    def train(self):
        self.model.train()
        self.trained = True

    def eval(self):
        self.model.eval()

    def init_model(self, num_classes, freeze_backbone):
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        if self.backbone_name == 'resnet18':
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model.to(self.device)

        self.learnable_parameters = [param for param in self.model.parameters()
                                     if param.requires_grad==True]

    def get_logits(self, x):
        return self.model(x)

    def get_info(self):
        info = { "backbone_name": self.backbone_name,
                 "device": self.device,
                 "trained": self.trained}
        if self.trained:
            info = info.update({"num_classes": self.num_classes,
                     "backbone_freezed": self.freeze_backbone})
        return info
