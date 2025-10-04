import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def get_resnet18_model(num_classes=4, device="cpu"):
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),  
        nn.ReLU(),                
        nn.Dropout(0.5),           
        nn.Linear(512, num_classes)          
    )
    
    model = model.to(device)
    return model

def setup_resnet18_optimizer(model, lr=0.001):
    return optim.Adam(model.fc.parameters(), lr=lr)

def setup_resnet18_finetuning(model, lr=1e-4):
    for name, param in model.named_parameters():
        if 'layer4' in name or 'layer3' in name or 'fc' in name:
            param.requires_grad = True
    
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)