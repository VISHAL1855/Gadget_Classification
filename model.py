import torch
import torchvision

from torch import nn


def create_gadgets_model(num_classes:int=3, 
                          seed:int=42):
    # Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.resnet50(weights=weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(in_features= 128,out_features=num_classes))
    
    return model, transforms
