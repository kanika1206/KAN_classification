import torch
import torch.nn as nn
from torchvision import models

from kan import KANLinear

class ConvNeXt(nn.Module):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        # Load pre-trained ConvNeXt model
        self.convnext = models.convnext_tiny(pretrained=True)

        # Freeze layers 
        for param in self.convnext.parameters():
            param.requires_grad = False

        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(num_features, 10)  # 10 classes

    def forward(self, x):
        return self.convnext(x)

class VGG16KAN(nn.Module):
    def __init__(self):
        super(VGG16KAN, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze layers 
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modify classifier part
        num_features = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Identity()  # Remove original classifier

        # Add KANLinear layers 
        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, 10)  # 10 classes in the dataset

    def forward(self, x):
        x = self.vgg16.features(x)  # Extract features using VGG16
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.kan1(x)
        x = self.kan2(x)
        return x
    
class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze layers 
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modify the classifier to match the number of classes
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.vgg16(x)
    
    
class MobileNetV2Model(nn.Module):
    def __init__(self):
        super(MobileNetV2Model, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Freeze layers
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # Replace the classifier to match the number of classes
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.mobilenet(x)
    

class MobileNetV2KAN(nn.Module):
    def __init__(self):
        super(MobileNetV2Model, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        # Freeze layers
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Replace the classifier with KANLinear layers
        in_features = self.mobilenet.classifier[1].in_features
        self.kan1 = KANLinear(in_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.mobilenet.features(x)  # Use feature extractor
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.kan1(x)
        return self.kan2(x)


class EfficientNetB0Model(nn.Module):
    def __init__(self):
        super(EfficientNetB0Model, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Freeze features if needed
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.efficientnet(x)
    
    
class EfficientNetB0KAN(nn.Module):
    def __init__(self):
        super(EfficientNetB0Model, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Freeze features if needed
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Replace the classifier with KANLinear layers
        in_features = self.efficientnet.classifier[1].in_features
        self.kan1 = KANLinear(in_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.efficientnet.features(x)  # Extract features
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.kan1(x)
        return self.kan2(x)


class ResNet101Model(nn.Module):
    def __init__(self):
        super(ResNet101Model, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        # Freeze features if needed
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.resnet(x)
    
class ResNet101KAN(nn.Module):
    def __init__(self):
        super(ResNet101Model, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

        # Freeze features if needed
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with KANLinear layers
        in_features = self.resnet.fc.in_features
        self.kan1 = KANLinear(in_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.resnet.forward_features(x)  # Extract features
        x = self.kan1(x)
        return self.kan2(x)


class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        # Freeze features if needed
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 10)

    def forward(self, x):
        return self.vit(x)
    


class ViTKAN(nn.Module):
    def __init__(self):
        super(ViTKAN, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        in_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        # Freeze the parameters 
        for param in self.vit.parameters():
            param.requires_grad = False

        # Define the KANLinear layers
        self.kan1 = KANLinear(in_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.vit(x)  # Extract features
        x = self.kan1(x)
        return self.kan2(x)


