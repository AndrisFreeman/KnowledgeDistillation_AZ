import torchvision.models as models
import torch.nn as nn
import torch

def get_moblienet2(pretrained = True, num_classes=4, greyscale=True):

    if pretrained:
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        for param in model.features.parameters():
            param.requires_grad = False
        model.features[0][0].in_channels = 1
        model.classifier[1].out_features = num_classes
    else:
        model = models.mobilenet_v2(weights=None, num_classes=num_classes)
        model.features[0][0].in_channels = 1
        model.classifier[1].out_features = num_classes
        for param in model.features.parameters():
            param.requires_grad = True
    return model

def get_moblienet3(pretrained = True, n_classes=4, greyscale=True):

    if pretrained:
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        for param in model.features.parameters():
            param.requires_grad = False
        model.features[0][0].in_channels = 1
        model.classifier[3].out_features = n_classes
    else:
        model = models.mobilenet_v3_small(weights=None, num_classes=n_classes)
        model.features[0][0].in_channels = 1
        model.classifier[3].out_features = n_classes
        for param in model.features.parameters():
            param.requires_grad = True
    return model

def get_resnet18(pretrained = True, n_classes=4, greyscale=True):
    if pretrained:
        model = models.resnet18(weights = 'ResNet18_Weights.IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        model.fc.out_features = n_classes
        model.conv1.in_channels = 1
    else:
        model = models.resnet18(weights = None, num_classes=n_classes)
        model.fc.out_features = n_classes
        model.conv1.in_channels = 1
        for param in model.parameters():
            param.requires_grad = True
    return model



model_dict = {
    "mobilenet-V2": get_moblienet2,
    "mobilenet-V3": get_moblienet3,
    "resnet18": get_resnet18
}

if __name__ == "__main__":
    model_1 = get_moblienet2()
    model_2 = get_resnet18()
    model_3 = get_moblienet3()