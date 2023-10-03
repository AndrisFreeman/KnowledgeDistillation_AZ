import torchvision.models as models
import torch.nn as nn
import torch
from ghostnet import ghostnet

def get_moblienet2(pretrained = True, n_classes=4, dropout = 0.2):

    if pretrained:
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
    else:
        model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=False),
                                     nn.Linear(in_features=1280,
                                               out_features=n_classes, bias=True))
    return model

def get_moblienet3_small(pretrained = True, n_classes=4, dropout = 0.2):

    if pretrained:
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    else:
        model = models.mobilenet_v3_small(weights=None)

    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1024, out_features=n_classes, bias=True))
    return model
def get_resnet18(pretrained = True, n_classes=4):
    if pretrained:
        model = models.resnet18(weights = 'ResNet18_Weights.IMAGENET1K_V1')
    else:
        model = models.resnet18(weights = None)
    model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    return model

def get_moblienet3_large(pretrained = True, n_classes=4, dropout = 0.2):
    if pretrained:
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    else:
        model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(nn.Linear(in_features=960, out_features=1280, bias=True),
                                     nn.Hardswish(),
                                     nn.Dropout(p=dropout, inplace=True),
                                     nn.Linear(in_features=1280, out_features=n_classes, bias=True)
                                     )
    return model

def get_densenet(pretrained = True, n_classes=4):
    if pretrained:
        model = models.densenet121(weights='IMAGENET1K_V1')
    else:
        model = models.densenet121(weights=None)

    model.classifier = nn.Linear(in_features=1024, out_features=n_classes, bias=True)

    return model
def get_vgg11(pretrained = True, n_classes=4):
    if pretrained:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)

    num_ftrs = model.classifier[-1].in_features
    model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=n_classes, bias=True)

    return model


def get_ghostnet(pretrained=True, n_classes=4):
    """
    Creates a GhostNet Model as defined in:
    GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
    https://arxiv.org/abs/1911.11907
    Modified from https://github.com/d-li14/mobilenetv3.pytorch
    Source: https://github.com/d-li14/ghostnet.pytorch/tree/master
    """
    model = ghostnet()
    if pretrained:
        model.load_state_dict(torch.load('ghostnet_1x-f97d70db.pth'))

    model.load_state_dict(torch.load('ghostnet_1x-f97d70db.pth'))
    model.classifier[-1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True)

    return model

def get_shufflenet(pretrained = True, n_classes=4):
    if pretrained:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)
    model.fc = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
    return model

def get_efficientnet_b3(pretrained = True, n_classes=4, dropout = 0.5):
    if pretrained:
        model = models.efficientnet_b3(weights = 'IMAGENET1K_V1')
    else:
        model = models.efficientnet_b3(weights = None)
    model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(in_features=1536, out_features=n_classes, bias=True))
    return model

def get_efficientnet_b4(pretrained = True, n_classes=4, dropout = 0.4):
    if pretrained:
        model = models.efficientnet_b3(weights = 'IMAGENET1K_V1')
    else:
        model = models.efficientnet_b3(weights = None)
    model.classifier = nn.Sequential(
                        nn.Dropout(p=dropout, inplace=True),
                        nn.Linear(in_features=1792, out_features=n_classes, bias=True))
    return model

model_dict = {
    "mobilenet-V2": get_moblienet2,
    "mobilenet-V3-small": get_moblienet3_small,
    "resnet18": get_resnet18,
    "mobilenet-V3-large": get_moblienet3_large,
    "densenet121": get_densenet,
    "vgg11": get_vgg11,
    'ghostnet': get_ghostnet,
    'shufflenet': get_shufflenet,
    'efficientnet-b3': get_efficientnet_b3,
    'efficientnet-b4': get_efficientnet_b4
}

if __name__ == "__main__":
    model_1 = get_moblienet2()
    model_2 = get_resnet18()
    model_3 = get_moblienet3_small()
    model_4 = get_moblienet3_large()
    model_5 = get_densenet()
    model_6 = get_vgg11()
    model_7 = get_ghostnet()
    model_8 = get_shufflenet()
    model_9 = get_efficientnet_b3()
    model_10 = get_efficientnet_b4()

