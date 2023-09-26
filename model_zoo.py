import torchvision.models as models
import torch.nn as nn
import torch



def get_mobilenet_small(pretrained=False, n_classes=4, greyscale=True):

    if pretrained:
        model = models.mobilenet_v3_small(num_classes=n_classes, pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.last_channel, n_classes)
    else:
        model = models.mobilenet_v3_small(num_classes=n_classes)
    return model




model_dict = {
    "mobilenet_small": get_mobilenet_small
}

if __name__ == "__main__":
    model = get_mobilenet_small()