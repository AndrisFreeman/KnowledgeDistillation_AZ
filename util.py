import torch
import os
import itertools
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def save_model(config, best_model_params):
    torch.save(best_model_params, f"models/{config['filename']}.pt")

def prep_directories():
    try: 
        os.mkdir("models")
    except:
        print("Model directory already exists")
    try: 
        os.mkdir("results")
    except:
        print("Results directory already exists")

def make_grid(hyperparam_dict):
    """ Creates grid of all hyperparameter combinations
    """
    keys=hyperparam_dict.keys()
    combinations=itertools.product(*hyperparam_dict.values())
    grid=[dict(zip(keys,cc)) for cc in combinations]
    return grid

def get_model_size(model):
    param_mem = 0
    param_num = 0
    for param in model.parameters():
        param_num += param.nelement()
        param_mem += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_mem + buffer_size) / 1024**2
    return param_num, size_all_mb

def get_dataloader(train_dir, val_dir, batch_size=128, train_split=0.85):
    # Transformations
    input_size = 112
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) ])
    # Dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    # train_data = Subset(train_data, list(range(1000)))
    val_data = datasets.ImageFolder(root=val_dir, transform=transform)

    # generator1 = torch.Generator().manual_seed(42)
    # full_length = len(data)
    # train_len = int(full_length * train_split)

    # train_data, val_data = random_split(data, [train_len, full_length - train_len], generator=generator1)
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

class Global_T(torch.nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        
        self.global_T = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)


from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)