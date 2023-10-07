import torch
import os
import itertools
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math
try:
    from bird_data import BirdData
except:
    print("no birds")
# from bird_data import BirdData

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

def get_dataloader(config_dict):
    # Transformations
    input_size = 112
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) ])
    # Dataset
    train_data = datasets.ImageFolder(root=config_dict.get("train_dir", "real_data"), transform=transform)
    # train_data = Subset(train_data, list(range(2000)))
    val_data = datasets.ImageFolder(root=config_dict.get("val_dir", "val"), transform=transform)

    # generator1 = torch.Generator().manual_seed(42)
    # full_length = len(data)
    # train_len = int(full_length * train_split)

    # train_data, val_data = random_split(data, [train_len, full_length - train_len], generator=generator1)
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=config_dict.get("bs", 128), shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config_dict.get("bs", 128), shuffle=False)
    return train_dataloader, val_dataloader

def get_bird_dataloader(config_dict):
    # Transformations

    # Dataset
    train_data = BirdData(config_dict.get("train_dir", "bird_train.zip"))
    # train_data = Subset(train_data, list(range(2000)))
    val_data = BirdData(config_dict.get("val_dir", "bird_valid.zip"))

    # generator1 = torch.Generator().manual_seed(42)
    # full_length = len(data)
    # train_len = int(full_length * train_split)

    # train_data, val_data = random_split(data, [train_len, full_length - train_len], generator=generator1)
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=config_dict.get("bs", 128), shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config_dict.get("bs", 128), shuffle=False)
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

# class CosineDecay(object):
#     def __init__(self,
#                 max_value,
#                 min_value,
#                 num_loops):
#         self._max_value = max_value
#         self._min_value = min_value
#         self._num_loops = num_loops

#     def get_value(self, i):
#         if i < 0:
#             i = 0
#         if i >= self._num_loops:
#             i = self._num_loops
#         value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
#         value = value * (self._max_value - self._min_value) + self._min_value
#         return value


# class LinearDecay(object):
#     def __init__(self,
#                 max_value,
#                 min_value,
#                 num_loops):
#         self._max_value = max_value
#         self._min_value = min_value
#         self._num_loops = num_loops

#     def get_value(self, i):
#         if i < 0:
#             i = 0
#         if i >= self._num_loops:
#             i = self._num_loops - 1

#         value = (self._max_value - self._min_value) / self._num_loops
#         value = i * (-value)

#         return value

class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value

        return 1-value


class LinearDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (value)

        return value

def compute_final_pruning_rate(pruning_rate, num_iterations):
    final_pruning_rate = 1 - (1 - pruning_rate)**num_iterations

    return final_pruning_rate


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity