import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from model_zoo import model_dict
import time
import json
import itertools
import os
from copy import deepcopy
from queue import Queue
import torch.nn.functional as F
torch.manual_seed(1)


def make_grid(hyperparam_dict):
    """ Creates grid of all hyperparameter combinations
    """
    keys=hyperparam_dict.keys()
    combinations=itertools.product(*hyperparam_dict.values())
    grid=[dict(zip(keys,cc)) for cc in combinations]
    return grid

def prep_directories():
    try: 
        os.mkdir("models")
    except:
        print("Model directory already exists")
    try: 
        os.mkdir("results")
    except:
        print("Results directory already exists")

def run_experiment(student_model_name, batch_size=32, data_dir="real_data", n_classes=4, num_epochs=20, T=.8, loss_ratio=.5, greyscale=True):
    prep_directories()
    train_dataloader, val_dataloader = get_dataloader(data_dir, batch_size)
    teacher_model = initialize_teacher_model(n_classes=n_classes)
    student_model = model_dict[student_model_name]()
    optimizer = Adam(params=student_model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    param_num, size_all_mb = get_model_size(student_model)
    best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times = training_loop(teacher_model, student_model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs, T, loss_ratio, greyscale)
    report_name = construct_report_name(student_model_name, T, loss_ratio)
    res_dict = get_res_report(report_name, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, size_all_mb)
    # save_model(config, best_model_params)

    return res_dict

def save_model(config, best_model_params):
    torch.save(best_model_params, f"models/{config['filename']}.pt")

def construct_report_name(student_model_name, T, loss_ratio):
    report_name = f"{student_model_name}_{T}_{loss_ratio}"
    return report_name

def initialize_teacher_model(model_checkpoint="models/VGG16_FF_2023_05_11_2214281.pt", n_classes=4):
    model = models.vgg16()
    # Change first layer input channels to 1
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Update final layer to proper number of outputs
    num_ftrs = model.classifier[-1].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
    # Load pretrained weights
    model.load_state_dict(torch.load(model_checkpoint))
    for param in model.parameters():
        param.requires_grad = False

    return model

def get_dataloader(data_dir, batch_size=32):
    # Transformations
    input_size = 112
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) ])
    # Dataset
    data = datasets.ImageFolder(root=data_dir, transform=transform)
    generator1 = torch.Generator().manual_seed(42)
    test = len(data) - 1000 - 100
    train_data, val_data, _ = random_split(data, [1000, 100, test], generator=generator1)
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

def training_loop(teacher_model, student_model, optimizer, loss_fn, train_loader, val_loader, num_epochs, T, loss_ratio, greyscale):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    print(f"Device: {device}")
    teacher_model.to(device)
    student_model.to(device)
    train_losses, train_accs, val_losses, val_accs, train_times, val_times = [], [], [], [], [], []
    best_model_params = None
    best_val_loss = None
    best_val_acc = None
    early_stop_list = Queue(maxsize=5)
    for epoch in range(1, num_epochs+1):
        train_t1 = time.time()
        student_model, train_loss, train_acc = train_epoch(teacher_model, student_model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   val_loader,
                                                   device,
                                                   T,
                                                   loss_ratio,
                                                   greyscale)
        train_t2 = time.time()
        train_times.append(train_t2 - train_t1)
        val_t1 = time.time()
        val_loss, val_acc = validate(teacher_model, student_model, loss_fn, val_loader, device, T, loss_ratio, greyscale)
        val_t2 = time.time()
        val_times.append(val_t2 - val_t1)
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {train_loss:.3f}, "
              f"Train acc.: {train_acc:.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # save best model
        if best_val_acc is None or val_acc > best_val_acc:
                    
            best_model_params = deepcopy(student_model.state_dict())
            best_val_loss = val_loss
            best_val_acc = val_acc
        # early stop
        if early_stop_list.empty() or not early_stop_list.full():
            early_stop_list.put(val_acc)
        else:
            early_stop_list.get()
            early_stop_list.put(val_acc)
        if max(list(early_stop_list.queue)) < best_val_acc and early_stop_list.full():
            break
        print(f"Best achieved val acc: {best_val_acc}")
    return best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times

def temperature_softmax(logits, T):
    logits = logits / T
    return F.softmax(logits, dim=0)

def train_epoch(teacher_model, student_model, optimizer, loss_fn, train_loader, val_loader, device, T, loss_ratio, greyscale):
    student_model.train()
    teacher_model.eval()
    train_loss_batches, train_acc_batches = [], []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        if not greyscale:
            student_x = torch.repeat_interleave(x, 3, 1)
            teacher_inputs, student_inputs, labels = x.to(device), student_x.to(device), y.to(device)
            student_pred = student_model.forward(student_inputs)
            teacher_pred = teacher_model.forward(teacher_inputs)
        else:
            inputs, labels = x.to(device), y.to(device)
            student_pred = student_model.forward(inputs)
            teacher_pred = teacher_model.forward(inputs)

        loss, hard_student_pred = compute_distillation_loss(teacher_pred, student_pred, labels, T, loss_ratio, loss_fn)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_batches.append(loss.item())

        hard_preds = hard_student_pred.argmax(dim=1)
        acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)

    return student_model, sum(train_loss_batches)/len(train_loss_batches), sum(train_acc_batches)/len(train_acc_batches)

def validate(teacher_model, student_model, loss_fn, val_loader, device, T, loss_ratio, greyscale):
    val_loss_cum = 0
    val_acc_cum = 0
    student_model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            if not greyscale:
                student_x = torch.repeat_interleave(x, 3, 1)
                teacher_inputs, student_inputs, labels = x.to(device), student_x.to(device), y.to(device)
                student_pred = student_model.forward(student_inputs)
                teacher_pred = teacher_model.forward(teacher_inputs)
            else:
                inputs, labels = x.to(device), y.to(device)
                student_pred = student_model.forward(inputs)
                teacher_pred = teacher_model.forward(inputs)

            loss, hard_student_pred = compute_distillation_loss(teacher_pred, student_pred, labels, T, loss_ratio, loss_fn)
            val_loss_cum += loss.item()
            hard_preds = hard_student_pred.argmax(dim=1)
            acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)

def compute_distillation_loss(teacher_pred, student_pred, labels, T, loss_ratio, loss_fn):
    soft_student_pred = temperature_softmax(student_pred, T)
    soft_teacher_pred = temperature_softmax(teacher_pred, T)
    hard_student_pred = temperature_softmax(student_pred, 1)
    distillation_loss = loss_fn(soft_student_pred, soft_teacher_pred)
    student_loss = loss_fn(hard_student_pred, labels)
    loss = loss_ratio * distillation_loss + (1- loss_ratio) * student_loss
    return loss, hard_student_pred

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

def get_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, model_size):
    res_report = {
        "train_speed": train_times,
        "inference_speed": val_times,
        "student_memory": model_size,
        "student_num_params": param_num,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "F1_score": [],
    }
    output_file_name = construct_result_filename(config_dict)
    config_dict["filename"] = output_file_name
    for key, value in res_report.items():
        config_dict[key] = value

    with open(f'results/{output_file_name}.json', 'w') as fp:
        json.dump(res_report, fp)
    return res_report

def construct_result_filename(config_dict):
    filename = [config_dict["model_name"]]
    for key, value in config_dict.items():
        if key != "model_name":
            filename.append(f"{key}:{value}")
    return "_".join(filename)



if __name__ == "__main__":
   run_experiment("mobilenet_small", num_epochs=2)