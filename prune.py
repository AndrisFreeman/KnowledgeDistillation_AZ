import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from model_zoo import model_dict
import time
import json
from statistics import median
import os
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics import f1_score as f1
import torch.nn.utils.prune as prune

from util import *
torch.manual_seed(1)


def run_experiment(config):
    try:
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
    except:
        print("not running sbatch array")
        task_id = 0
    try:
        config_dict = make_grid(config)[task_id]
    except:
        print("not enough tasks available")
        return
    prep_directories()
    if config_dict["bird"]:
        train_dataloader, val_dataloader = get_bird_dataloader(config_dict)
    else:
        train_dataloader, val_dataloader = get_dataloader(config_dict)
    model = model_dict[config_dict.get("model_name")](config_dict.get("pretrained", False))
    optimizer = Adam(params=model.parameters(), lr=config_dict.get("lr", 0.001), weight_decay=config_dict.get("weight_decay", 0))
    loss_fn = nn.CrossEntropyLoss()
    for i in range(config_dict.get("num_iter", 5)):
        if config_dict.get("structured", False):
            structured_pruning(model, config_dict)
        else:
            unstructured_pruning(model, config_dict)
        if config_dict.get("finetune", True):
            best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_time, train_f1s, val_f1s = training_loop(model, optimizer, loss_fn, train_dataloader, val_dataloader, config_dict.get("num_epochs"))
        num_zeros, num_elements, sparsity = measure_global_sparsity(model)
        copy_model = deepcopy(model)
        copy_model = remove_parameters(copy_model)
        param_num, size_all_mb = get_model_size(copy_model)
        append_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, size_all_mb, batch_time, train_f1s, val_f1s, sparsity, i+1)
    if not (config_dict.get("num_iter", 5) > 1 and config_dict.get("global_prune", False)):
        final_pruning_rate = compute_final_pruning_rate(config_dict.get("conv_prune"), config_dict.get("num_iter"))
        config_dict["final_pruning_rate"] = final_pruning_rate
    save_report(config_dict)
    return

def unstructured_pruning(model, config_dict):
    if config_dict.get("global_prune", False):
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            if config_dict.get("random_prune", False):
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=config_dict.get("conv_prune", 0.2),
                )
            else:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=config_dict.get("conv_prune", 0.2),
                )
    else:
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if config_dict.get("random_prune", False):
                    prune.random_unstructured(module,
                                            name="weight",
                                            amount=config_dict.get("conv_prune", 0.2))
                else:
                    prune.l1_unstructured(module,
                                            name="weight",
                                            amount=config_dict.get("conv_prune", 0.2))
            elif isinstance(module, torch.nn.Linear):
                if config_dict.get("random_prune", False):
                    prune.random_unstructured(module,
                                            name="weight",
                                            amount=config_dict.get("lin_prune", 0.2))
                else:
                    prune.l1_unstructured(module,
                                            name="weight",
                                            amount=config_dict.get("lin_prune", 0.2))

def structured_pruning(model, config_dict):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if config_dict.get("random_prune", False):
                prune.random_structured(module,
                                        name="weight",
                                        amount=config_dict.get("conv_prune", 0.2),
                                        dim=0)
            else:
                prune.ln_structured(module,
                                        name="weight",
                                        amount=config_dict.get("conv_prune", 0.2),
                                        dim=0,
                                        n=2)
        elif isinstance(module, torch.nn.Linear):
            if config_dict.get("random_prune", False):
                prune.random_structured(module,
                                        name="weight",
                                        amount=config_dict.get("lin_prune", 0.2),
                                        dim=0)
            else:
                prune.ln_structured(module,
                                        name="weight",
                                        amount=config_dict.get("lin_prune", 0.2),
                                        dim=0,
                                        n=2)


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def get_dataloader(data_dir, batch_size=128, train_split=0.85):
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
    full_length = len(data)
    train_len = int(full_length * train_split)

    train_data, val_data = random_split(data, [train_len, full_length - train_len], generator=generator1)
    # Dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    print(f"Device: {device}")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_times, train_f1s, val_f1s = [], [], [], [], [], [], [], [], []
    best_model_params = None
    best_val_loss = None
    best_val_acc = None
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)
    for epoch in range(1, num_epochs+1):
        train_t1 = time.time()
        print(f"Start epoch {epoch}: {time.ctime(train_t1)}")
        model, train_loss, train_acc, batch_time, hard_preds, labels = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   val_loader,
                                                   device)
        train_t2 = time.time()
        print(f"End epoch {epoch}: {time.ctime(train_t2)}")
        train_times.append(train_t2 - train_t1)
        val_t1 = time.time()
        val_loss, val_acc, val_preds, val_labels = validate(model, loss_fn, val_loader, device)
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
        batch_times.append(batch_time)
        print(labels[:5])
        print(hard_preds[:5])
        for x in [labels, hard_preds, val_labels, val_preds]:
            if torch.is_tensor(x):
                x.cpu()
        train_f1s.append(f1(labels, hard_preds, average="micro"))
        val_f1s.append(f1(val_labels, val_preds, average="micro"))
        # save best model
        if best_val_acc is None or val_acc > best_val_acc:   
            best_model_params = deepcopy(model.state_dict())
            best_val_loss = val_loss
            best_val_acc = val_acc
        # early stop
        if early_stopper.early_stop(val_acc):
            break
    print(f"Best achieved val acc: {best_val_acc}")
    return best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_time, train_f1s, val_f1s

def train_epoch(model, optimizer, loss_fn, train_loader, val_loader, device):
    model.train()
    train_loss_batches, train_acc_batches, epoch_hard_preds, epoch_labels = [], [], [], []
    times_batch = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        t1 = time.time()

        inputs, labels = x.to(device), y.to(device)
        teacher_pred = model.forward(inputs)
        loss = loss_fn(teacher_pred, labels)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        times_batch.append(t2-t1)
        optimizer.zero_grad()
        train_loss_batches.append(loss.item())

        hard_preds = F.softmax(teacher_pred, dim=1).argmax(dim=1)
        epoch_hard_preds.extend(hard_preds.cpu())
        epoch_labels.extend(labels.cpu())
        acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)
    return model, sum(train_loss_batches)/len(train_loss_batches), sum(train_acc_batches)/len(train_acc_batches), median(times_batch), epoch_hard_preds, epoch_labels

def validate(model, loss_fn, val_loader, device):
    val_loss, val_acc, epoch_hard_preds, epoch_labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):

            inputs, labels = x.to(device), y.to(device)
            teacher_pred = model.forward(inputs)

            loss = loss_fn(teacher_pred, labels)
            val_loss.append(loss.item())
            hard_preds = F.softmax(teacher_pred, dim=1).argmax(dim=1)
            acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
            val_acc.append(acc_batch_avg)
            epoch_hard_preds.extend(hard_preds.cpu())
            epoch_labels.extend(labels.cpu())
    return sum(val_loss)/len(val_loss), sum(val_acc)/len(val_acc), epoch_hard_preds, epoch_labels

def append_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, model_size, batch_time, train_f1s, val_f1s, sparsity, iter):
    res_report = {
        "train_speed": train_times,
        "inference_speed": val_times,
        "student_memory": model_size,
        "student_num_params": param_num,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_F1_score": train_f1s,
        "val_F1_score": val_f1s,
        "batch_time": batch_time,
        "sparsity": sparsity
    }
    
    if "results" not in config_dict.keys():
        config_dict["results"] = {}
    config_dict["results"][iter] = res_report
    return

def save_report(config_dict):
    output_file_name = construct_result_filename(config_dict)
    config_dict["filename"] = output_file_name
    with open(f'results/{output_file_name}.json', 'w') as fp:
        json.dump(config_dict, fp, indent=2)

def construct_result_filename(config_dict):
    filename = [config_dict["model_name"]]
    ignore_list = ["student_model_name", "train_dir", "val_dir", "train_split", "pretrained", "greyscale", "n_classes", "num_epochs", "model_name", "T", "loss_ratio", "conditional", "cosine_decay", "curriculum"]
    for key, value in config_dict.items():
        if key not in ignore_list:
            filename.append(f"{key};{value}")
    return "-".join(filename)

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    run_experiment(config)
    