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
    train_dataloader, val_dataloader = get_dataloader(config_dict.get("train_dir", "real_data"), config_dict.get("val_dir", "val"), config_dict.get("bs", 128), config_dict.get("train_split", 0.85))
    teacher_model = initialize_teacher_model(n_classes=config_dict.get("n_classes", 4))
    student_model = model_dict[config_dict.get("student_model_name")](config_dict.get("pretrained", False))
    optimizer = Adam(params=student_model.parameters(), lr=config_dict.get("lr", 0.001), weight_decay=config_dict.get("weight_decay", 0))
    loss_fn = nn.CrossEntropyLoss()
    param_num, size_all_mb = get_model_size(student_model)
    best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_time = training_loop(teacher_model, student_model, optimizer, loss_fn, train_dataloader, val_dataloader, config_dict.get("num_epochs"), config_dict.get("T"), config_dict.get("loss_ratio"), config_dict.get("greyscale"))
    res_dict = get_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, size_all_mb, batch_time)
    save_model(config_dict, best_model_params)

    return res_dict

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
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)
    for epoch in range(1, num_epochs+1):
        train_t1 = time.time()
        print(f"Start epoch {epoch}: {time.ctime(train_t1)}")
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
        print(f"End epoch {epoch}: {time.ctime(train_t2)}")
        train_times.append(train_t2 - train_t1)
        val_t1 = time.time()
        val_loss, val_acc = validate(teacher_model, student_model, loss_fn, val_loader, device, T, loss_ratio, greyscale)
        val_t2 = time.time()
        val_times.append(val_t2 - val_t1)
        # print(f"Epoch {epoch}/{num_epochs}: "
        #       f"Train loss: {train_loss:.3f}, "
        #       f"Train acc.: {train_acc:.3f}, "
        #       f"Val. loss: {val_loss:.3f}, "
        #       f"Val. acc.: {val_acc:.3f}")
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
        if early_stopper.early_stop(val_acc):
            break
    print(f"Best achieved val acc: {best_val_acc}")
    return best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times

def train_epoch(teacher_model, student_model, optimizer, loss_fn, train_loader, val_loader, device, T, loss_ratio, greyscale):
    student_model.train()
    teacher_model.eval()
    train_loss_batches, train_acc_batches = [], []
    times_batch = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        t1 = time.time()
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
        t2 = time.time()
        times_batch.append(t2-t1)
        optimizer.zero_grad()
        train_loss_batches.append(loss.item())

        hard_preds = hard_student_pred.argmax(dim=1)
        acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)
    return student_model, sum(train_loss_batches)/len(train_loss_batches), sum(train_acc_batches)/len(train_acc_batches), median(times_batch)

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
    soft_student_pred = student_pred / T
    soft_teacher_pred = teacher_pred / T
    # print(soft_student_pred.size())
    # print(soft_teacher_pred.size())
    soft_student_prob = F.log_softmax(soft_student_pred, dim=1)
    soft_teacher_prob = F.log_softmax(soft_teacher_pred, dim=1)
    distillation_loss = F.kl_div(soft_student_prob, soft_teacher_prob, reduction="batchmean", log_target=True) * T * T
    student_loss = loss_fn(student_pred, labels)
    loss = loss_ratio * distillation_loss + (1 - loss_ratio) * student_loss
    hard_student_pred  = F.softmax(student_pred, dim=1)
    return loss, hard_student_pred

def get_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, model_size, batch_time):
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
        "batch_time": batch_time
    }
    output_file_name = construct_result_filename(config_dict, res_report)
    config_dict["filename"] = output_file_name
    for key, value in res_report.items():
        config_dict[key] = value

    with open(f'results/{output_file_name}.json', 'w') as fp:
        json.dump(config_dict, fp, indent=2)
    return res_report

def construct_result_filename(config_dict, res_report):
    filename = [config_dict["student_model_name"], f"best-{round(max(res_report['val_acc']), 6)}"]
    ignore_list = ["student_model_name", "data_dir", "train_split", "pretrained", "greyscale", "n_classes", "num_epochs","teacher_model_name", "finetuned"]
    for key, value in config_dict.items():
        if key not in ignore_list:
            filename.append(f"{key};{value}")
    return "-".join(filename)

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    run_experiment(config)
    