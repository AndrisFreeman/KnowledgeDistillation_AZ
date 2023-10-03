import torch

from torch.optim import Adam
import torch.nn as nn
from torchvision import models
from model_zoo import model_dict
import time
import json
from statistics import median
import os
from copy import deepcopy
import torch.nn.functional as F
from sklearn.metrics import f1_score as f1


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
    if config_dict.get("curriculum", False) and config_dict.get("conditional", False):
        print("cant combine conditional with curriculum")
        return
    prep_directories()
    train_dataloader, val_dataloader = get_dataloader(config_dict)
    teacher_model = initialize_teacher_model(config_dict)
    student_model = model_dict[config_dict.get("student_model_name")](config_dict.get("pretrained", False))
    trainable_list = nn.ModuleList([])
    trainable_list.append(student_model)

    if config_dict.get("curriculum", False):
        mlp_net = Global_T()
        trainable_list.append(mlp_net)
        if config_dict.get("cosine_decay", False):
            gradient_decay = CosineDecay(max_value=1, min_value=0, num_loops=10)
        else:
            gradient_decay = LinearDecay(max_value=1, min_value=0, num_loops=10)
    else:
        mlp_net = None
        gradient_decay = None
    optimizer = Adam(params=trainable_list.parameters(), lr=config_dict.get("lr", 0.001), weight_decay=config_dict.get("weight_decay", 0))
    class_weights = torch.tensor([31130/2752, 31130/11635, 31130/14573, 31130/2170])
    loss_fn = nn.CrossEntropyLoss()
    param_num, size_all_mb = get_model_size(student_model)
    best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_time, train_f1s, val_f1s = training_loop(teacher_model,
                                                                                                                          student_model,
                                                                                                                          optimizer,
                                                                                                                          loss_fn,
                                                                                                                          train_dataloader,
                                                                                                                          val_dataloader,
                                                                                                                          config_dict,
                                                                                                                          gradient_decay,
                                                                                                                          mlp_net)
    res_dict = get_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, size_all_mb, batch_time, train_f1s, val_f1s)
    save_model(config_dict, best_model_params)

    return res_dict

def initialize_teacher_model(config_dict):
    model = models.vgg16()
    # Change first layer input channels to 1
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Update final layer to proper number of outputs
    num_ftrs = model.classifier[-1].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, config_dict.get("n_classes", 4))
    # Load pretrained weights
    if config_dict.get("teacher_model_name", "vgg16") == "vgg16":
        model_checkpoint = "models/VGG16_FF_2023_05_11_2214281.pt"
    model.load_state_dict(torch.load(model_checkpoint))
    for param in model.parameters():
        param.requires_grad = False

    return model

def training_loop(teacher_model, student_model, optimizer, loss_fn, train_loader, val_loader, config_dict, gradient_decay, mlp_net):
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    print(f"Device: {device}")
    teacher_model.to(device)
    student_model.to(device)
    train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_times, train_f1s, val_f1s = [], [], [], [], [], [], [], [], []
    best_model_params = None
    best_val_loss = None
    best_val_acc = None
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)
    for epoch in range(1, config_dict.get("num_epochs")+1):
        train_t1 = time.time()
        print(f"Start epoch {epoch}: {time.ctime(train_t1)}")
        if gradient_decay is not None:
            decay_value = gradient_decay.get_value(epoch)
        else:
            decay_value = None
        student_model, train_loss, train_acc, batch_time, train_f1 = train_epoch(teacher_model, student_model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   device,
                                                   config_dict,
                                                   decay_value,
                                                   mlp_net)
        train_t2 = time.time()
        print(f"End epoch {epoch}: {time.ctime(train_t2)}")
        train_times.append(train_t2 - train_t1)
        val_t1 = time.time()
        val_loss, val_acc, val_f1 = validate(teacher_model, student_model, loss_fn, val_loader, device, config_dict)
        val_t2 = time.time()
        val_times.append(val_t2 - val_t1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        batch_times.append(batch_time)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        # save best model
        if best_val_acc is None or val_acc > best_val_acc:   
            best_model_params = deepcopy(student_model.state_dict())
            best_val_loss = val_loss
            best_val_acc = val_acc
        # early stop
        if early_stopper.early_stop(val_acc):
            break
    print(f"Best achieved val acc: {best_val_acc}")
    return best_model_params, train_losses, train_accs, val_losses, val_accs, train_times, val_times, batch_times, train_f1s, val_f1s

def train_epoch(teacher_model, student_model, optimizer, loss_fn, train_loader, device, config_dict, decay_value, mlp_net):
    student_model.train()
    teacher_model.eval()
    if mlp_net is not None:
        mlp_net.train()
    train_loss_batches, train_acc_batches, f1_scores_batches = [], [], []
    times_batch = []
    Ts = []
    for batch_index, (x, y) in enumerate(train_loader, 1):
        t1 = time.time()
        if not config_dict.get("greyscale"):
            student_x = torch.repeat_interleave(x, 3, 1)
            teacher_inputs, student_inputs, labels = x.to(device), student_x.to(device), y.to(device)
            student_pred = student_model.forward(student_inputs)
            with torch.no_grad():
                teacher_pred = teacher_model.forward(teacher_inputs)
        else:
            inputs, labels = x.to(device), y.to(device)
            student_pred = student_model.forward(inputs)
            with torch.no_grad():
                teacher_pred = teacher_model.forward(inputs)
        if mlp_net is not None:
            # print(f"Decay:{decay_value}")
            T = mlp_net(teacher_pred, student_pred, decay_value)  # (teacher_output, student_output)
            # print(T)
            T = 1 + 20 * torch.sigmoid(T)
            # print(T)
            T = T.cuda()
            Ts.append(T.item())
        else:
            T = (config_dict.get("T") * torch.ones(1)).cuda()
        loss, hard_student_pred = compute_distillation_loss(teacher_pred, student_pred, labels, T, loss_fn, config_dict)
        loss.backward()

        optimizer.step()
        t2 = time.time()
        times_batch.append(t2-t1)
        optimizer.zero_grad()
        train_loss_batches.append(loss.item())

        hard_preds = hard_student_pred.argmax(dim=1)
        f1_score = f1(labels.cpu(), hard_preds.cpu(), average="micro")
        acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)
        f1_scores_batches.append(f1_score)
    return student_model, sum(train_loss_batches)/len(train_loss_batches), sum(train_acc_batches)/len(train_acc_batches), median(times_batch), sum(f1_scores_batches)/len(f1_scores_batches)

def validate(teacher_model, student_model, loss_fn, val_loader, device, config_dict):
    val_loss_cum = 0
    val_acc_cum = 0
    f1_score_cum = 0
    student_model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            if not config_dict.get("greyscale"):
                student_x = torch.repeat_interleave(x, 3, 1)
                teacher_inputs, student_inputs, labels = x.to(device), student_x.to(device), y.to(device)
                student_pred = student_model.forward(student_inputs)
                teacher_pred = teacher_model.forward(teacher_inputs)
            else:
                inputs, labels = x.to(device), y.to(device)
                student_pred = student_model.forward(inputs)
                teacher_pred = teacher_model.forward(inputs)

            loss, hard_student_pred = compute_distillation_loss(teacher_pred, student_pred, labels, config_dict.get("T"), loss_fn, config_dict)
            val_loss_cum += loss.item()
            hard_preds = hard_student_pred.argmax(dim=1)
            f1_score = f1(labels.cpu(), hard_preds.cpu(), average="micro")
            acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
            f1_score_cum += f1_score
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader), f1_score_cum/len(val_loader)

def compute_distillation_loss(teacher_pred, student_pred, labels, T, loss_fn, config_dict):
    soft_student_pred = student_pred / T
    soft_teacher_pred = teacher_pred / T
    # print(soft_student_pred.size())
    # print(soft_teacher_pred.size())
    soft_student_prob = F.log_softmax(soft_student_pred, dim=1)
    soft_teacher_prob = F.softmax(soft_teacher_pred, dim=1)
    if config_dict.get("conditional", False):
        for i,item in enumerate(teacher_pred):  
            if item.argmax(dim=0) != labels[i]:
                soft_teacher_prob[i] = torch.nn.functional.one_hot(labels[i], num_classes=4)
        loss = F.kl_div(soft_student_prob, soft_teacher_prob, log_target=False, reduction="batchmean")
    else:
        distillation_loss = F.kl_div(soft_student_prob, soft_teacher_prob, reduction="batchmean") * T * T
        student_loss = loss_fn(student_pred, labels)
        loss = config_dict.get("loss_ratio") * distillation_loss + (1 - config_dict.get("loss_ratio")) * student_loss
    hard_student_pred  = F.softmax(student_pred, dim=1)
    return loss, hard_student_pred

def get_res_report(config_dict, train_losses, train_accs, val_losses, val_accs, train_times, val_times, param_num, model_size, batch_time, train_f1s, val_f1s):
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
    filename = [config_dict["student_model_name"], f"best;{round(max(res_report['val_acc']), 6)}"]
    ignore_list = ["student_model_name", "train_dir", "val_dir", "train_split", "pretrained", "greyscale", "n_classes", "num_epochs","teacher_model_name", "finetuned"]
    for key, value in config_dict.items():
        if key not in ignore_list:
            filename.append(f"{key};{value}")
    return "-".join(filename)

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    run_experiment(config)
    