import torch
from torch.optim import Adam
import torch.nn as nn
from model_zoo import model_dict
import time
import json
from statistics import median
import os
import glob
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
        eval_config_dict = make_grid(config)[task_id]
    except:
        print("not enough tasks available")
        return
    prep_directories()
    files = glob.glob(f"results/{eval_config_dict.get('model_name', 'resnet50')}*")
    print(files)
    for file in files:
        with open(file, 'r') as f:
            config_dict = json.load(f)
        try:
            model_checkpoint = find_model_checkpoint(file, eval_config_dict)
            if model_checkpoint is not None:
                n_classes = 525 if config_dict.get("bird") else 4
                model = model_dict[eval_config_dict.get("model_name")](pretrained=False, n_classes=n_classes)
                print("built model")
            else: continue
            model.load_state_dict(torch.load(model_checkpoint))
        except:
            continue
        if config_dict["bird"]:
            train_dataloader, val_dataloader = get_bird_dataloader(config_dict)
        else:
            train_dataloader, val_dataloader = get_dataloader(config_dict)
        
        val_loss, val_acc, hard_preds, labels, batch_time = validate(model, loss_fn = nn.CrossEntropyLoss(), val_loader=val_dataloader)
        config_dict["hard_preds"] = hard_preds
        config_dict["labels"] = labels
        config_dict["inference_batch_time"] = batch_time
        with open(f'file', 'w') as fp:
            json.dump(config_dict, fp, indent=2)
        
    return 


def find_model_checkpoint(res_filename, config_dict):
    candidates = glob.glob(f"models/{config_dict.get('model_name', 'resnet50')}*")
    match_string = res_filename.split("\\")[1].split("-")[1]
    print(match_string)
    if ";" not in match_string:
        match_string = res_filename.split("\\")[1].split("-")[2]
    
    for candidate in candidates:
        if match_string in candidate:
            return candidate
    return None


def validate(model, loss_fn, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() 
                                else "cpu")
    val_loss, val_acc, epoch_hard_preds, epoch_labels = [], [], [], []
    inf_batch_time = []
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):

            inputs, labels = x.to(device), y.to(device)
            t1 = time.time()
            teacher_pred = model.forward(inputs)
            t2 = time.time()
            inf_batch_time.append(t2-t1)
            loss = loss_fn(teacher_pred, labels)
            val_loss.append(loss.item())
            hard_preds = F.softmax(teacher_pred, dim=1).argmax(dim=1)
            acc_batch_avg = (hard_preds.to(device) == labels).float().mean().item()
            val_acc.append(acc_batch_avg)
            epoch_hard_preds.extend(hard_preds.cpu())
            epoch_labels.extend(labels.cpu())
    return sum(val_loss)/len(val_loss), sum(val_acc)/len(val_acc), epoch_hard_preds, epoch_labels, median(inf_batch_time)


if __name__ == "__main__":
    with open('eval_config.json', 'r') as f:
        config = json.load(f)
    run_experiment(config)
    