import torch
import numpy as np
from tqdm import tqdm
from torch import optim

from dataset import SingleLabelDataset
from torch.utils.data import DataLoader
import torch.nn as nn

from models import ClassifyModel
from tokenizer import Tokenizer
import os
import torch.nn.functional as F


class Experiment1Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_bert_path = "pretrained_models/nezha-cn-base"
        self.tokenizer = Tokenizer(device=self.device, pretrained_bert_path=self.pretrained_bert_path)
        self.batch_size = 120
        self.class_num = 17


global_config = Experiment1Config()


def load_original_data(data_dir="data/tnews_public"):
    train_data = []
    with open("{}/train.txt".format(data_dir), 'r', encoding="utf8") as f:
        for line in f.readlines():
            parts = line.split("_")
            label = parts[0]
            content = parts[2]
            item = content.strip(), label
            train_data.append(item)

    validate_data = []
    with open("{}/dev.txt".format(data_dir), 'r', encoding="utf8") as f:
        for line in f.readlines():
            parts = line.split("_")
            label = parts[0]
            content = parts[2]
            item = content, label
            validate_data.append(item)

    return train_data, validate_data


def collate_fn(examples):
    sentences = np.array([ex[0] for ex in examples])
    inputs = global_config.tokenizer.to_tensor(sentences)
    targets = torch.tensor([[int(ex[1]) - 100] for ex in examples])
    ach = torch.tensor([[0], [global_config.class_num - 1]])
    targets = torch.cat([targets, ach], 0)
    targets = F.one_hot(targets)
    targets = targets.squeeze(1)
    targets = targets[:global_config.batch_size, :]
    return inputs, targets


def train(total_epoch=100, pth_path="", load_data=load_original_data, lr=0.01, print_log=True,
          is_decline_lr=False):
    device = global_config.device
    train_data, validate_data = load_data()

    train_dataset = SingleLabelDataset(train_data)
    validate_dataset = SingleLabelDataset(validate_data)

    train_data_loader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collate_fn,
                                   batch_size=global_config.batch_size)
    validate_data_loader = DataLoader(dataset=validate_dataset, shuffle=True, collate_fn=collate_fn,
                                      batch_size=global_config.batch_size)
    model = ClassifyModel(class_num=global_config.class_num)
    loss_fun = nn.CrossEntropyLoss().cuda()
    best_params = None
    best_evaluate_loss = None
    evaluate_losses = None
    last_lr = 0.1
    lrs = None
    if os.path.exists(pth_path):
        checkpoint = torch.load(pth_path, map_location=device)
        best_params = checkpoint['params']
        best_evaluate_loss = checkpoint['best_evaluate_loss']
        evaluate_losses = checkpoint['evaluate_losses']
        lrs = checkpoint['lrs']
        last_lr = checkpoint['last_lr']

    if best_params is not None:
        model.load_state_dict(best_params)
    model.to(device)

    composes = pth_path.split("/")
    file_name = composes[len(composes) - 1]
    lr = lr if lr < last_lr else last_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)

    for epoch in range(total_epoch):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_data_loader, desc=F"Training Epoch{epoch}")):
            inputs, targets = [x for x in batch]
            outputs = model(inputs)
            if print_log:
                print("===================================outputs=======================================")
                print(outputs)
                print("===================================targets=======================================")
                print(targets)

            outputs.to(device)
            targets.to(device)
            loss = loss_fun(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print(f"Loss:{total_loss / (i + 1):.2f}")

        evaluate_loss = 0
        average_loss = None
        for i, batch in enumerate(validate_data_loader):
            inputs, targets = [x for x in batch]
            outputs = model(inputs)
            outputs.to(device)
            targets.to(device)
            loss = loss_fun(outputs, targets)
            evaluate_loss += loss.item()
            average_loss = evaluate_loss / (i + 1)

        if evaluate_losses is None:
            evaluate_losses = []
        if lrs is None:
            lrs = []
        evaluate_losses.append(average_loss)
        lrs.append(lr)

        if best_evaluate_loss is None or average_loss < best_evaluate_loss:
            best_evaluate_loss = average_loss
            best_params = model.state_dict()
        print(
            "current_lr:{} pth_path:{} loss:{}/{}".format(lr, file_name, average_loss, best_evaluate_loss)
        )

    state = {
        'params': best_params,
        'best_evaluate_loss': best_evaluate_loss,
        'last_lr': lr,
        'lrs': lrs,
        'evaluate_losses': evaluate_losses,
    }
    torch.save(state, pth_path)

    if is_decline_lr is True:
        train(total_epoch, pth_path, load_data, lr / 10, model, False, lr > 0.00001)
    else:
        best_evaluate_loss = best_evaluate_loss * 1000
        decimal_part = str(int(best_evaluate_loss))
        parts = pth_path.split("/")
        name = parts[len(parts) - 1]
        parts = parts[0:len(parts) - 1]
        pth_path_new_name = name.split(".")[0] + "_" + decimal_part + ".pth"
        path_new = ""
        for p in parts:
            path_new = path_new + p + "/"
        path_new = path_new + pth_path_new_name
        os.rename(pth_path, path_new)


if __name__ == '__main__':
    train()
