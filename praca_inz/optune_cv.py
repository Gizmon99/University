# Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
# Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

import os
import time
import jsonlines
import argparse
import torch
import numpy as np
import optuna

from collections import defaultdict, OrderedDict
from utils import compute_metrics
from utils import to_var, load_config_from_json

from torch.utils.data import DataLoader
from modcloth import ModCloth
from model import SFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


data_config = load_config_from_json("configs/data.jsonnet")
model_config = load_config_from_json("configs/model.jsonnet")
splits = ["train", "valid"]

datasets = OrderedDict()
datasets["full"] = ModCloth(data_config, split="full")
k_data = []

for line in datasets["full"]:
    k_data.append(line)

kfolds = [k_data[i*16558: (i+1)*16558] for i in range(5)]


def objective(trial):
    embedding_dim = trial.suggest_int("embedding_dim", 5, 20)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    dicto = {"embedding_dim": embedding_dim,
        "num_item_emb" : 1378,
        "num_category_emb" : 7,
        "num_cup_size_emb" : 12,
        "num_user_emb" : 47958,
        "num_user_numeric": 5,
        "num_item_numeric": 2,
        "user_pathway": [256, 128, 64],
        "item_pathway": [256, 128, 64],
        "combined_pathway": [256, 128, 64, 16],
        "activation": "relu",
        "dropout": dropout,
        "num_targets": 3}
    
    model = SFNet(dicto)
    model = model.to(device)

    loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    # m = torch.nn.LogSoftmax(dim=1)

    lr = trial.suggest_float("lr", 0.0001, 0.005)
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.005)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    step = 0
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    result = 0

    for i in range(5):
        for epoch in range(model_config["trainer"]["num_epochs"]):

            for split in splits:

                if split == "train":
                    data_loader = DataLoader(
                        dataset=kfolds[(i+1)%5]+kfolds[(i+2)%5]+kfolds[(i+3)%5]+kfolds[(i+4)%5],
                        batch_size=model_config["trainer"]["batch_size"],
                        shuffle=split == "train",
                    )
                else:
                    data_loader = DataLoader(
                        dataset=kfolds[i],
                        batch_size=model_config["trainer"]["batch_size"],
                        shuffle=split == "valid",
                    )
                loss_tracker = defaultdict(tensor)

                # Enable/Disable Dropout
                if split == "train":
                    model.train()
                else:
                    model.eval()
                    target_tracker = []
                    pred_tracker = []

                for iteration, batch in enumerate(data_loader):

                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = to_var(v)

                    # Forward pass
                    logits, pred_probs = model(batch)

                    # loss calculation
                    loss = loss_criterion(logits, batch["fit"])
                    # loss = loss_criterion(m(logits), batch["fit"])

                    # backward + optimization
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step += 1

                    # bookkeepeing
                    loss_tracker["Total Loss"] = torch.cat(
                        (loss_tracker["Total Loss"], loss.view(1))
                    )

                    if iteration % model_config["logging"][
                        "print_every"
                    ] == 0 or iteration + 1 == len(data_loader):
                        print(
                            "{} Batch Stats {}/{}, Loss={:.2f}".format(
                                split.upper(), iteration, len(data_loader) - 1, loss.item()
                            )
                        )

                    if split == "valid":
                        target_tracker.append(batch["fit"].cpu().numpy())
                        pred_tracker.append(pred_probs.cpu().data.numpy())

                print(
                    "%s Epoch %02d/%i, Mean Total Loss %9.4f"
                    % (
                        split.upper(),
                        epoch + 1,
                        model_config["trainer"]["num_epochs"],
                        torch.mean(loss_tracker["Total Loss"]),
                    )
                )

            if split == "valid" and model_config["logging"]["tensorboard"]:
                # not considering the last (incomplete) batch for metrics
                target_tracker = np.stack(target_tracker[:-1]).reshape(-1)
                pred_tracker = np.stack(pred_tracker[:-1], axis=0).reshape(
                    -1, model_config["sfnet"]["num_targets"]
                )
                precision, recall, f1_score, accuracy, auc = compute_metrics(
                    target_tracker, pred_tracker
                )
        result += accuracy

    return result/5


if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize"])
    study.optimize(objective, n_trials=30)

    trials = study.best_trials
    print("Best trials: ", trials)

    for c, trial in enumerate(trials):
        print("Trial nr: ", c, ", values: ", trial.values)
        print("Params:")
        for key, value in trial.params.items():
            print(" {}: {},".format(key, value))
