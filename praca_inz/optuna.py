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
from tensorboardX import SummaryWriter
from utils import compute_metrics
from utils import to_var, load_config_from_json

from torch.utils.data import DataLoader
from modcloth import ModCloth
from model import SFNet


data_config = load_config_from_json("configs/data.jsonnet")
splits = ["train", "valid"]

datasets = OrderedDict()
for split in splits:
    datasets[split] = ModCloth(data_config, split=split)


def objective(trial):
    

