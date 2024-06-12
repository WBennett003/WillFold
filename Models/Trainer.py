import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from Utils.dataset import mmcif_dataset

class Trainer:
    def __init__(self, trainer_config_filepath, dataset_samples=[]):
        with open(trainer_config_filepath, 'r') as f:
            self.config = json.load(f)

    def eval(self, inputs, targets):
        pass

    def log(self, log_info):
        pass

    def train(): 
        pass