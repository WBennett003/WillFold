import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
import glob

from Utils.dataset import mmcif_dataset, collate
from Models.Modules import AlphaFold3_TEMPLE

class Trainer:
    def __init__(self, config_filepath=None, config=None):
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if config is None:
            with open(config_filepath, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        self.device = self.config['model']['device']

        training, test, val = self.config['training_split']

        dataset_samples = ['1A00', '1A0A', '1A0B', '1A0C', '1A0D', '1A0E', '1A0F', '1A0G', '1A0H', '1A0I'] #glob.glob(os.path.join(self.config['mmcif_path'], "*", "*.cif"))        
        
        self.Ntokens = self.config['Ntokens']
        self.Natoms = self.config['Natoms']

        total_length = len(dataset_samples)
        self.train_length = int(total_length * training)
        self.test_length = int(total_length * test)
        self.val_length = int(total_length * val)

        training_data = mmcif_dataset(self.config['mmcif_path'], samples=dataset_samples[:self.train_length], Ntokens=self.Ntokens, Natoms=self.Natoms, device=self.device)

        self.dataloader = DataLoader(training_data, self.config['batch_size'], shuffle=True, collate_fn=collate)

        self.model = AlphaFold3_TEMPLE(self.config['model']).to(self.device)

    def eval(self, inputs, targets):
        pass

    def log(self, log_info):
        pass

    def train(self): 
        optimizer = torch.optim.AdamW(self.model.parameters())
        # optimizer.load_state_dict(torch.load('clip_optim.pt'))
        if self.config['schedule_type'] == 'one':
            schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], epochs=self.config['EPOCHS'], steps_per_epoch=self.config['EPOCH_STEPS'])
        else:
            schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config['lr'])

        if self.device.type == "cuda" and self.config['scaleing']: #reduced precision to improve performance https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#4-use-automatic-mixed-precision-amp-
            scaler = torch.cuda.amp.GradScaler()
            torch.backends.cudnn.benchmark = True #improve performance
        elif self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True #improve performance

        for epoch in range(self.config['EPOCHS']):
            loss_sum = 0
            prev_batch = 0
            batch_idx = torch.randperm(self.train_length)
            
            for f, xyz in self.dataloader:
                optimizer.zero_grad(set_to_none=True) #i dont even know

                if self.config['scaleing']:
                    with torch.cuda.amp.autocast():
                        pred = self.model(f)
                        loss = F.mse_loss(pred, xyz)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self.model(f)
                    loss = F.mse_loss(pred, xyz)
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.detach()

