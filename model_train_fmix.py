# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:01:53 2022

@author: Wasik
"""


import torch.optim as optim
from torchvision import transforms
import torch as ch
import torch.nn as nn
import numpy as np
import json
import time
from argparse import ArgumentParser
from tools.datasets import ImageNet9
from tools.datasets_training import ImageNet, ImageNet9_training
from tools.model_utils import make_and_restore_model, eval_model
from imagenet_models.resnet import resnet50
from tqdm import tqdm as tqdm
from pathlib import Path, PureWindowsPath
from tools.loading_model import precomputed_model
from torchvision import datasets, transforms, models
import torch
from torch import optim
from fmix.lightning import FMix
from pytorch_lightning import LightningModule, Trainer, data_loader



parser = ArgumentParser()
parser.add_argument('--arch', default='resnet18',
                    help='Model architecture, if loading a model checkpoint.')
parser.add_argument('--datapath', default='original',
                    help='Emplacement données')
parser.add_argument('--n-model', default='',
                    help="Nom du modèle que l'on doit entrainer")
parser.add_argument('--precompute', default='False',
                    help="Dit si le modèle que l'on entraine est déjà précompute ou non")


    
class FMixExp(LightningModule):
    def __init__(self,model,trainloader,valloader):
        super().__init__()
        self.net = model
        self.fmix = FMix()
        self.trainloader = trainloader
        self.valloader =valloader

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = self.fmix(x)

        x = self.forward(x)

        loss = self.fmix.loss(x, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        x = self.forward(x)

        labels_hat = torch.argmax(x, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        loss = self.fmix.loss(x, y, train=False)
        output = {
            'val_loss': loss,
            'val_acc': val_acc,
        }

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    @data_loader
    def train_dataloader(self):
        return self.trainloader

    @data_loader
    def val_dataloader(self):
        return self.valloader
    
    


def main(args):
    BATCH_SIZE = 26
    WORKERS = 8
    
    arch = args.arch
    data_path = args.datapath
    n_model = args.n_model
    pr = args.precompute
    fp=Path(r'C:\Users\Wasik\MAP583\Project\backgrounds_challenge-data\Nos_classifieurs', n_model+'.pt')
    #Chargement des données
    in9_ds = ImageNet9_training(f'{data_path}')
    train_loader = in9_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
    in9_ds_val = ImageNet9(f'{data_path}')
    val_loader = in9_ds_val.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
    
    if pr=="False": 
        model, _ = make_and_restore_model(arch = arch, dataset = in9_ds, pytorch_pretrained=False )
        print("Modèle non préentrainé")
    else:
        model = precomputed_model(arch = 18)
        print("Modèle préentrainé")
    
    
    print('==> Starting training..')
    trainer = Trainer(gpus=1, early_stop_callback=False, max_epochs=10, checkpoint_callback=False)
    mod = FMixExp(model,train_loader,val_loader)
    trainer.fit(mod)
    
    print("training fini")
    
    ch.save(model, fp )

    
        
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)