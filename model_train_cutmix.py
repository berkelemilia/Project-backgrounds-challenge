# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:46:28 2022

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
from tools.datasets_training import ImageNet, ImageNet9_training
from tools.model_utils import make_and_restore_model, eval_model
from imagenet_models.resnet import resnet50
from tqdm import tqdm as tqdm
from pathlib import Path, PureWindowsPath
from torch.autograd import Variable
from tools.loading_model import precomputed_model

parser = ArgumentParser()
parser.add_argument('--arch', default='resnet18',
                    help='Model architecture, if loading a model checkpoint.')
parser.add_argument('--datapath', default='original',
                    help='Emplacement données')
parser.add_argument('--n-model', default='',
                    help="Nom du modèle que l'on doit entrainer")
parser.add_argument('--precompute', default='False',
                    help="Dit si le modèle que l'on entraine est déjà précompute ou non")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_model(model,loader,path):
    
    model.train()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ep = 10
    
    for epoch in range(ep):
        ch.save(model, path )
        print("number of epoch=" + str(epoch) + "/" + str(ep))
        running_loss = 0.0
        iterator = tqdm(enumerate(loader), total=len(loader))
        
        for i, (inp, label) in iterator:
            inp = inp.cuda()
            label = label.cuda()
        
        
            lam = np.random.beta(1, 1)
            rand_index = ch.randperm(inp.size()[0]).cuda()
            target_a = label
            target_b = label[rand_index]
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(inp.size(), lam)
            inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.size()[-1] * inp.size()[-2]))
            # compute output
            output = model(inp)
            
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            
            
            
            optimizer.zero_grad()
            
            outputs = model(inp)
            outputs.cuda()
            loss = criterion(outputs, label.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')

            
    ch.save(model, path )

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
    val_loader = in9_ds.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
    
    if pr=="False": 
        model, _ = make_and_restore_model(arch = arch, dataset = in9_ds, pytorch_pretrained=False )
        print("Modèle non préentrainé")
    else:
        model = precomputed_model(arch = 18)
        print("Modèle préentrainé")
    
    
    
    model.cuda()
    model.train()
    model = nn.DataParallel(model)
    
    
    train_model(model,val_loader,fp)

    
        
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)