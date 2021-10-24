"""
@author: Chonghan Chen <chonghac@cs.cmu.edu>
"""

import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from transformers import BertTokenizer, BertModel

from mm_models import DenseNetBertMMModel
from crisismmd_dataset import CrisisMMDataset
from trainer import Trainer

from args import get_args
opt = get_args

model_to_load = None

device = 'cuda'
num_workers = 12

EVAL = False
USE_TENSORBOARD = False

TASK = 'task1'
MAX_ITER = 30
OUTPUT_SIZE = 2 if TASK == 'task1' else 8
DIM_VISUAL_REPR = 1000
DIM_TEXT_REPR = 756
DIM_PROJ = 756

# General hyper parameters
learning_rate = 2e-3
batch_size = 20

if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, dev_loader = None, None
    if not EVAL:
        train_set = CrisisMMDataset(opt, phase='train', cat='all', task='task2', tokenizer=tokenizer)
        train_loader = DataLoader(
            train_set.data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dev_set = CrisisMMDataset(opt, phase='dev', cat='all', task='task2', tokenizer=tokenizer)
    dev_loader = DataLoader(
        dev_set.data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = CrisisMMDataset(opt, phase='test', cat='all', task='task2', tokenizer=tokenizer)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loss_fn = nn.CrossEntropyLoss()
    model = DenseNetBertMMModel()

    # The authors did not mention configurations of SGD. We assume they did not use momentum or weight decay.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                       
    # The authors used factor=0.1, but did not mention other configs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, cooldown=0, verbose=True)


    trainer = Trainer(train_loader, dev_loader, test_loader,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD)

    if model_to_load:
        model.load(model_to_load)
        print("\n***********************")
        print("Model Loaded!")
        print("***********************\n")
    else:
        print("No previous model loaded. Training new model!")

    if not EVAL:
        print("\n================Training Summary=================")
        print("Training Summary: ")
        print("Learning rate {}".format(learning_rate))
        print("Batch size {}".format(batch_size))
        print(trainer.model)
        print("\n=================================================")

        trainer.train(MAX_ITER)

    else:
        print("\n================Evaluating Model=================")
        print(trainer.model)

        trainer.validate()
