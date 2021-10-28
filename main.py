"""
@author: Chonghan Chen <chonghac@cs.cmu.edu>
"""

from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset, CrisisMMDatasetWithSSE
from mm_models import DenseNetBertMMModel
import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

import nltk
nltk.download('stopwords')


if __name__ == '__main__':
    opt = get_args()

    model_to_load = None

    device = 'cuda'
    num_workers = 0

    EVAL = False
    USE_TENSORBOARD = False

    TASK = 'task1'
    MAX_ITER = 300
    OUTPUT_SIZE = 2 if TASK == 'task1' else 8
    DIM_VISUAL_REPR = 1000
    DIM_TEXT_REPR = 756
    DIM_PROJ = 756

    # The authors did not report the following values, but they tried
    # pv, pt in [10, 20000], and pv0, pt0 in [0, 1]
    WITH_SSE = True
    pv = 1000
    pt = 1000
    pv0 = 0.5
    pt0 = 0.5

    # General hyper parameters
    learning_rate = 2e-3
    batch_size = 8

    # Tokenizer for bert

    train_loader, dev_loader = None, None
    if not EVAL:
        if WITH_SSE:
            train_set = CrisisMMDatasetWithSSE()
            train_set.initialize(opt, pv, pt, pv0, pt0, phase='train', cat='all',
                                 task=TASK)
        else:
            train_set = CrisisMMDataset()
            train_set.initialize(opt, phase='train', cat='all',
                                 task=TASK)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if WITH_SSE:
        dev_set = CrisisMMDatasetWithSSE()
        dev_set.initialize(opt, pv, pt, pv0, pt0, phase='dev', cat='all',
                           task=TASK)
    else:
        dev_set = CrisisMMDataset()
        dev_set.initialize(opt, phase='dev', cat='all',
                           task=TASK)

    dev_loader = DataLoader(
        dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if WITH_SSE:
        test_set = CrisisMMDatasetWithSSE()
        test_set.initialize(opt, pv, pt, pv0, pt0, phase='test', cat='all',
                            task=TASK)
    else:
        test_set = CrisisMMDataset()
        test_set.initialize(opt, phase='test', cat='all',
                            task=TASK)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loss_fn = nn.CrossEntropyLoss()
    model = DenseNetBertMMModel().to(device)

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
