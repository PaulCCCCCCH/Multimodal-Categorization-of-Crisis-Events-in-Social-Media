"""
@author: Chonghan Chen <chonghac@cs.cmu.edu>
"""
from os import path as osp
import os
import logging
from PIL.Image import SAVE
from torch.serialization import save
from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset, CrisisMMDatasetWithSSE
from mm_models import DenseNetBertMMModel, ImageOnlyModel, TextOnlyModel
import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import time

import nltk
nltk.download('stopwords')


if __name__ == '__main__':
    opt = get_args()

    model_to_load = opt.model_to_load
    image_model_to_load = opt.image_model_to_load
    text_model_to_load = opt.text_model_to_load

    device = opt.device
    num_workers = opt.num_workers

    EVAL = opt.eval
    USE_TENSORBOARD = opt.use_tensorboard
    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))

    MODE = opt.mode
    TASK = opt.task
    MAX_ITER = opt.max_iter
    OUTPUT_SIZE = 2 if TASK == 'task1' else 8

    # The authors did not report the following values, but they tried
    # pv, pt in [10, 20000], and pv0, pt0 in [0, 1]
    WITH_SSE = opt.with_sse
    pv = opt.pv # How many times more likely do we transit to the same class
    pt = opt.pt 
    pv0 = opt.pv0  # Probability of not doing a transition
    pt0 = opt.pt0

    # General hyper parameters
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # Create folder for saving
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)


    # set logger
    logging.basicConfig(filename=osp.join(save_dir, 'output_{}.log'.format(int(time.time()))), level=logging.INFO)


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

    dev_set = CrisisMMDataset()
    dev_set.initialize(opt, phase='dev', cat='all',
                       task=TASK)

    dev_loader = DataLoader(
        dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = CrisisMMDataset()
    test_set.initialize(opt, phase='test', cat='all',
                        task=TASK)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    loss_fn = nn.CrossEntropyLoss()
    if MODE == 'text_only':
        model = TextOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'image_only':
        model = ImageOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'both':
        model = DenseNetBertMMModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    else:
        raise NotImplemented

    # The authors did not mention configurations of SGD. We assume they did not use momentum or weight decay.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # The authors used factor=0.1, but did not mention other configs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, cooldown=0, verbose=True)

    trainer = Trainer(train_loader, dev_loader, test_loader,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD, mode=MODE)

    if model_to_load:
        model.load(model_to_load)
        logging.info("\n***********************")
        logging.info("Model Loaded!")
        logging.info("***********************\n")
    if text_model_to_load:
        model.load(text_model_to_load)
    if image_model_to_load:
        model.load(image_model_to_load)

    if not EVAL:
        logging.info("\n================Training Summary=================")
        logging.info("Training Summary: ")
        logging.info("Learning rate {}".format(learning_rate))
        logging.info("Batch size {}".format(batch_size))
        logging.info(trainer.model)
        logging.info("\n=================================================")

        trainer.train(MAX_ITER)

    else:
        logging.info("\n================Evaluating Model=================")
        logging.info(trainer.model)

        trainer.validate()
        trainer.predict()
