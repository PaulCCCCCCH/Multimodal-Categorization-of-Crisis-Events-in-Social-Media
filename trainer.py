import os
import torch
from torch import nn
from torch._C import dtype
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging
import numpy as np
import sys


class Trainer:
    def __init__(self, train_loader, dev_loader, test_loader, model: nn.Module, loss_fn, optimizer, scheduler, save_dir='.', display=100, eval=False, device='cuda', tensorboard=False, mode='both'):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

        self.display = display

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        self.eval = eval

        self.device = device

        self.tensorboard = tensorboard
        if mode == 'both':
            self.label_key = 'label'
        elif mode == 'image_only':
            self.label_key = 'label_image'
        elif mode =='text_only':
            self.label_key = 'label_text'

        if not eval and tensorboard:
            self.writer = SummaryWriter()

    def train(self, max_iter):
        if self.device != 'cpu':
            self.scaler = torch.cuda.amp.GradScaler()

        best_dev_loss = float('inf')

        for idx_iter in range(max_iter):
            logging.info("Training iteration {}".format(idx_iter))
            correct = 0
            display_correct = 0
            total = 0
            display_total = 0
            total_loss = 0
            display_total_loss = 0
            batch = 0
            for data in tqdm(self.train_loader, total=len(self.train_loader)):
                # for data in self.train_loader:
                self.model.train()
                self.model.zero_grad()

                x = (data['image'].to(self.device),
                     {k: v.to(self.device) for k, v in data['text_tokens'].items()})
                y = data[self.label_key].to(self.device)

                # For mixed-precision training
                if self.device != 'cpu':
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss = self.loss_fn(logits, y)

                    total_loss += loss.item()

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    display_total_loss += loss.item()
                else:

                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                    total_loss += loss.item()

                    loss.backward()
                    self.optimizer.step()

                    display_total_loss += loss.item()

                indices = torch.argmax(logits, dim=1)
                batch_correct = sum(indices == y).item()

                correct += batch_correct
                display_correct += batch_correct

                total += x[0].shape[0]
                display_total += x[0].shape[0]

                batch += 1
                if batch % self.display == 0:
                    display_loss = display_total_loss / display_total
                    display_acc = display_correct / display_total
                    # logging.info("Correct: {}".format(display_correct))
                    # logging.info("Total: {}".format(display_total))
                    logging.info("Finished {} / {} batches with loss: {}, accuracy {}"
                          .format(batch, len(self.train_loader), display_loss, display_acc))
                    total_batch = idx_iter * len(self.train_loader) + batch

                    if self.tensorboard:
                        self.writer.add_scalar(
                            'Train Batch Loss', display_loss, total_batch)
                        self.writer.add_scalar(
                            'Train Batch Acc', display_acc, total_batch)

                    display_correct = 0
                    display_total = 0
                    display_total_loss = 0

            logging.info("=============Iteration {}=============".format(idx_iter))
            logging.info("Training accuracy {}".format(correct / total))
            logging.info("Avg Training loss {}".format(total_loss / total))
            logging.info("Saving model...")
            self.model.save('checkpoint_{}'.format(idx_iter))
            logging.info("done")
            logging.info("Calculating validation loss...")
            del x  # save some memory here before validating
            del y
            dev_loss = self.validate(idx_iter)
            if dev_loss < best_dev_loss:
                self.model.save('best')
                best_dev_loss = dev_loss

            self.scheduler.step(dev_loss)
            logging.info("======================================\n".format(idx_iter))

        self.predict()

    def validate(self, idx_iter=0):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0

        for data in self.dev_loader:

            x = (data['image'].to(self.device),
                 {k: v.to(self.device) for k, v in data['text_tokens'].items()})
            y = data[self.label_key].to(self.device)

            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            total_loss += loss.item()

            indices = torch.argmax(logits, dim=1)
            correct += sum(indices == y).item()
            total += x[0].shape[0]

        dev_acc = correct / total
        dev_loss = total_loss / total
        logging.info("Dev set accuracy {}".format(dev_acc))
        logging.info("Dev set loss {}".format(dev_loss))
        if not self.eval and self.tensorboard:
            self.writer.add_scalar('Dev Loss', dev_loss, idx_iter)
            self.writer.add_scalar('Dev Acc', dev_acc, idx_iter)
        return dev_loss

    def predict(self):
        self.model.eval()
        predictions = []
        correct = 0
        total = 0
        for data in self.test_loader:

            x = (data['image'].to(self.device),
                 {k: v.to(self.device) for k, v in data['text_tokens'].items()})
            y = data[self.label_key].to(self.device)

            logits = self.model(x)

            # indices is a tensor of predictions
            indices = torch.argmax(logits, dim=1).to(dtype=torch.int32)
            correct += sum(indices == y).item()

            total += x[0].shape[0]
            predictions.extend([pred.item() for pred in indices])
            with open('prediction.csv', 'w') as f:
                for pred in predictions:
                    f.write("{}\n".format(str(pred)))
        logging.info("Test set accuracy: {}".format(correct / total))

        return predictions
