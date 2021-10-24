import os
import numpy as np
from imageio import imread
from PIL import Image
import glob

from termcolor import colored, cprint

from preprocess import clean_text

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision.utils as vutils
from torchvision import datasets

from base_dataset import BaseDataset
from base_dataset import scale_shortside

from paths import dataroot

task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'affected_individuals': 0,
    'infrastructure_and_utility_damage': 1,
    'injured_or_dead_people': 2,
    'missing_or_found_people': 3,
    'not_humanitarian': 4,
    'other_relevant_information': 5,
    'rescue_volunteering_or_donation_effort': 6,
    'vehicle_damage': 7,
}


class CrisisMMDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', task='task2', tokenizer=None):
        self.opt = opt

        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = labels_task1 if task == 'task1' else labels_task2
        self.tokenizer = tokenizer

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text,	label_image, label_text_image = l.split(
                '\t')
            self.data_list.append(
                {
                    'image': image,
                    'text_tokens': tokenizer(tweet_text) if self.tokenizer is not None else tweet_text,
                    'text': tweet_text,
                    'label': label,
                    'label_image': label_image,
                    'label_text': label_text,
                    'label_text_image': label_text_image
                }
            )

        np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.transforms = transforms.Compose([
            # transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            transforms.Lambda(lambda img: scale_shortside(
                img, opt.load_size, opt.crop_size, Image.BICUBIC)),
            transforms.RandomCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):

        d = self.data_list[index]
        path_image = '%s/%s' % (self.dataset_root, d['image'])
        label_image_str = d['label_image']
        label_image = self.label_map[label_image_str]
        tweet_text = d['text']
        tweet_tokens = d['text_tokens']
        label_text_str = d['label_text']
        label_text = self.label_map[label_text_str]
        image = Image.open(path_image).convert('RGB')
        image = self.transforms(image)
        image = np.array([0])

        ret = {
            'image': image,
            'label_image': label_image,
            'text': tweet_text,
            'label_text': label_text,
            'text_tokens': tweet_tokens,
            'label_image_str': label_image_str,
            'label_text_str': label_text_str,
            'path_image': path_image,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'SDFDataset'


if __name__ == '__main__':

    opt = object()

    dset = CrisisMMDataset(opt, 'train')
    import pdb
    pdb.set_trace()
