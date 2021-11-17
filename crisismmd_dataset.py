import os
import torch
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

from transformers import BertTokenizer
from preprocess import clean_text

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

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text,	label_image, label_text_image = l.split(
                '\t')
            self.data_list.append(
                {
                    'path_image': '%s/%s' % (self.dataset_root, image),

                    'text': tweet_text,
                    'text_tokens': self.tokenize(tweet_text),

                    'label_str': label,
                    'label': self.label_map[label],

                    'label_image_str': label_image,
                    'label_image': self.label_map[label_image],

                    'label_text_str': label_text,
                    'label_text': self.label_map[label_text]
                }
            )

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}

    def initialize(self, opt, phase='train', cat='all', task='task2', shuffle=False):
        self.opt = opt
        self.shuffle = shuffle

        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0_toy' if opt.debug else f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = labels_task1 if task == 'task1' else labels_task2

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        # Append list of data to self.data_list
        self.read_data(ann_file)

        if self.shuffle:
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
            # transforms.Resize((opt.crop_size, opt.crop_size)),
            transforms.RandomCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        data = self.data_list[index]
        if 'image' not in data:
            with Image.open(data['path_image']).convert('RGB') as img:
                image = self.transforms(img)
            data['image'] = image

        return data

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'


class CrisisMMDatasetWithSSE(CrisisMMDataset):

    def initialize(self, opt, pv, pt, pv0, pt0, phase='train', cat='all', task='task2'):
        super(CrisisMMDatasetWithSSE, self).initialize(
            opt, phase=phase, cat=cat, task=task)
        self.pv = pv
        self.pt = pt
        self.pv0 = pv0
        self.pt0 = pt0
        # Probability of transition to a connected image embedding
        self.p_img_conn = pv / (1 + pv)
        # Probability of transition to a connected text embedding
        self.p_txt_conn = pt / (1 + pt)
        self.build_trainsition_probs()

    def build_trainsition_probs(self):
        # Transition probability from one class to another for every class pairs
        self.transition_probs = {}
        for class_a in self.class_dict:
            len_class_a = self.class_lengths[class_a]
            len_non_class_a = sum(self.class_lengths.values()) - len_class_a
            transition_prob = []
            for class_b in self.class_dict:
                if class_a == class_b:
                    continue
                len_class_b = self.class_lengths[class_b]
                transition_prob.append(
                    (class_b, len_class_b / len_non_class_a))
            self.transition_probs[class_a] = transition_prob

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        # self.data_list stores samples
        self.data_list = []
        # self.class_dict stores {label: [sample_idx]}
        self.class_dict = {}
        self.class_lengths = {}

        for idx, l in enumerate(self.info):
            l = l.rstrip('\n')
            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text,	label_image, label_text_image = l.split(
                '\t')
            mapped_label = self.label_map[label]
            self.data_list.append(
                {
                    'path_image': '%s/%s' % (self.dataset_root, image),

                    'text': tweet_text,
                    'text_tokens': self.tokenize(tweet_text),

                    'label_str': label,
                    'label': mapped_label,

                    'label_image_str': label_image,
                    'label_image': self.label_map[label_image],

                    'label_text_str': label_text,
                    'label_text': self.label_map[label_text]
                }
            )

            if mapped_label in self.class_dict:
                self.class_dict[mapped_label].append(idx)
            else:
                self.class_dict[mapped_label] = []
        self.class_lengths = {class_idx: len(
            indices) for class_idx, indices in self.class_dict.items()}

    def should_do(self, p):
        # Toss a biased coin that gives a head with a probability of p
        # Returns True if it results in head.
        if np.random.random() > p:
            return False
        return True

    def transit_same_class(self, curr_class, curr_idx):
        while True:
            target_idx = np.random.choice(self.class_dict[curr_class])
            if target_idx != curr_idx:
                return self.data_list[target_idx]

    def get_transit_data(self, curr_class, curr_idx):
        should_keep_same_class = self.should_do(self.p_img_conn)
        if should_keep_same_class:
            # Transit to the same class, but not to itself
            target_data = self.transit_same_class(curr_class, curr_idx)
        else:
            # Transit to another class
            rand_idx = np.random.choice(len(self.transition_probs[curr_class]), p=[
                p[1] for p in self.transition_probs[curr_class]])
            target_class = self.transition_probs[curr_class][rand_idx][0]
            target_data_idx = np.random.choice(self.class_dict[target_class])
            target_data = self.data_list[target_data_idx]
        return target_data

    def __getitem__(self, index):
        data = self.data_list[index]
        curr_class = data['label']

        # Make transition on the image side
        should_transit_image = self.should_do(self.pv0)
        if should_transit_image:
            target_data = self.get_transit_data(curr_class, index)
            data['path_image'] = target_data['path_image']

        # Do the same on the text side
        should_transit_text = self.should_do(self.pt0)
        if should_transit_text:
            target_data = self.get_transit_data(curr_class, index)
            for attr in ['text', 'text_tokens']:
                data[attr] = target_data[attr]

        # Open image and assign as before
        data = self.data_list[index]
        if 'image' not in data:
            with Image.open(data['path_image']).convert('RGB') as img:
                image = self.transforms(img)
            data['image'] = image

        return data

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDatasetWithSSE'


if __name__ == '__main__':

    opt = object()

    dset = CrisisMMDataset(opt, 'train')
    import pdb
    pdb.set_trace()
