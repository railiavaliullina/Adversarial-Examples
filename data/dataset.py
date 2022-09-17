import numpy as np
import torch
from torchvision import transforms as transforms_
from torchvision.datasets import ImageFolder


class SOPDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_type):
        """
        Class for getting SOP dataset
        :param cfg: cfg['data'] part of config
        :param dataset_type: type of data ('train' or 'valid')
        """
        cfg_aug = cfg['augmentation']
        self.sz_crop = cfg_aug['sz_crop']
        self.sz_resize = cfg_aug['sz_resize']
        self.mean = cfg_aug['mean']
        self.std = cfg_aug['std']
        self.contrast = cfg_aug['contrast']
        self.saturation = cfg_aug['saturation']
        self.brightness = cfg_aug['brightness']

        self.nb_classes = cfg['nb_classes']
        self.dataset_type = dataset_type
        # directory with all images
        self.dataset_path = cfg['dataset_path']  # + dataset_type + "/"

        if dataset_type == 'train':
            transforms = transforms_.Compose([
                transforms_.RandomResizedCrop(self.sz_crop),
                transforms_.RandomHorizontalFlip(),
                transforms_.ColorJitter(contrast=self.contrast, saturation=self.saturation, brightness=self.brightness),
                transforms_.ToTensor(),
                transforms_.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            ])
        elif dataset_type == 'valid':
            transforms = transforms_.Compose([
                    transforms_.Resize(self.sz_resize),
                    transforms_.CenterCrop(self.sz_crop),
                    transforms_.ToTensor(),
                    transforms_.Normalize(
                        mean=self.mean,
                        std=self.std,
                    )
                ])
        else:
            raise Exception

        with open('../filenames.txt', 'r') as f:
            t = f.readlines()

        filenames = [t_.replace('\n', '').replace('/', '_') for t_ in t]

        print(f'Creating ImageFolder for {dataset_type} set...')
        self.image_folder = ImageFolder(self.dataset_path, transforms)
        self.image_folder.dataset_type = dataset_type
        self.image_folder.nb_classes = cfg['nb_classes']
        samples_ = []
        for filename in filenames:
            for i, sample in enumerate(self.image_folder.samples):
                if sample[0].split('\\')[-1] == filename:
                    samples_.append(sample)
                    # a=1
        # filenames_ = np.asarray([i for i, sample in enumerate(self.image_folder.samples) if sample[0].split('\\')[-1] in filenames])
        # samples_ = np.asarray(self.image_folder.samples)[filenames_]
        # self.image_folder.samples = samples_
        self.image_folder.labels = [sample[1] for i, sample in enumerate(samples_)]
        self.image_folder.paths = [sample[0] for i, sample in enumerate(samples_)]
        self.image_folder.samples = samples_
        a=1
        # assert len(self.image_folder) == cfg[f'nb_{dataset_type}_images'], \
        #     f'Incorrect number of images in {dataset_type} set.'
