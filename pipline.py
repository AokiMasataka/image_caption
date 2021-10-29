import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Compose, Resize, ColorJitter, RandomPerspective


def get_loader(data, image_dir, image_size, batch_size, tokenizer, train: bool):
    transform = get_transform(image_size, train)
    dataset = StairDataset(data, image_dir, tokenizer, transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn
    )


class StairDataset(Dataset):
    def __init__(self, data, image_dir, tokenizer, transform):
        self.data = data
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.length = data.__len__()
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        loc = self.data[index]
        text_ids = self.tokenizer.encode(loc['caption'])
        text_ids = torch.tensor(text_ids, dtype=torch.long)

        file_name = loc['file_name']
        if 'train' in file_name:
            image_path = 'D:/data_set/COCO_images/train2014/' + file_name
        else:
            image_path = 'D:/data_set/COCO_images/val2014/' + file_name
        # image_path = self.image_dir + loc['file_name']
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        image = image.float() / 255.0

        return image, text_ids


def collate_fn(batch):
    images, text_ids = list(zip(*batch))
    text_ids = pad_sequence(text_ids, batch_first=True)
    target = text_ids[:, :-1]
    label = text_ids[:, 1:]
    seq_mask = get_seq_mask(target)
    pad_mask = get_pad_mask(target)
    mask = seq_mask & pad_mask

    images = torch.stack(images)
    return images, target, label, mask


def get_transform(image_size, train=True):
    if train:
        transform = Compose([
            ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            Resize((image_size, image_size))
        ])
    else:
        transform = Compose([
            Resize((image_size, image_size))
        ])
    return transform


def get_seq_mask(seq):
    batch_size, seq_len = seq.shape
    return torch.tensor(np.tri(seq_len, dtype=np.uint8)).unsqueeze(0).repeat(batch_size, 1, 1)


def get_pad_mask(seq, padding=0):
    batch_size, seq_len = seq.shape
    seq = (seq != padding).to(torch.uint8)
    seq = seq.unsqueeze(1).repeat(1, seq_len, 1)
    return seq
