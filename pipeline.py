import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Compose, Resize, ColorJitter, RandomPerspective


def build_loader(data_path, image_size, batch_size, tokenizer, train=True):
    num_worker = min(batch_size, 8)
    transform = get_transform(image_size=image_size, train=train)
    dataset = CaptionData(data_path, tokenizer, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_worker, pin_memory=True, collate_fn=collate_fn
    )
    return loader


class CaptionData(Dataset):
    def __init__(self, data_path, tokenizer, transform):
        self.data_list = self._data_cache(data_path)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return self.data_list.__len__()

    def __getitem__(self, index):
        image_path, caption = self.data_list[index].split('||')
        text_ids = self.tokenizer.encode(caption)
        text_ids = torch.tensor(text_ids, dtype=torch.long)

        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        image = image.float() / 255.0
        return image, text_ids

    @staticmethod
    def _data_cache(data_path):
        if type(data_path) != list:
            data_path = [data_path]

        data_list = []
        for one_data_path in data_path:
            with open(one_data_path, 'r', encoding='UTF-8') as f:
                data = f.read()

            data_list += data.split('\n')
        return data_list


def collate_fn(batch):
    images, text_ids = list(zip(*batch))

    images = torch.stack(images)
    src_ids = pad_sequence(text_ids, batch_first=True)
    prov_ids = src_ids[:, :-1]
    next_ids = src_ids[:, 1:]
    mask = get_mask(prov_ids)
    return images, prov_ids, next_ids, mask


def get_transform(image_size, train=True):
    if train:
        transform = Compose([
            Resize((image_size, image_size)),
            ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
            RandomPerspective(distortion_scale=0.2, p=0.5)
        ])
    else:
        transform = Compose([
            Resize((image_size, image_size))
        ])
    return transform


def get_mask(sequence, padding=0):
    batch_size, sequence_len = sequence.shape
    sequence_mask = torch.tensor(np.tri(sequence_len, dtype=np.uint8)).unsqueeze(0).repeat(batch_size, 1, 1)
    padding_mask = (sequence != padding).to(torch.uint8).unsqueeze(1).repeat(1, sequence_len, 1)
    return sequence_mask & padding_mask


def get_seq_mask(seq):
    batch_size, seq_len = seq.shape
    return torch.tensor(np.tri(seq_len, dtype=np.uint8)).unsqueeze(0).repeat(batch_size, 1, 1)


def get_pad_mask(seq, padding=0):
    _, seq_len = seq.shape
    seq = (seq != padding).to(torch.uint8)
    seq = seq.unsqueeze(1).repeat(1, seq_len, 1)
    return seq
