import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Compose, Resize, ColorJitter, RandomPerspective, InterpolationMode


def build_loader(data, data_dir, image_size, batch_size, tokenizer, train=False):
    transform = get_transform(image_size, train)
    dataset = StairDataset(data, data_dir, tokenizer, transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn
    )


class StairDataset(Dataset):
    def __init__(self, data, data_dir, tokenizer, transform):
        self.data = data
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.length = data.__len__()
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        loc = self.data[index]
        text_ids = self.tokenizer.encode(loc['caption'])
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        text_len = text_ids.shape[0]
        
        mask_ids = self._mask(deepcopy(text_ids), text_len)

        file_name = loc['file_name']
        image_path = str(self.data_dir / file_name)
        image = read_image(image_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        image = image.float() / 255.0

        return image, text_ids, mask_ids, text_len

    def _mask(self, input_ids, text_length):
        mask_pos = np.random.randint(1,  text_length)
        input_ids[mask_pos] = 4
        return input_ids


def collate_fn(batch):
    inputs = {}
    images, text_ids, mask_ids, text_len = list(zip(*batch))
    
    src_ids = pad_sequence(text_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    prov_ids = src_ids[:, :-1]
    next_ids = src_ids[:, 1:]

    inputs['image'] = torch.stack(images)
    inputs['length'] = torch.tensor(text_len, dtype=torch.long)
    inputs['src_ids'] = src_ids
    inputs['masked_src'] = mask_ids
    inputs['src_pad_mask'] = get_pad_mask(src_ids)
    inputs['prov_ids'] = prov_ids
    inputs['next_ids'] = next_ids
    inputs['prov_mask'] = get_pad_mask(prov_ids) & get_seq_mask(next_ids)
    return inputs


def get_transform(image_size, train=True):
    if train:
        transform = Compose([
            ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
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
    _, seq_len = seq.shape
    seq = (seq != padding).to(torch.uint8)
    seq = seq.unsqueeze(1).repeat(1, seq_len, 1)
    return seq
