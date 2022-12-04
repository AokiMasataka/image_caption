import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import io
from torchvision.transforms import Compose, Resize, ColorJitter, RandomPerspective


class CaptionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, transform=None):
        assert data_path.split('.')[-1] == 'txt'
        self.data_list = self._data_cache(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
    
    def __len__(self):
        return self.data_list.__len__()

    def __getitem__(self, index):
        image_path, caption = self.data_list[index].split('||')
        inputs = self.tokenizer(
            caption,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )

        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)

        image = io.read_image(image_path, mode=io.image.ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        image = image.float() / 255.0
        return image, input_ids
    
    @staticmethod
    def _data_cache(data_path):
        if isinstance(data_path, (list, tuple)):
            data = []
            for one_data_path in data_path:
                data += CaptionDataset._data_cache(one_data_path)
        else:
            with open(data_path, 'r', encoding='UTF-8') as f:
                data = f.read()
            return data.split('\n')
    
    @staticmethod
    def collate_fn(batch):
        images, input_ids = list(zip(*batch))
        images = torch.stack(images)
        input_ids = pad_sequence(input_ids, batch_first=True)
        return images, input_ids


def build_transforms(image_size):
    train_transform = Compose([
        Resize((image_size, image_size)),
        ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
        RandomPerspective(distortion_scale=0.2, p=0.5)
    ])

    valid_transform = Compose([Resize(size=(image_size, image_size))])
    return train_transform, valid_transform


def build_loader(data_path, image_size, batch_size, max_length, tokenizer, shuffle=True):
    num_worker = min(batch_size, 10)
    transform, _ = build_transforms(image_size=image_size)
    dataset = CaptionDataset(data_path, tokenizer, max_length, transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker, collate_fn=CaptionDataset.collate_fn
    )
    return data_loader
