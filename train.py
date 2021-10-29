import os
import pickle
import torch
from torch.nn import functional
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Resize
from transformers import AutoTokenizer

from pipline import get_loader
from transformer import Transformer, PositionNonAutoregressiveTransformer


class Config:
    train_image_dir = 'D:/data_set/COCO_images/train2014/'
    train_pickle_path = 'stair_captions/stair_captions_train.pickle'

    valid_image_dir = 'D:/data_set/COCO_images/val2014/'
    valid_pickle_path = 'stair_captions/stair_captions_valid.pickle'
    weight_dir = 'weight_pnat'

    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    image_size = 224
    patch_size = 16
    dim = 384
    heads = 8
    encoder_depth = 8
    decoder_depth = 8
    max_seq_len = 128
    drop_prob = 0.125

    epochs = 4
    batch_size = 8
    lr = 0.00005
    check_point_step = 4000


def test_fn(model, tokenizer, image_path=None):
    model.eval()
    resize = Resize((Config.image_size, Config.image_size))
    if image_path is None:
        image_path = [
            'D:/data_set/COCO_images/val2014/COCO_val2014_000000000042.jpg',
            'D:/data_set/COCO_images/val2014/COCO_val2014_000000000073.jpg',
            'D:/data_set/COCO_images/train2014/COCO_train2014_000000000009.jpg',
            'D:/data_set/COCO_images/train2014/COCO_train2014_000000000025.jpg',
        ]

    for path in image_path:
        image = read_image(path, mode=ImageReadMode.RGB)
        image = resize(image).float() / 255.0
        image = image.unsqueeze(0).cuda()
        seq = model.forward_ones(image, Config.max_seq_len)
        seq = seq[0].tolist()
        text = tokenizer.decode(seq)
        print(f'caption: {text}')
    model.train()


def main():
    os.makedirs(Config.weight_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    with open(Config.train_pickle_path, mode='rb') as f:
        train_list = pickle.load(f)

    with open(Config.valid_pickle_path, mode='rb') as f:
        valid_list = pickle.load(f)

    train_list = train_list + valid_list
    del valid_list

    train_loader = get_loader(
        train_list, Config.train_image_dir, Config.image_size, Config.batch_size, tokenizer, train=False
    )

    vocab_size = tokenizer.get_vocab().__len__()
    # model = Transformer(Config, vocab_size).cuda()
    model = PositionNonAutoregressiveTransformer(Config, vocab_size).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

    train_loss = Avg()
    for epoch in range(Config.epochs):
        model.train()
        train_loss.__init__()
        for step, (images, target, labels, mask) in enumerate(train_loader, 1):

            if step % Config.check_point_step == 0:
                print(f'\rstep: {step}   train loss: {train_loss()}')
                test_fn(model, tokenizer)
            images, target, labels, mask = images.cuda(), target.cuda(), labels.cuda(), mask.cuda()

            optimizer.zero_grad()
            logit = model(images, target, mask)
            loss = functional.cross_entropy(logit.view(-1, vocab_size), labels.view(-1), ignore_index=0)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())
            print(f'\rstep: {step}   train loss: {loss.item()}', end='')
        print(f'save model to: {Config.weight_dir}/epoch{epoch}_baseline.pth')
        torch.save(model.state_dict(), f'{Config.weight_dir}/epoch{epoch}_baseline.pth')


class Avg:
    def __init__(self):
        self.total = 0.0
        self.n = 0

    def __call__(self):
        return self.total / self.n

    def update(self, value):
        self.total += value
        self.n += 1


if __name__ == '__main__':
    main()
