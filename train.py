import os
import logging
import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Resize
from transformers import AutoTokenizer

from pipeline import build_loader
from caption_model import Transformer


class Config:
    # model name in 'baseline', 'MAT', 'PNAT', 'mask', 'oscar'
    model_name = 'custom'
    exp = 'custom'
    discription = ''
    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    train_data_path = 'stair_captions/stair_captions_train_data.txt'
    valid_data_path = 'stair_captions/stair_captions_valid_data.txt'

    log_path = f'exp/{exp}/train.txt'
    weight_dir = f'exp/{exp}/weight'

    image_size = 224
    patch_size = 16
    dim = 384
    heads = 8
    encoder_depth = 8
    decoder_depth = 4
    max_seq_len = 128
    drop_prob = 0.125

    epochs = 4
    batch_size = 64
    lr = 0.00005
    check_point_step = 4000
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def config_info():
    logger.info(f'model name: {Config.model_name}')
    logger.info(f'image size: {Config.image_size}')
    logger.info(f'patch size: {Config.patch_size}')
    logger.info(f'embed dim: {Config.dim}')
    logger.info(f'encoder depth: {Config.encoder_depth}')
    logger.info(f'decoder depth: {Config.decoder_depth}')
    logger.info(f'drop prob: {Config.drop_prob}')
    logger.info(f'LR: {Config.lr}')
    logger.info(f'discription: {Config.discription}\n')


class Trainer(Config):
    def __init__(self):
        os.makedirs(self.weight_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.train_loader = build_loader(
            data_path=[self.train_data_path, self.valid_data_path],
            image_size=self.image_size,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            train=True
        )

        self.model = Transformer(Config, self.tokenizer.vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = AvgManager()
        for step, (images, prov_ids, next_ids, mask) in enumerate(self.train_loader, 1):
            images, prov_ids, next_ids, mask = images.cuda(), prov_ids.cuda(), next_ids.cuda(), mask.cuda()

            self.optimizer.zero_grad()
            loss = self.model(images, prov_ids, next_ids, mask)
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item())
            print(f'\rstep: {step}   train loss: {loss.item():.4f}', end='')
        print(f'save model to: {Config.weight_dir}/epoch{epoch}_{Config.model_name}.pth')
        logger.info(f'save model to: {Config.weight_dir}/epoch{epoch}_{Config.model_name}.pth')
        torch.save(self.model.state_dict(), f'{Config.weight_dir}/epoch{epoch}_{Config.model_name}.pth')

    def _test_fn(self):
        self.model.eval()
        for index in range(4):
            image, _ = self.train_loader.dataset.__getitem__(0)
            image = image.unsqueeze(0).to(self.device)
            sequence = self.model.inference(image).squeeze(0)
            text = self.tokenizer.decode(sequence.tolist())
            print(f'caption: {text}')
            logger.info(f'caption: {text}')


class AvgManager:
    def __init__(self):
        self.total = 0.0
        self.n = 0

    def __call__(self):
        return self.total / self.n

    def update(self, value):
        self.total += value
        self.n += 1


def get_logger(logging_file, stream=True):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(filename=logging_file, encoding='utf-8'))
    if stream:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


def main():
    os.makedirs(Config.weight_dir, exist_ok=True)
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--info', action='store_true', default=False)
    parser.add_argument('-w', '--weight', type=str, default='weight/')
    parser.add_argument('-i', '--image_path', type=str, default=None)
    args = parser.parse_args()

    logger = get_logger(Config.log_path, stream=False)
    if args.info:
        config_info()
    main()
