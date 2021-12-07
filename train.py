import os
import logging
import pickle
import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Resize
from transformers import AutoTokenizer

from pipeline import build_loader
from transformer import build_model


class Config:
    # model name in 'baseline', 'PNAT', 'mask transform'
    model_name = 'baseline'
    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    
    home = Path(os.path.expanduser('~'))
    train_image_dir = home / 'dataset/coco_images/train2014/'
    train_pickle_path = 'stair_captions/stair_captions_train.pickle'
    valid_image_dir = home / 'dataset/COCO_images/val2014/'
    valid_pickle_path = 'stair_captions/stair_captions_valid.pickle'

    log_path = f'exp/{model_name}/train.txt'
    weight_dir = f'exp/{model_name}/weight'

    image_size = 224
    patch_size = 16
    dim = 384
    heads = 8
    encoder_depth = 8
    decoder_depth = 8
    max_seq_len = 128
    drop_prob = 0.125

    epochs = 4
    batch_size = 16
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
    logger.info(f'LR: {Config.lr}\n')


class Trainer(Config):
    def __init__(self, model, train_loader, tokenizer):
        os.makedirs(Config.weight_dir, exist_ok=True)

        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def train_epoch(self, epoch):
        train_loss = AvgManager()
        for step, (images, target, labels, text_len, mask) in enumerate(self.train_loader, 1):
            if step % Config.check_point_step == 0:
                print(f'\rstep: {step}   train loss: {train_loss():.4f}')
                logger.info(f'step: {step}   train loss: {train_loss():.4f}')
                self._test_fn()
            
            inputs = {
                'image': images.to(self.device),
                'target': target.to(self.device),
                'label': labels.to(self.device),
                'text_len': text_len.to(self.device),
                'mask': mask.to(self.device)
            }

            self.optimizer.zero_grad()
            loss = self.model(inputs)
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item())
            print(f'\rstep: {step}   train loss: {loss.item():.4f}', end='')
        print(f'save model to: {Config.weight_dir}/epoch{epoch}_baseline.pth')
        logger.info(f'save model to: {Config.weight_dir}/epoch{epoch}_baseline.pth')
        torch.save(self.model.state_dict(), f'{Config.weight_dir}/epoch{epoch}_baseline.pth')

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)

    def _test_fn(self):
        self.model.eval()
        image_path = [
            Config.home / 'dataset/coco_images/val2014/COCO_val2014_000000000042.jpg',
            Config.home / 'dataset/coco_images/val2014/COCO_val2014_000000000073.jpg',
            Config.home / 'dataset/coco_images/train2014/COCO_train2014_000000000009.jpg',
            Config.home / 'dataset/coco_images/train2014/COCO_train2014_000000000025.jpg',
        ]

        for path in image_path:
            text = _inference(self.model, self.tokenizer, path)
            print(f'caption: {text}')
            logger.info(f'caption: {text}')
        self.model.eval()


def _inference(model, tokenizer, image_path):
    image = read_image(path=image_path, mode=ImageReadMode.RGB).unsqueeze(0)
    image = Resize(size=(Config.image_size, Config.image_size))(image)
    image = image.float() / 255.0
    image = image.to(Config.device)
    sequence = model.inference(image).squeeze(0)
    text = tokenizer.decode(sequence.tolist())
    return text


def inference():
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    vocab_size = tokenizer.get_vocab().__len__()
    model = build_model(Config, vocab_size).to(Config.device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    return _inference(model, tokenizer, args.image_path)


def main():
    os.makedirs(Config.weight_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    with open(Config.train_pickle_path, mode='rb') as f:
        train_list = pickle.load(f)

    with open(Config.valid_pickle_path, mode='rb') as f:
        valid_list = pickle.load(f)

    train_list = train_list + valid_list
    del valid_list

    train_loader = build_loader(train_list, Config.train_image_dir, Config.image_size, Config.batch_size, tokenizer)
    vocab_size = tokenizer.get_vocab().__len__()
    model = build_model(Config, vocab_size)
    trainer = Trainer(model=model, train_loader=train_loader, tokenizer=tokenizer)
    trainer.train()


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-w', '--weight', type=str, default='weight/')
    parser.add_argument('-i', '--image_path', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        os.makedirs(Config.weight_dir, exist_ok=True)
        logger = get_logger(Config.log_path, stream=False)
        config_info()
        main()
    elif args.mode == 'inference':
        print('inference mode')
        inference_text = inference()
        print(f'image path: {args.image_path}')
        print(f'caption: {inference_text}')
