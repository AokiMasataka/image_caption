import sys

import torch
from transformers import AutoTokenizer
from train import Config
from caption_model import Transformer

from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import Resize


@torch.inference_mode()
def main():
    weight_file = sys.argv[1]
    image_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
    model = Transformer(Config, tokenizer.vocab_size).eval()
    model.load_state_dict(torch.load(weight_file))

    image = read_image(path=image_path, mode=ImageReadMode.RGB).unsqueeze(0)
    image = Resize(size=(Config.image_size, Config.image_size))(image)
    image = image.float() / 255.0
    sequence = model.inference(image).squeeze(0)
    text = tokenizer.decode(sequence.tolist())
    print(f'caption: {text}')


if __name__ == '__main__':
    main()
