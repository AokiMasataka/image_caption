import time
import torch
from torchvision import io
from torchvision.transforms import Resize
from transformers import BertTokenizer, logging
from ..transformer import Transformer


def inference(config, weight_path, image_path_list):
    logging.set_verbosity_error()
    assert isinstance(image_path_list, (tuple, list))
    model_config = config['model']

    tokenizer = BertTokenizer.from_pretrained(model_config.pop('tokenizer'))
    model = Transformer(**model_config).eval()
    model.load_state_dict(torch.load(weight_path))

    images = list()
    size = (config['data']['image_size'], config['data']['image_size'])
    for image_path in image_path_list:
        image = io.read_image(image_path, mode=io.image.ImageReadMode.RGB)
        image = Resize(size=size)(image)
        image = image.float() / 255.0
        images.append(image)
    
    images = torch.stack(images, dim=0)
    start = time.time()
    tokens = model.forward_test(images=images, max_length=128)
    print(f'inference time: {time.time() - start:.4f}')

    texts = tokenizer.batch_decode(tokens)
    for image_path, text in zip(image_path_list, texts):
        print(f'image path: {image_path} - text: {text}')
