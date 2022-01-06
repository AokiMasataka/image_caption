from .transformer import Transformer
from .nat import NAT


def build_model(config, vocab_size):
    if config.model_name == 'baseline':
        return Transformer(config, vocab_size)
    elif config.model_name == 'nat':
        return NAT(config, vocab_size)
    else:
        raise ValueError(f'{config.model_name} is invalid model name')
