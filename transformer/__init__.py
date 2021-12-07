from .transformer import Transformer
from .pnat import PositionNonAutoregressiveTransformer
from .mask_decoder import MaskTransformer


def build_model(config, vocab_size):
	if config.model_name == 'baseline':
		return Transformer(config, vocab_size)
	elif config.model_name == 'PNAT':
		return PositionNonAutoregressiveTransformer(config, vocab_size, config.max_seq_len)
	elif config.model_name == 'mask transform':
		return MaskTransformer(config, vocab_size)
	else:
		raise ValueError(f'{config.model_name} is invalid model name')
