# https://arxiv.org/pdf/1911.10677.pdf
import torch
from torch import nn

from .transformer_base import Decoder, MultiHeadAttention
from .transformer import VisionTransformer, Decoder


class PNAT(nn.Module):
    def __init__(self, config, vocab_size):
        super(PNAT, self).__init__()
        self.encoder = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.encoder_depth,
            heads=config.heads,
            drop_prob=0.1,
            length_pred=False
        )

        self.decoder = Decoder(
            dim=config.dim,
            heads=config.heads,
            depth=config.decoder_depth,
            vocab_size=vocab_size,
            max_len=config.max_seq_len
        )
    
    def forward(self):
        return 0
    
    @torch.inference_mode()
    def inference(self, x):
        return 0


class Bridge(nn.Module):
    def __init__(self):
        super(Bridge, self).__init__()
        pass
    
    def forward(self, x):
        return 0


class PositionPredicter(nn.Module):
    def __init__(self):
        super(PositionPredicter, self).__init__()
