from torch import nn
import torch
from torch._C import dtype
from torch.nn import functional

from .transformer import Decoder, VisionTransformer


class MaskTransformer(nn.Module):
    def __init__(self, config, vocab_size):
        super(MaskTransformer, self).__init__()
        self.encoder = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.encoder_depth,
            heads=config.heads,
            drop_prob=config.drop_prob,
            length_pred=True,
        )

        self.decoder = Decoder(
            dim=config.dim,
            heads=config.heads,
            depth=config.decoder_depth,
            vocab_size=vocab_size, max_len=config.max_seq_len
        )

        self.length_pred = nn.Linear(config.dim, config.max_seq_len)
        self.vocab_size = vocab_size

    def forward(self, inputs: dict):
        image = inputs['image']
        label = inputs['label']
        text_len = inputs['text_len']
        mask = inputs['mask']

        x = self.encoder(image)
        memory = x[:, 1:, :]
        length = x[:, 1, :]
        length = self.length_pred(length)

        logit = self.decoder(label, memory, mask)
        
        logit_loss = functional.cross_entropy(logit.view(-1, self.vocab_size), label.view(-1), ignore=0)
        length_loss = functional.cross_entropy(length, text_len)
        return logit_loss + length_loss
    
    def inference(self, image):
        x = self.encoder(image)
        memory = x[:, 1:, :]
        text_length = self.length_pred(x[:, 1, :])

        query = torch.zeros(text_length.shape[1], dtype=torch.long).unsqueeze(0)

        for _ in range(3):
            query = self.decoder(query, memory)
        return query

