from torch import nn
import torch
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
        self.max_seq_len = config.max_seq_len

    def forward(self, inputs: dict):
        image = inputs['image']
        src_length = inputs['length']
        src_ids = inputs['src_ids']
        masked_src = inputs['masked_src']
        src_pad_mask = inputs['src_pad_mask']

        encoder_out = self.encoder(image)

        length_pred = self.length_pred(encoder_out[:, 0, :])
        logit = self.decoder(src_ids, encoder_out[:, 1:, :], src_pad_mask)
        
        logit = logit.view(-1, self.vocab_size)
        logit_loss = functional.cross_entropy(logit, masked_src.view(-1), ignore_index=0)
        length_loss = functional.cross_entropy(length_pred, src_length)
        return logit_loss + length_loss
    
    @torch.inference_mode()
    def inference(self, image):
        x = self.encoder(image)
        device = x.device
        memory = x[:, 1:, :]
        length_pred = self.length_pred(x[:, 0, :]).argmax(dim=1)
        length_pred = length_pred.cpu().numpy()[0]

        # mask id: 4 pad id: 0
        query = torch.zeros(length_pred, dtype=torch.long, device=device) + 4
        query = query.unsqueeze(0)
        
        n_i = 3
        for i in range(n_i):
            query = self.decoder(query, memory, mask=None)
            index = query.argmax(dim=2)
            if i == n_i - 1:
                break

            value = query.max(dim=2).values
            index[value < 0.001] = 4
            query = index.long()
        return index

