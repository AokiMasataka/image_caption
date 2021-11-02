from torch import nn

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

        self.lenght_pred = nn.Linear(config.dim, config.max_seq_len)

    def forward(self, x, target, mask=None):
        x = self.encoder(x)
        lenght = x[:, 1, :]
        memory = x[:, 1:, :]

        logit = self.decoder(target, memory, mask)
        return logit, lenght
