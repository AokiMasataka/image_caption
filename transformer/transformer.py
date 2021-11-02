import torch
from torch import nn
from torch.nn import functional
from .transformer_base import EncoderLayer, Encoder, Decoder, PositionalEmbedding, VisionEmbedding


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, drop_prob, length_pred=False):
        super(VisionTransformer, self).__init__()
        seq_len = (image_size // patch_size) ** 2
        seq_len = seq_len + 1 if length_pred else seq_len
        self.length_pred = length_pred

        self.image_embed = VisionEmbedding(dim, patch_size)
        self.pos_embed = PositionalEmbedding(dim, seq_len)
        if self.length_pred:
            self.query = nn.Parameter(torch.rand(1, 1, dim))

        self.encoder_layer = nn.Sequential(
            *[EncoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)]
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.image_embed(x)
        if self.length_pred:
            x = torch.cat((self.query.repeat(batch, 1, 1), x), dim=1)
        x = self.pos_embed(x)
        x = self.encoder_layer(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cnf, vocab_size):
        super(Transformer, self).__init__()
        self.encoder = VisionTransformer(
            image_size=cnf.image_size,
            patch_size=cnf.patch_size,
            dim=cnf.dim,
            depth=cnf.encoder_depth,
            heads=cnf.heads,
            drop_prob=cnf.drop_prob
        )

        self.decoder = Decoder(
            dim=cnf.dim, heads=cnf.heads, depth=cnf.decoder_depth, vocab_size=vocab_size, max_len=cnf.max_seq_len
        )
        self.max_seq_len = cnf.max_seq_len
        self.vocab_size = vocab_size

    def forward(self, image, target, labels, mask=None):
        memory = self.encoder(image)
        logits = self.decoder(target, memory, mask)

        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        return functional.cross_entropy(logits, labels, ignore_index=0)

    @torch.inference_mode()
    def inference(self, image):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        memory = self.encoder(image)
        sequence = torch.full((1, 1), fill_value=2, dtype=torch.long, device=device)

        for i in range(self.max_seq_len):
            logit = self.decoder(sequence, memory, mask=None)
            logit = logit[:, -1, :].argmax(dim=1).view(-1, 1)
            sequence = torch.cat((sequence, logit), dim=1)
        return sequence


# [PAD] 0
# [UNK] 1
# [CLS] 2
# [SEP] 3
# [MASK] 4
