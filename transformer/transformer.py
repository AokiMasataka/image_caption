import torch
from torch import nn
from torch.nn import functional
from .transformer_base import Decoder


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

    def forward(self, inputs: dict):
        image = inputs['image']
        prov_ids = inputs['prov_ids']
        next_ids = inputs['next_ids']
        mask = inputs['prov_mask']

        memory = self.encoder(image)
        logits = self.decoder(prov_ids, memory, mask)

        logits = logits.view(-1, self.vocab_size)
        next_ids = next_ids.view(-1)
        return functional.cross_entropy(logits, next_ids, ignore_index=0)

    @torch.inference_mode()
    def inference(self, image):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        memory = self.encoder(image)
        sequence = torch.full((1, 1), fill_value=2, dtype=torch.long, device=device)

        for i in range(self.max_seq_len):
            logit = self.decoder(sequence, memory, mask=None)
            logit = logit[:, -1, :].argmax(dim=1)
            sequence = torch.cat((sequence, logit.unsqueeze(0)), dim=1)
            if logit.item() == 3:
                break
        return sequence


# [PAD] 0
# [UNK] 1
# [CLS] 2
# [SEP] 3
# [MASK] 4
