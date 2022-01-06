import torch
from torch import nn
from torch.nn import functional
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from .transformer_base import MultiHeadAttention, FeedForward, EncoderLayer, VisionEmbedding, PositionalEmbedding


class NAT(nn.Module):
    def __init__(self, config, vocab_size):
        super(NAT, self).__init__()
        self.encoder = NATVisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.encoder_depth,
            heads=config.heads,
        )
        
        self.decoder = NATDecoder(
            dim=config.dim,
            depth=config.decoder_depth,
            heads=config.heads,
            max_seq_len=config.max_seq_len
        )
        self.head = nn.Linear(config.dim, vocab_size, bias=False)
        self.max_seq_len = config.max_seq_len
        self.vocab_size = vocab_size

    def forward(self, inputs: dict):
        image = inputs['image']
        src_ids = inputs['src_ids']
        src_mask = inputs['src_pad_mask']
        batch_size = image.shape[0]
        _src_mask = torch.zeros((batch_size, self.max_seq_len, self.max_seq_len), dtype=torch.bool, device='cuda')
        _src_mask[:, :src_mask.shape[1], :src_mask.shape[2]] = src_mask
        src_mask = _src_mask

        memory, rep = self.encoder(image)
        hidden_state = self.decoder(rep, memory, mask=src_mask)
        predict = self.head(hidden_state)

        _src_ids = torch.zeros((batch_size, self.max_seq_len), dtype=torch.long, device='cuda')
        _src_ids[:, :src_ids.shape[1]] = src_ids
        src_ids = _src_ids
        
        predict = predict.view(-1, self.vocab_size)
        src_ids = src_ids.view(-1)
        return functional.cross_entropy(predict, src_ids, ignore_index=0)

    @torch.inference_mode()
    def inference(self, image):
        memory, rep = self.encoder(image)
        hidden_state = self.decoder(rep, memory, mask=None)
        hidden_state = hidden_state[:, :self.max_seq_len, :]
        predict = self.head(hidden_state)
        return predict.argmax(dim=1)


class NATVisionEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, max_repeat=16, drop_prob=0.1):
        super(NATVisionEncoder, self).__init__()
        sequence_length = (image_size // patch_size) ** 2
        self.sequence_length = sequence_length

        self.patch_embed = VisionEmbedding(dim, patch_size)
        self.pos_embed = PositionalEmbedding(dim, sequence_length)
        self.encoder_layer = nn.Sequential(
            *[EncoderLayer(dim, heads, drop_prob=drop_prob, attention_drop_prob=drop_prob) for _ in range(depth)]
        )
        
        self.repeater = nn.Sequential(
            nn.Linear(dim, max_repeat, bias=True),
            nn.Softmax(dim=2)
        )
        self.max_sequence = 128

    def forward(self, x):
        patch_embed = self.patch_embed(x)   #patch_embed: (batch_size, sequence, dim)
        pos_embed = self.pos_embed(patch_embed)
        out_state = self.encoder_layer(pos_embed)
        
        rep = self.repeater(out_state)
        query = self._f(patch_embed, rep)
        return out_state, query

    def _f(self, patch, rep):
        # patch: (batch_size, sequence, dim)
        # rep: (batch_size, sequence, num_repeat)
        batch_size, sequence, dim = patch.shape

        rep = rep.argmax(dim=2) 
        querys = []
        for i in range(batch_size):
            query = []
            for j in range(sequence):
                query.append(patch[i, j].unsqueeze(0).repeat(rep[i, j], 1))
            query = torch.cat(query, dim=0)
            querys.append(query[:self.max_sequence])

        querys, _ = pad_packed_sequence(pack_sequence(querys), batch_first=True, total_length=self.max_sequence)

        return querys


class NATDecoder(nn.Module):
    def __init__(self, dim, heads, depth, max_seq_len, drop_prob=0.1):
        super(NATDecoder, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros((max_seq_len, dim)))

        self.decoder_blocks = nn.ModuleList([
            NATDecoderLayer(dim=dim, heads=heads, max_seq_len=max_seq_len, drop_prob=drop_prob) for _ in range(depth)
        ])

    def forward(self, src, memory, mask=None):
        hidden_state = self.pos_embed[:src.shape[1], :] + src
        
        for decoder_block in self.decoder_blocks:
            hidden_state = decoder_block(hidden_state, memory, mask=mask)
        return hidden_state


class NATDecoderLayer(nn.Module):
    def __init__(self, dim, heads, max_seq_len, drop_prob=0.1):
        super(NATDecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(dim, heads, drop_prob=drop_prob)
        self.pos_embed = MultiHeadAttention(dim, heads, drop_prob=0.0)
        self.mask_attn = MultiHeadAttention(dim, heads, drop_prob=drop_prob)
        self.feed_forward = FeedForward(dim, dim * 4, drop_prob)
        
        self.pos_embed_params = nn.Embedding(max_seq_len, dim)

    def forward(self, x, memory, mask=None):
        batch_size, seq_len, _ = x.shape
        pos_range = torch.arange(seq_len, device='cuda').view(1, seq_len).repeat(batch_size, 1)
        pos_state = self.pos_embed_params(pos_range)
        
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.pos_embed(q=x, k=pos_state, v=pos_state, mask=mask)
        x = self.mask_attn(q=x, k=memory, v=memory, mask=None)
        x = self.feed_forward(x)
        return x


class Filer(nn.Module):
    def __init__(self, max_repeat=16):
        super(self).__init__()
        self.max_repeat = max_repeat
        self.head = nn.Linear(dim, max_repeat, bias=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoder_out):
        # encoder_out: batch, sequence, dim
        repeat = self.softmax(self.head(encoder_out))   # batch sequence n
        repeat = torch.max(repeat, dim=2)
        

        return x


if __name__ == '__main__':
    model = NATDecoder(dim=256, heads=4, depth=4, vocab_size=1024, max_seq_len=64)
    x = torch.rand(4, 32, 256)
    memory = torch.rand(4, 100, 256)

    y = model(x, memory)

    print(y.shape)
    print('Done')

