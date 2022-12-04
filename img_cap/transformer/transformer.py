import os
import math
import json
import torch
from torch import nn
from torch.nn import functional
from .base_module import BaseModule
from .mlp import Mlp
from .embed import build_embedding


class AttentionOutputs:
    def __init__(self, state, k=None, v=None):
        self.state = state
        self.k = k
        self.v = v


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

def compute_bias(query_length, key_length, bidirectional, num_buckets, max_distance, device=None):
    context_position = torch.arange(query_length, dtype=torch.long, device=device).unsqueeze(1)
    memory_position = torch.arange(key_length, dtype=torch.long, device=device).unsqueeze(0)
    relative_position = memory_position - context_position
    return relative_position_bucket(
        relative_position, bidirectional=bidirectional, num_buckets=num_buckets, max_distance=max_distance,
    )


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, pos_bias=False, attn_drop=0.0, proj_drop=0.0):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_bias = pos_bias

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if pos_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        scores, key, value = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        scores = (scores @ key.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        if self.pos_bias:
            position_bias = compute_bias(
                query_length=N,
                key_length=key.shape[2],
                bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=128,
                device=scores.device
            )
            position_bias = self.relative_attention_bias(position_bias)  # shape (query_length, key_length, num_heads)
            position_bias = position_bias.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)

            print(position_bias.shape, scores.shape, k.shape)
            scores = scores + position_bias[:, :, -q.size(1):, :]

        attn_weight = functional.softmax(input=scores.float(), dim=-1, dtype=scores.dtype)
        attn_weight = self.attn_drop(attn_weight)

        attn_output = (attn_weight @ value).transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return AttentionOutputs(state=attn_output, k=None, v=None)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, pos_bias=False, attn_drop=0.0, proj_drop=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_bias = pos_bias

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if pos_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

    def forward(self, q, kv, cache=False, cache_kv=None):
        B, N, C = q.shape

        scores = self.q(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if cache and cache_kv is not None:
            key, value = cache_kv
        else:
            key, value = self.kv(kv).view(B, kv.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # kv shape: (batch, heads, sequence, embed_dim)

        scores = (scores @ key.transpose(-2, -1)) * self.scale

        if self.pos_bias:
            position_bias = compute_bias(
                query_length=N,
                key_length=key.shape[2],
                bidirectional=True,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=128,
                device=scores.device
            )
            position_bias = self.relative_attention_bias(position_bias)  # shape (query_length, key_length, num_heads)
            position_bias = position_bias.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)

            scores = scores + position_bias[:, :, -q.size(1):, :]

        attn_weight = functional.softmax(input=scores.float(), dim=-1, dtype=scores.dtype)
        attn_weight = self.attn_drop(attn_weight)

        attn_output = (attn_weight @ value).transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return AttentionOutputs(state=attn_output, k=key, v=value)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, pos_bias=False, attn_drop=0.0, drop=0.0, eps=1e-6):
        super(EncoderBlock, self).__init__()
        self.attn_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pos_bias=pos_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.ffn = Mlp(embed_dim=embed_dim, ext_rate=4)

    def forward(self, x):
        x = self.attn_norm(x + self.self_attention(x=x).state)
        x = self.ffn_norm(x + self.ffn(x=x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True, pos_bias=False, attn_drop=0.0, drop=0.0, eps=1e-6):
        super(DecoderBlock, self).__init__()
        self.self_attn_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pos_bias=pos_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim, eps=eps)
        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pos_bias=pos_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.ffn_nrom = nn.LayerNorm(embed_dim, eps=eps)
        self.ffn = Mlp(embed_dim=embed_dim, ext_rate=4)
    
    def forward(self, q, kv, mask=None, cache=False, cache_kv=None):
        self_attention_outputs = self.self_attention(x=q, mask=mask)
        q = self.self_attn_norm(q + self_attention_outputs.state)

        cross_attention_outputs = self.cross_attention(q=q, kv=kv, cache=cache, cache_kv=cache_kv)
        q = self.cross_attn_norm(q + cross_attention_outputs.state)

        q = self.ffn_nrom(q + self.ffn(x=q))
        return q, cross_attention_outputs


class EncoderOutputs:
    def __init__(self, x, hidden_states):
        self.last_states = x
        self.hidden_states = tuple(hidden_state for hidden_state in hidden_states)
        self.cache = None


class DecoderOutputs:
    def __init__(self, x, hidden_states):
        self.last_states = x
        self.hidden_states = tuple(hidden_state[0] for hidden_state in hidden_states)
        self.cache = tuple((hidden_state[1].k, hidden_state[1].v) for hidden_state in hidden_states)


class Encoder(nn.Module):
    def __init__(self, encoder_config: dict):
        super(Encoder, self).__init__()
        assert isinstance(encoder_config, dict)
        depth = encoder_config.pop('depth')
        self.encoder_layers = nn.ModuleList([EncoderBlock(**encoder_config) for _ in range(depth)])

    def forward(self, x, cache=False):
        hidden_states = list()
        for layer in self.encoder_layers:
            x = layer(x=x)
            hidden_states.append(x)
        return EncoderOutputs(x=x, hidden_states=hidden_states)


class Decoder(nn.Module):
    def __init__(self, decoder_config: dict):
        super(Decoder, self).__init__()
        assert isinstance(decoder_config, dict)
        depth = decoder_config.pop('depth')
        self.decoder_layers = nn.ModuleList([DecoderBlock(**decoder_config) for _ in range(depth)])

    def forward(self, x, memory, mask=None, cache=False, cache_kv=None):
        hidden_states = list()
        for index, layer in enumerate(self.decoder_layers):
            if cache and cache_kv is not None:
                one_cache_kv = cache_kv[index]
            else:
                one_cache_kv = None
            x, cross_attention_outputs = layer(q=x, kv=memory, mask=mask, cache=cache, cache_kv=one_cache_kv)
            hidden_states.append((x, cross_attention_outputs))
        
        return DecoderOutputs(x=x, hidden_states=hidden_states)


class TransformerOutputs:
    def __init__(self, loss, encoder_last_state=None, decoder_last_state=None):
        self.loss = loss
        self.encoder_last_state = encoder_last_state
        self.decoder_last_state = decoder_last_state


class Transformer(BaseModule):
    def __init__(
        self,
        vocab_size: int,
        init_config: dict,
        encoder_config: dict,
        decoder_config: dict,
        encoder_embed_config: dict,
        decoder_embed_config: dict,
        meshed=False
    ):
        super(Transformer, self).__init__(init_config=init_config)
        assert isinstance(encoder_config, dict)
        assert isinstance(decoder_config, dict)
        assert isinstance(encoder_embed_config, dict)
        assert isinstance(decoder_embed_config, dict)
        self.vocab_size = vocab_size
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.encoder_embed_config = encoder_embed_config
        self.decoder_embed_config = decoder_embed_config
        self.meshed = meshed
        if meshed:
            self.layer_weight = nn.Parameter(torch.rand(encoder_config['depth'], 1, 1, 1))

        self.encoder = Encoder(encoder_config=encoder_config)
        self.decoder = Decoder(decoder_config=decoder_config)

        decoder_embed_config['vocab_size'] = vocab_size
        self.encoder_embedding = build_embedding(encoder_embed_config)
        self.decoder_embedding = build_embedding(decoder_embed_config)

        self.head = nn.Linear(in_features=decoder_config['embed_dim'], out_features=vocab_size, bias=True)

        self.init()

    def forward(self, images, input_ids, mask=None):
        pass

    def forward_train(self, images, input_ids):
        input_ids, target_ids = input_ids[:, :-1], input_ids[:, 1:]

        attention_mask = get_mask(input_ids, padding=0)
        attention_mask = attention_mask.to(images.device)

        encoder_embed_state = self.encoder_embedding(x=images)
        encoder_outputs = self.encoder(x=encoder_embed_state)

        last_state = self._is_meshed(encoder_outputs=encoder_outputs)
        
        decoder_embed_state = self.decoder_embedding(x=input_ids)
        decoder_outputs = self.decoder(x=decoder_embed_state, memory=last_state, mask=attention_mask)

        logits = self.head(decoder_outputs.last_states).contiguous().view(-1, self.vocab_size)
        loss = functional.cross_entropy(input=logits, target=target_ids.contiguous().view(-1))
        return loss

    def forward_test(self, images, max_length=196, cls_token_id=2):
        tokens = torch.tensor([cls_token_id for _ in range(images.shape[0])], dtype=torch.long, device=images.device).unsqueeze(1)

        cache = None

        encoder_embed_state = self.encoder_embedding(x=images)
        encoder_outputs = self.encoder(x=encoder_embed_state)
        last_state = self._is_meshed(encoder_outputs=encoder_outputs)

        for _ in range(max_length):
            decoder_embed_state = self.decoder_embedding(x=tokens)
            decoder_outputs = self.decoder(x=decoder_embed_state, memory=last_state, mask=None, cache=True, cache_kv=cache)
            logits = self.head(decoder_outputs.last_states)

            generated_tokens = logits.argmax(dim=2).long()[:, -1:]
            tokens = torch.cat((tokens, generated_tokens), dim=1)

            cache = decoder_outputs.cache

            if generated_tokens.sum() == 0.0:
                break
        
        return tokens

    def _is_meshed(self, encoder_outputs):
        if self.meshed:
            hidden_states_list = torch.stack(encoder_outputs.hidden_states, dim=0)
            last_state = torch.sum(hidden_states_list * torch.sigmoid(self.layer_weight), dim=0)
        else:
            last_state = encoder_outputs.last_states
        return last_state
        
    def save_pretrained(self, path):
        torch.save(self.state_dict(), f=os.path.join(path, 'weight.pth'))

        save_config = dict(
            model=dict(
                vocab_size=self.vocab_size,
                encoder_config=self.encoder_config,
                decoder_config=self.decoder_config,
                encoder_embed_config=self.encoder_embed_config,
                decoder_embed_config=self.decoder_embed_config,
                meshed=self.meshed
            )
        )

        with open(os.path.join(path, 'config.py'), mode='w') as f:
            json.dump(save_config, f)

    @staticmethod
    def load_pretrained(path):
        with open(os.path.join(path, 'config.py'), mode='r') as f:
            config = json.load(f)
        
        assert isinstance(config, dict)
        config = config['model']
        model = Transformer(**config)
        model.load_state_dict(torch.load(os.path.join(path, 'weight.pth')))
        return model


def get_mask(sequence, padding=0):
    _, sequence_len = sequence.shape
    sequence_mask = torch.tril(torch.ones((sequence_len, sequence_len), dtype=torch.uint8, device=sequence.device)).unsqueeze(0)
    padding_mask = (sequence != padding).to(torch.uint8).unsqueeze(1).repeat(1, sequence_len, 1)
    return (sequence_mask & padding_mask).unsqueeze(1)


# [PAD] 0
# [UNK] 1
# [CLS] 2
# [SEP] 3
# [MASK] 4