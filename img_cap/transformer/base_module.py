import logging
import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, init_config: dict):
        super(BaseModule, self).__init__()
        self.init_config = init_config

    def init(self):
        if self.init_config['pretrained'] is not None:
            state_dict = torch.load(self.init_config['pretrained'], map_location='cpu')
            miss_match = self.load_state_dict(state_dict=state_dict, strict=False)
            logging.info(miss_match)
        
        if self.init_config['encoder_weight'] is not None:
            state_dict = torch.load(self.init_config['encoder_weight'], map_location='cpu')
            miss_match = self.encoder.load_state_dict(state_dict=state_dict, strict=False)
            logging.info(msg=f'encoder miss match keys: {miss_match}')
            
        if self.init_config['decoder_weight'] is not None:
            state_dict = torch.load(self.init_config['decoder_weight'], map_location='cpu')
            miss_match = self.decoder.load_state_dict(state_dict=state_dict, strict=False)
            logging.info(msg=f'decoder miss match keys: {miss_match}')
        
        if self.init_config['encoder_embed_weight'] is not None:
            state_dict = torch.load(self.init_config['encoder_embed_weight'], map_location='cpu')
            miss_match = self.encoder_embedding.load_state_dict(state_dict=state_dict, strict=False)
            logging.info(msg=f'encoder_embedding miss match keys: {miss_match}')
        
        if self.init_config['decoder_embed_weight'] is not None:
            state_dict = torch.load(self.init_config['decoder_embed_weight'], map_location='cpu')
            miss_match = self.decoder_embedding.load_state_dict(state_dict=state_dict, strict=False)
            logging.info(msg=f'decoder_embedding miss match keys: {miss_match}')

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_encoder_embedding(self):
        return self.encoder_embedding
    
    def get_decoder_embedding(self):
        return self.decoder_embedding
    
    def get_head(self):
        return self.head