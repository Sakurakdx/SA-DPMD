from transformers import BertModel
from .audio_encoder import AudioTransformerEncoder
import torch.nn as nn
from .attention_fusion_layer import ShortCutAttentionFusionLayer, GateAttentionFusionLayer
from .umt import CoupledCMT
from .base_encoder import BaseEncoder
from .text_encoder import TextEncoder
from .layer import NonLinear
import math
from .scale_mix import *


class MultiModalEncoder(BaseEncoder):
    def __init__(self, config, text_encoder):
        super(MultiModalEncoder, self).__init__(config)
        self.config = config
        self.text_encoder = text_encoder
        self.drop_emb = nn.Dropout(config.dropout_emb)
        self.start_layer = text_encoder.start_layer
        self.bert_layers = text_encoder.bert_layers
        self.layer_num = text_encoder.layer_num
        config.bert_hidden_size = text_encoder.bert.config.hidden_size

        self.audio_encoder = AudioTransformerEncoder(self.config.num_mel_bins, d_model=self.config.audio_hidden_dim, n_blocks=self.config.n_blocks)
        self.dropout = nn.Dropout(config.dropout_emb)

        self.fusion_layer = CoupledCMT(self.config)
        output_dim = text_encoder.config.hidden_size * 3
        
        # self.mlp_words = nn.ModuleList([NonLinear(config.bert_hidden_size, config.bert_hidden_size, activation=nn.Tanh()) \
        #                                 for i in range(self.layer_num)])
        # for i in range(self.layer_num):
        #     nn.init.kaiming_uniform_(self.mlp_words[i].linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')

        # self.rescale = ScalarMix(mixture_size=self.layer_num)

        if config.fix_embeddings:
            self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False

    def forward(self, input):
        batch_size, edu_num, edu_length = input["input_ids"].shape
        
        # 获得text的表示
        textual_repr = self.text_encoder(
            input["input_ids"].view(-1, edu_length), 
            token_type_ids=input["token_type_ids"].view(-1, edu_length),
            attention_mask=input["input_masks"].view(-1, edu_length),
            ).last_hidden_state

        # bert_inputs = []
        # for idx in range(self.start_layer, self.bert_layers):
        #     bert_input = textual_repr[idx][:, 0]
        #     bert_inputs.append(bert_input)

        # proj_hiddens = []
        # for idx in range(self.layer_num):
        #     proj_hidden = self.mlp_words[idx](bert_inputs[idx])
        #     proj_hiddens.append(proj_hidden)
        # x_embed = self.rescale(proj_hiddens)
        # x_embed = self.drop_emb(x_embed)

        batch_size, audio_num, audio_length, feat_dim = input["audio_features"].shape
        audio_repr, audio_mask = self.audio_encoder(
            input["audio_features"].view(-1, audio_length, feat_dim), 
            input["audio_feature_masks"].view(-1, audio_length),
            )
        multimodal_repr = self.fusion_layer(
            textual_repr, input["input_masks"].view(-1, edu_length), 
            audio_repr, audio_mask.float())

        return textual_repr, audio_repr
