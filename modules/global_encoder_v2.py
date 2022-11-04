import torch.nn as nn
import torch
from modules.layer import NonLinear
from modules.scale_mix import *
import math

class GlobalEncoder(nn.Module):
    def __init__(self, vocab, config, multimodal_encoder):
        super(GlobalEncoder, self).__init__()
        self.multimodal_encoder = multimodal_encoder
        self.drop_emb = nn.Dropout(config.dropout_emb)

        config.bert_hidden_size = multimodal_encoder.text_encoder.bert.config.hidden_size

        self.mlp_words = NonLinear(config.bert_hidden_size, config.word_dims, activation=nn.Tanh())
        self.mlp_fusion = NonLinear((config.audio_hidden_dim + config.word_dims), config.word_dims, activation=nn.Tanh())

        nn.init.kaiming_uniform_(self.mlp_words.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')

        self.edu_GRU = nn.GRU(input_size=config.word_dims,
                              hidden_size=config.gru_hiddens // 2,
                              num_layers=config.gru_layers,
                              bidirectional=True, batch_first=True)
        self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)

    def forward(self, input_ids, token_type_ids, attention_mask, edu_lengths, 
    audio_feats, audio_feature_masks, feature_lengths
    ):
        batch_size, max_edu_num, max_tok_len = input_ids.size()

        input = {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "input_masks": attention_mask, 
            "audio_features": audio_feats, 
            "audio_feature_masks": audio_feature_masks, 
            "feature_lengths": feature_lengths
        }
        # encoder_outputs = self.multimodal_encoder(input)
        textual_repr, audio_repr = self.multimodal_encoder(input)

        proj_hidden = self.mlp_words(textual_repr[:, 0])
        textual_embed = proj_hidden.view(batch_size, max_edu_num, -1)

        audio_repr_hidden = torch.mean(audio_repr, dim=1)
        audio_embed = audio_repr_hidden.view(batch_size, max_edu_num, -1)

        x_embed = torch.cat([textual_embed, audio_embed], dim=2)
        x_embed = self.mlp_fusion(x_embed)

        gru_input = nn.utils.rnn.pack_padded_sequence(x_embed, edu_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.edu_GRU(gru_input)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.hidden_drop(outputs[0])

        global_output = torch.cat([x_embed, hidden], dim=-1)

        return global_output
