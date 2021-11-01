import torch.nn as nn
from modules.Layer import *
import torch.nn as nn
import math


class StateEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(StateEncoder, self).__init__()
        self.nonlinear1 = NonLinear(input_size=config.word_dims * 2 + config.gru_hiddens * 2 ,
                                    hidden_size=config.hidden_size,
                                    activation=nn.Tanh())

        self.nonlinear2 = NonLinear(input_size=config.hidden_size,
                                    hidden_size=config.mlp_arc_size + config.mlp_rel_size,
                                    activation=nn.Tanh())

        nn.init.kaiming_uniform_(self.nonlinear1.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.nonlinear2.linear.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='tanh')

    def forward(self, global_outputs, feats):
        batch_size, max_edu_len, _ = global_outputs.size()

        global_outputs = global_outputs.unsqueeze(1).repeat(1, max_edu_len, 1, 1)

        state_input = torch.cat([global_outputs, global_outputs.transpose(1, 2), feats], dim=-1)

        hidden = self.nonlinear1(state_input)
        state_hidden = self.nonlinear2(hidden)
        return state_hidden
