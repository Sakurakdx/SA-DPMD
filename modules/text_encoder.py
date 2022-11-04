from modules.layer import *
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, plm_name_or_path, config, tok_helper):
        super(TextEncoder, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(plm_name_or_path)
        self.bert.resize_token_embeddings(len(tok_helper.tokenizer))
        print("Load bert model finished.")


        self.tune_start_layer = config.tune_start_layer
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        if self.start_layer > self.bert_layers - 1: self.start_layer = self.bert_layers - 1
        self.layer_num = self.end_layer - self.start_layer

        for p in self.bert.named_parameters():
            p[1].requires_grad = False

        for p in self.bert.named_parameters():
            items = p[0].split('.')
            if len(items) < 3: continue
            if items[0] == 'embeddings' and 0 >= self.tune_start_layer: p[1].requires_grad = True
            if items[0] == 'encoder' and items[1] == 'layer':
                layer_id = int(items[2]) + 1
                if layer_id >= self.tune_start_layer: p[1].requires_grad = True



    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        return outputs

