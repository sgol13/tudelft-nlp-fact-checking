from torch import nn
from transformers import GPT2Tokenizer, GPT2Model


class Gpt2Tokenizer:
    def __init__(self):
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

    def __call__(self, x):
        encoded_dict = self._tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']


class Gpt2Classifier(nn.Module):
    def __init__(self, model_path='gpt2',
                 labels_count=3, mlp_dim=1024, dropout=0.1):
        super().__init__()

        self.gpt2 = GPT2Model.from_pretrained(
            model_path,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_dim = self.gpt2.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )

        for layer in self.gpt2.h[:5]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, tokens, masks):
        backbone_output = self.gpt2(tokens, attention_mask=masks)
        last_hidden_state = backbone_output[0]
        cls_representation = last_hidden_state[:, 0, :]

        mlp_output = self.mlp(cls_representation)
        return mlp_output


GPT2_CONFIG = {
    'name': 'gpt2',
    'model': Gpt2Classifier,
    'tokenizer': Gpt2Tokenizer,
}