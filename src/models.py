from typing import Dict

from model.NextXVisit import BertForMultiLabelPrediction
from pytorch_pretrained_bert import BertConfig


#################################
# Models for federated learning #
#################################
class CustomBertConfig(BertConfig):
    def __init__(self, config):
        super(CustomBertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings=config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range')
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')


class CustomBertForMultiLabelPrediction(BertForMultiLabelPrediction):
    def __init__(self, name: str, num_labels: int, feature_dict_config: Dict, **kwargs):
        super(CustomBertForMultiLabelPrediction, self).__init__(config=CustomBertConfig(kwargs), num_labels=num_labels,
                                                                feature_dict=feature_dict_config)
        self.name = name
