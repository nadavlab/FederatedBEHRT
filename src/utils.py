import ast
import glob
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from dataLoader.NextXVisit import NextVisit
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


#######################
# TensorBoard setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.

    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(
        f"/home/benshoho/.conda/envs/my_env/bin/python -m tensorboard.main --logdir={log_path} --port={port} --host={host}")
    return True


#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model


def create_dataset(data_path: str, bert_vocab: Dict, age_vocab_dict: Dict, max_len_seq: int, min_visit: int) -> \
        Optional[NextVisit]:
    df = pd.read_csv(data_path)
    label_vocab = format_label_vocab(bert_vocab['token2idx'])
    token2idx = bert_vocab['token2idx']
    df['length'] = df['code'].apply(lambda codes: count_visits(codes))
    df = df[df['length'] >= min_visit]
    df = df.reset_index(drop=True)
    if not _is_dataset_valid(df):
        return None
    return NextVisit(token2idx=token2idx, label2idx=label_vocab, age2idx=age_vocab_dict, dataframe=df,
                     max_len=max_len_seq)


def format_label_vocab(token2idx):
    token2idx = token2idx.copy()
    del token2idx['PAD']
    del token2idx['SEP']
    del token2idx['CLS']
    del token2idx['MASK']
    token = list(token2idx.keys())
    label_vocab = {}
    for i, x in enumerate(token):
        label_vocab[x] = i
    return label_vocab


def _is_dataset_valid(df: pd.DataFrame) -> bool:
    number_of_rows = df.shape[0]
    return number_of_rows > 0  # true if valid. false if invalid.


def create_datasets(data_dir_path: str, test_path: str, bert_vocab: Dict, age_vocab_dict: Dict, max_len_seq: int,
                    min_visit: int):
    local_datasets = []
    test_dataset = create_dataset(data_path=test_path, bert_vocab=bert_vocab, age_vocab_dict=age_vocab_dict,
                                  max_len_seq=max_len_seq, min_visit=min_visit)
    if test_dataset is None:
        raise Exception(f"test dataset {test_path} has zero rows for min_visit={min_visit}! exit..")
    for data_path in glob.iglob(f'{data_dir_path}/*'):
        if "test.csv" in data_path:
            continue
        print(f'data_path={data_path}')
        dataset = create_dataset(data_path=data_path, bert_vocab=bert_vocab, age_vocab_dict=age_vocab_dict,
                                 max_len_seq=max_len_seq, min_visit=min_visit)
        if dataset is not None:
            local_datasets.append(dataset)
        else:
            print(f'{data_path} is invalid (zero rows) for min_visit={min_visit}')
    return local_datasets, test_dataset


def calc_measurements(logits, label, threshold=0.5):
    sig = nn.Sigmoid()
    output = sig(logits)
    average_precision = average_precision_score(label.numpy(), output.numpy(), average='samples')
    auc_roc = roc_auc_score(label.numpy(), output.numpy(), average='samples')
    outputs_backup = torch.clone(output)

    output = np.array(output) >= threshold
    recall = recall_score(label.numpy(), output, average='micro')
    f1 = f1_score(label.numpy(), output, average='micro')
    return average_precision, auc_roc, recall, f1, outputs_backup, label


def count_visits(codes) -> int:
    return len([code for code in ast.literal_eval(codes) if code == 'SEP'])


def load_pretrained_model(pretrain_model_path: str, model: nn.Module):
    # load pretrained model and update weights
    pretrained_dict = torch.load(pretrain_model_path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def init_multi_label_binarizer(label_vocab: Dict) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[each] for each in list(label_vocab.values())])
    return mlb
