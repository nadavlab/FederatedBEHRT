import glob
import os
import logging
import ast
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.init as init
from common.common import load_obj
from dataLoader.MLM import MLMLoader
from model.utils import age_vocab

from torch.utils.data import Dataset
from sklearn.metrics import precision_score

logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
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


def create_dataset(data_path: str, bert_vocab: Dict, age_vocab_dict: Dict, max_len_seq: int, min_visit: int) -> Dataset:
    df = pd.read_csv(data_path)
    token2idx = bert_vocab['token2idx']
    df['length'] = df['code'].apply(lambda codes: count_visits(codes))
    df = df[df['length'] >= min_visit]
    df = df.reset_index(drop=True)
    if not _is_dataset_valid(df):
        return None
    return MLMLoader(dataframe=df, token2idx=token2idx, age2idx=age_vocab_dict, max_len=max_len_seq)

def _is_dataset_valid(df: pd.DataFrame) -> bool:
    number_of_rows = df.shape[0]
    return number_of_rows > 0  # true if valid. false if invalid.

def count_visits(codes) -> int:
    return len([code for code in ast.literal_eval(codes) if code == 'SEP'])


def create_datasets(data_dir_path: str, test_path: str, vocab_pickle_path: str, age_vocab_dict: Dict, max_len_seq: int, min_visit: int):
    bert_vocab = load_obj(name=vocab_pickle_path)
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

def calc_acc(label, pred):
    logs = nn.LogSoftmax(dim=1)
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    if len(outs) == 0:
        return None
    precision = precision_score(truelabel, outs, average='micro')
    return precision
