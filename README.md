BEHRT federated learning Implementation of the `FederatedAveraging` algorithm.
Code of the paper: Federated Learning of Medical Concepts Embedding using BEHRT
https://arxiv.org/abs/2305.13052

**If you used our code, please cite us: **
@article{shoham2023federated,
  title={Federated Learning of Medical Concepts Embedding using BEHRT},
  author={Shoham, Ofir Ben and Rappoport, Nadav},
  journal={arXiv preprint arXiv:2305.13052},
  year={2023}
}
`FederatedAveraging` algorithm proposed in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) in PyTorch.

Thanks to https://github.com/vaseline555/Federated-Learning-PyTorch for the initial code base. 



## Requirements
* See `requirements.txt`

## Configurations
* See `config.yaml`
data_dir_path: directory path that contains the multi-center data (csv files), BEHRT format. Each center data is one csv file. Example format: see behrt_nextvisit_example_data.csv
test_path: path to test csv file, same format as behrt_nextvisit_example_data.csv.
vocab_pickle_path: path to the pickle that contains the vocab. 

You can you this script to create your pickle vocab: https://github.com/Ofir408/BEHRT/blob/master/preprocess/bert_vocab_builder.py

In order to create token2idx you can use the following script:
```
from typing import Dict
import json
import pandas as pd
from typing import List 

def get_all_codes(df: pd.DataFrame, codes_to_ignore: List[str]) -> List[str]:
    codes = []
    for df_list_codes in list(df['code']):
        codes.extend(df_list_codes)
    return list(set(codes) - set(codes_to_ignore))

def get_bert_tokens() -> Dict[str, int]:
    return {
      "PAD": 0,
      "UNK": 1,
      "SEP": 2,
      "CLS": 3,
      "MASK": 4,
    }
    
def build_token2index_dict(df: pd.DataFrame) -> Dict[str, int]:
    token2inx_dict = get_bert_tokens()
    next_index = max(token2inx_dict.values()) + 1
    
    codes = get_all_codes(df= df, codes_to_ignore=token2inx_dict.keys())
    for code in codes:
        token2inx_dict[str(code)] = next_index
        next_index += 1
    return token2inx_dict

def create_token2index_file(df: pd.DataFrame, output_file_path: str):
    token2inx_dict = build_token2index_dict(df= df)
    with open(output_file_path, 'w') as f:
        json.dump(token2inx_dict, f)
        print(f'token2inx was created, path={output_file_path}')

```


## Run
* `python3 main.py` <path_to_config.yaml>


