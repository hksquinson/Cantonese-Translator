# %%
import os
import gc

import pandas as pd
import numpy as np

import torch

from modelscope import snapshot_download

from transformers import AutoTokenizer, PreTrainedTokenizerFast, LlamaTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm

from custom_tokenizers import YueTokenizer

# %%
PROJ_DIRECTORY = r'AIST4010-Cantonese-Translator/'
DATA_DIRECTORY = r'AIST4010-Cantonese-Translator-Data/'

yue_path = r'AIST4010-Cantonese-Translator/yue_tokenizer/yue'
model_path =r'01ai/Yi-6B-Chat'

#%%
print(os.listdir())

# %%
model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='', revision='master')
yi_tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='', local_files_only=True, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')
print(len(yi_tokenizer.get_vocab()))
yue_tokenizer = YueTokenizer.from_pretrained(model_path, padding_side='right', max_length=512, return_tensors='pt')

# %%
print(len(yi_tokenizer.get_vocab()))

# %%
print(yue_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yi_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yue_tokenizer.tokenize("嗌呃畀啲嘢噃"))
print(yi_tokenizer.tokenize("嗌呃畀啲嘢噃"))

# %%
# raise Exception

# %%
# yi_tokenizer.save_pretrained('/root/new_tokenizer')

# %%
# print(len(yi_tokenizer.get_vocab()))
# print(yi_tokenizer.get_vocab())

# %%



