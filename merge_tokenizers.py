import os
import gc

import pandas as pd
import numpy as np

import torch

from modelscope import snapshot_download

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm

PROJ_DIRECTORY = r'/root/autodl-tmp/AIST4010-Cantonese-Translator/'
DATA_DIRECTORY = r'/root/autodl-tmp/AIST4010-Cantonese-Translator-Data/'

yue_json = r'/root/autodl-tmp/AIST4010-Cantonese-Translator/yue_tokenizer.json'
model_path =r'/root/autodl-tmp/01ai/Yi-6B-Chat'

model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='/root/autodl-tmp', revision='master')
yi_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')


#get vocab of yi tokenizer
yi_vocab = yi_tokenizer.get_vocab()
print(len(yi_vocab))
#get vocab of yue tokenizer
yue_vocab = Tokenizer.from_file(yue_json).get_vocab()
print(len(yue_vocab))

# for each vocab in yue tokenizer, if it is not in yi tokenizer, add it to yi tokenizer
for vocab in tqdm(yue_vocab):
    if vocab not in yi_vocab:
        yi_tokenizer.add_tokens(vocab)

yi_tokenizer.save_pretrained(f"root/new_tokenizer")

print(len(yi_vocab))
