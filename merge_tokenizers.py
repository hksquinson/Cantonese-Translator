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

yue_path = r'AIST4010-Cantonese-Translator/yue_tokenizer/yue_tokenizer.json'
model_path =r'01ai/Yi-6B-Chat'

# %%
# model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='', revision='master')
yi_tokenizer = LlamaTokenizerFast.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')
yue_tokenizer = PreTrainedTokenizerFast(tokenizer_file=yue_path, padding_side='right', max_length=512, return_tensors='pt')

# %%
print(yue_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yi_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yue_tokenizer.tokenize("嗌呃畀啲嘢噃"))
print(yi_tokenizer.tokenize("嗌呃畀啲嘢噃"))

#compare ids 
print(yue_tokenizer.convert_tokens_to_ids(yue_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃")))
print(yue_tokenizer.convert_tokens_to_ids(yue_tokenizer.tokenize("嗌呃畀啲嘢噃")))

# %%
# add vocab in yue tokenizer but not in yi tokenizer to yi tokenizer
yue_vocab = list(yue_tokenizer.get_vocab().keys())
print(yue_vocab)
new_vocab = set(yue_vocab) - set(yi_tokenizer.get_vocab().keys())
print(len(new_vocab))

print(new_vocab)


# %%
yi_tokenizer.add_tokens(list(new_vocab))

# %%
def pre_tokenize(text):
    tokens = yue_tokenizer.tokenize(text)
    return " ".join(tokens)

text = "嗌呃畀啲嘢噃"
pre_tokenized_text = pre_tokenize(text)
print(yi_tokenizer.tokenize(pre_tokenized_text))

# %%
from transformers import LlamaTokenizer, PreTrainedTokenizerFast

class CustomTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yue_tokenizer = PreTrainedTokenizerFast(tokenizer_file=yue_path, padding_side='right', max_length=512, return_tensors='pt')

    def _tokenize(self, text, **kwargs):
        # Pre-tokenize the text using the yue_tokenizer
        pre_tokenized_text = " ".join(self.yue_tokenizer.tokenize(text))
        
        # Tokenize the pre-tokenized text using the yi_tokenizer
        return super()._tokenize(pre_tokenized_text, **kwargs)

# Load the custom tokenizer
yi_tokenizer = CustomTokenizer.from_pretrained(model_path, padding_side='right', max_length=512, return_tensors='pt')

# Add vocab in yue tokenizer but not in yi tokenizer to yi tokenizer
yue_vocab = list(yue_tokenizer.get_vocab().keys())
new_vocab = set(yue_vocab) - set(yi_tokenizer.get_vocab().keys())
yi_tokenizer.add_tokens(list(new_vocab))

# Test the custom tokenizer
text = "Hello 嗌呃畀啲嘢噃"
print(yi_tokenizer.tokenize(text))

# %%
print(yue_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yi_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(yue_tokenizer.tokenize("嗌呃畀啲嘢噃"))
print(yi_tokenizer.tokenize("嗌呃畀啲嘢噃"))

# %%
yi_tokenizer.save_pretrained('/root/new_tokenizer')

# %%
print(len(yi_tokenizer.get_vocab()))
print(yi_tokenizer.get_vocab())

# %%



