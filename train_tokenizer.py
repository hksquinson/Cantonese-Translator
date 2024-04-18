import os
import gc

import pandas as pd
import numpy as np

import torch

from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm

PROJ_DIRECTORY = r'AIST4010-Cantonese-Translator'
DATA_DIRECTORY = r'AIST4010-Cantonese-Translator-Data'

yue_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
special_tokens = ["<unk>", "<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
                  "<|im_sep|>"]
                  
trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=20000, show_progress=True)
yue_tokenizer.pre_tokenizer = Whitespace()

#find all text files in data directory
all_files = []
for root, dirs, files in os.walk(DATA_DIRECTORY):
    for file in files:
        if file.endswith(".txt") or (file.startswith("wiki") and not file.endswith(".py")):
            all_files.append(os.path.join(root, file))

# for file_name in all_files:
#     print(f"Processing {file_name}")

#shuffle all files
all_files = np.array(all_files)
np.random.shuffle(all_files)
            
yue_tokenizer.train(files=all_files, trainer=trainer)

if not os.path.exists(f"{PROJ_DIRECTORY}/yue_tokenizer"):
    os.makedirs(f"{PROJ_DIRECTORY}/yue_tokenizer")

yue_tokenizer.save(f"{PROJ_DIRECTORY}/yue_tokenizer/yue_tokenizer.json")