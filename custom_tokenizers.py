from transformers import LlamaTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast
import os

YUE_PATH = './yue_tokenizer/yue_tokenizer.json'

class YueTokenizer(LlamaTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print('path to yue tokenizer:', os.path.abspath(YUE_PATH))
        self.yue_tokenizer = PreTrainedTokenizerFast(tokenizer_file=YUE_PATH, padding_side='right', max_length=512, return_tensors='pt')
        yue_vocab = list(self.yue_tokenizer.get_vocab().keys())
        new_vocab = set(yue_vocab) - set(self.get_vocab().keys())
        self.add_tokens(list(new_vocab))

    def tokenize(self, text, **kwargs):
        # Pre-tokenize the text using the yue_tokenizer
        pre_tokenized_text = " ".join(self.yue_tokenizer.tokenize(text))
        
        # Tokenize the pre-tokenized text using the yi_tokenizer
        return super().tokenize(pre_tokenized_text, **kwargs)