from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from accelerate import infer_auto_device_map
from pydantic import BaseModel, Field
import torch
from tqdm import trange


class TranslationInput(BaseModel):
    src_lang: str = Field(default="Cantonese", description="Source language")
    tgt_lang: str = Field(default="Cantonese", description="Target language")
    text: str = Field(default="", description="Text to translate")
    max_length: int = Field(default=512, description="Maximum length of the generated translation")

class BatchTranslationInput(BaseModel):
    src_lang: str = Field(default="Cantonese", description="Source language")
    tgt_lang: str = Field(default="Cantonese", description="Target language")
    batch_text: list[str] = Field(default=[], description="Text to translate")
    batch_size: int = Field(default=2, description="Batch size for translation")
    max_length: int = Field(default=512, description="Maximum length of the generated translation")

class ModelConfig(BaseModel):
    base_model: str = Field(default="01-ai/Yi-6B-Chat", description="Base model to use for translation")
    adapter: Optional[str] = Field(default=None, description="Adapter to load")
    device: Optional[str] = Field(default=None, description="Device to use for translation")
    eval: bool = Field(default=False, description="Whether to run model in inference mode")
    quantization: Optional[str] = Field(default=None, description="Quantization method: '8bit', '4bit', or None")

class CantoneseTranslator:

    def __init__(self, **kwargs):
        
        config = ModelConfig(**kwargs)
        
        # Set up quantization config
        if config.quantization == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif config.quantization == '4bit':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            quantization_config = None

        device_map = "auto"

        # Load the model with quantization if specified
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map=device_map,
            # torch_dtype=dtype,
            quantization_config=quantization_config
        )

        padding_side = "left" if config.eval else "right"
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, padding_side=padding_side)
        # self.device = config.device

        # if self.device is None:
        #     self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if config.adapter is not None:
            self.model.load_adapter(config.adapter)

        if config.eval:
            self.model.eval()
        elif quantization_config in ["4bit", "8bit"]:
            self.model = prepare_model_for_kbit_training(self.model)

    
    def translate(self, **kwargs):
        input_data = TranslationInput(**kwargs)

        # Ensure either src_lang or tgt_lang is Cantonese
        if input_data.src_lang != "Cantonese" and input_data.tgt_lang != "Cantonese":
            raise ValueError("Either src_lang or tgt_lang must be 'Cantonese'")
        elif input_data.src_lang == "Cantonese" and input_data.tgt_lang == "Cantonese":
            raise ValueError("src_lang and tgt_lang cannot both be 'Cantonese'")

        # Determine translation direction and construct system message
        system_message = {"role": "system", "content": f"Translate the given {input_data.src_lang} words into {input_data.tgt_lang}."}
        
        messages = [system_message, {"role": "user", "content": input_data.text}]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.model.device), max_length=input_data.max_length)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def batch_translate(self, **kwargs):

        input_data = BatchTranslationInput(**kwargs)
        batch_size = input_data.batch_size

        # Ensure either src_lang or tgt_lang is Cantonese
        if input_data.src_lang != "Cantonese" and input_data.tgt_lang != "Cantonese":
            raise ValueError("Either src_lang or tgt_lang must be 'Cantonese'")
        elif input_data.src_lang == "Cantonese" and input_data.tgt_lang == "Cantonese":
            raise ValueError("src_lang and tgt_lang cannot both be 'Cantonese'")
        
        results = []
        
        for i in trange(0, len(input_data.batch_text), batch_size):
            batch = input_data.batch_text[i:i+batch_size]
            system_message = {"role": "system", "content": f"Translate the given {input_data.src_lang} words into {input_data.tgt_lang}."}
            messages = [[system_message, {"role": "user", "content": line}] for line in batch]
            input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True, padding_side='left')
            output_ids = self.model.generate(input_ids.to(self.model.device), max_length=input_data.max_length)
            output_ids = [output_ids[i][input_ids[i].shape[0]:] for i in range(len(output_ids))]
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(response)
        
        return results

        
    
    def translate_from_cantonese(self, tgt_lang, text):
        return self.translate(tgt_lang=tgt_lang, text=text)
    
    def translate_to_cantonese(self, src_lang, text):
        return self.translate(src_lang=src_lang, text=text)
    
    def translate_dataset(self, dataset, src_lang, tgt_lang, current_time):
        language_data = dataset.get_language_data(src_lang)
        language_data = language_data[:5]
        dataset_name = dataset.get_name()
        results = [self.translate(src_lang=src_lang, tgt_lang=tgt_lang, text=line) for line in language_data]
        path = f"results/{current_time}/{dataset_name}_{src_lang}_to_{tgt_lang}.txt"
        with open(path, 'w') as f:
            for result in results:
                f.write(result + '\n')
