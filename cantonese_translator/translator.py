from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from pydantic import BaseModel, Field
import torch

class TranslationInput(BaseModel):

    src_lang: str = Field(None, description="Source language")
    tgt_lang: str = Field(None, description="Target language")
    text: str = Field(None, description="Text to translate")
    max_length: int = Field(default=512, description="Maximum length of the generated translation")

class ModelConfig(BaseModel):

    base_model: str = Field("01-ai/Yi-6B-Chat", description="Base model to use for translation")
    adapter: str = Field(None, description="Adapter to load")
    device: str = Field(None, description="Device to use for translation")
    eval: bool = Field(False, description="Whether to run model in inference mode")
    quantization: str = Field(None, description="Quantization method: '8bit', '4bit', or None")

class CantoneseTranslator:

    def __init__(self, **kwargs):
        
        config = ModelConfig(**kwargs)
        
        # Set up quantization config
        if config.quantization == '8bit':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype=torch.float16
        elif config.quantization == '4bit':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            dtype=torch.float32
        else:
            quantization_config = None
            dtype=torch.float32

        # Load the model with quantization if specified
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map=config.device,
            torch_dtype=dtype,
            quantization_config=quantization_config
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.device = config.device
        
        if config.adapter is not None:
            self.model.load_adapter(config.adapter)
        if config.eval:
            self.model.eval()


    def translate_to_cantonese(self, **kwargs):

        input_data = TranslationInput(**kwargs)
        system_message = {"role": "system", "content": f"Translate the given {input_data.src_lang} words into Chinese."}
        messages = [system_message, {"role": "user", "content": input_data.text}]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device), max_length=input_data.max_length)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def translate_from_cantonese(self, **kwargs):

        input_data = TranslationInput(**kwargs)
        system_message = {"role": "system", "content": f"Translate the given Chinese words into {input_data.tgt_lang}."}
        messages = [system_message, {"role": "user", "content": input_data.text}]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device), max_length=input_data.max_length)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response