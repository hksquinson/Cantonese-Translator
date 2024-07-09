
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
from cantonese_translator import CantoneseTranslator

def merge_adapter(base_model_name, adapter, quant, device):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, adapter)
    model = model.merge_and_unload()

    model.save_pretrained(f"models/{adapter.split('/')[-1]}_merged")
    tokenizer.save_pretrained(f"models/{adapter.split('/')[-1]}_merged")


    

if __name__ == "__main__":
    merge_adapter("01-ai/Yi-6B-Chat", "adapters/*peft_model_sft_only_20240705-111842", "none", "cpu")