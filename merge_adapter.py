
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def merge_adapter(base_model_name, adapter, model_name = None):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=None
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, adapter)
    model = model.merge_and_unload()

    if model_name == None or model_name == "":
        model_name = f"{adapter.split('/')[-1]}_merged"

    model.save_pretrained(f"models/{model_name}")
    tokenizer.save_pretrained(f"models/{model_name}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge adapter with base model')
    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--adapter', type=str, required=True, help='Adapter path')
    parser.add_argument('--model_name', type=str, help='Model name')
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter, args.model_name)