#%%
import torch
from cantonese_translator import CantoneseTranslator, ModelConfig, TranslationInput

#%%
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = CantoneseTranslator(device=device)
    print(translator.device)
    print(translator.translate_to_cantonese(src_lang="English", text="Hello, how are you?"))

if __name__ == "__main__":
    main()