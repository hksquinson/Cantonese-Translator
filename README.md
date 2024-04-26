# AIST4010-Cantonese-Translator

This is a Cantonese translator for the AIST4010 course project. The translator is implemented using the Yi-6B-Chat base model, using various sources such as Cantonese Wikipedia, OpenRice, Kaifangcidian and other sources. The model is trained using Huggingface Transformers and PEFT libraries.

## Notable Files

- `train_mono.py`: Pretraining script for the model for training on monolingual data.
- `train_parallel.py`: Training script for the model for training on parallel data.
- `train_parallel_with_pretrained.py`: Training script for the model for training on parallel data with a pretrained model.
- `inference.py`: Inference script for the model. The test input can be changed within the script.
- `eval_base.py`: Script for generating outputs on the FLORES+ test set using the base model.
- `eval_sft_only.py`: Script for generating outputs on the FLORES+ test set using the model trained with only parallel data.
- `eval_pretrained_sft.py`: Script for generating outputs from the FLORES+ test set using the model trained with both monolingual and parallel data.
