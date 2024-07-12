# Cantonese-Translator

This repository contains my code for the project 'Eﬀicient LLM Fine-tuning for Cantonese Text Translation', developed as part of the AIST4010 Foundation of Applied Deep Learning course at CUHK.

A detailed review of the background, training process and results can be found in the [project report](Efficient_LLM_Fine_tuning_for_Cantonese_Text_Translation_Report__Public_.pdf).

## Results Summary

The following table shows the results of the current training and evaluation process. The results are based on the BLEU score, BERT precision, BERT recall, BERT F1 score, and COMET score for the base model and the model trained with supervised fine-tuning (SFT) on various translation pairs:

| Translation Pair | Model | BLEU | BERTscore Precision | BERTscore Recall | BERTscore F1 | COMET |
|------------------|-------|------------|----------------|-------------|---------|-------------|
| Cantonese → Simplified Chinese | **Base Model** | **34.51** | **0.87** | **0.86** | 0.86 | **0.8945** |
| | SFT | 17.20 | 0.86 | 0.85 | 0.86 | 0.8871 |
| Cantonese → Traditional Chinese | Base Model | 31.74 | 0.84 | 0.84 | 0.84 | 0.8989 |
| | **SFT** | **32.31** | **0.85** | 0.84 | **0.85** | **0.9002** |
| Cantonese → English | **Base Model** | **20.81** | **0.83** | **0.84** | 0.83 | 0.8312 |
| | SFT | 18.26 | 0.82 | 0.83 | 0.83 | **0.8329** |
| Simplified Chinese → Cantonese | Base Model | 21.61 | 0.82 | 0.83 | 0.83 | 0.8721 |
| | **SFT** | **33.19** | **0.84** | **0.84** | **0.84** | **0.8923** |
| Traditional Chinese → Cantonese | Base Model | 24.05 | 0.82 | 0.82 | 0.82 | 0.8804 |
| | **SFT** | **29.71** | **0.83** | **0.83** | **0.83** | **0.8953** |
| English → Cantonese | **Base Model** | **19.29** | 0.78 | **0.78** | 0.78 | **0.7971** |
| | SFT | 18.08 | **0.80** | 0.76 | 0.78 | 0.7917 |

The following table shows the original results of the base model, SFT model and Pre-trained + SFT model on the FLORES+ test set when the project was first completed:

| Translation Pair | Model | BLEU | BERTscore F1 |
|------------------|-------|------------|--------------|
| Cantonese → English | Base model | **22.11** | 0.94 |
| | SFT only | 19.00 | 0.93 |
| | Pretraining + SFT | 18.14 | **0.94** |
| Cantonese → Mandarin | **Base model** | **26.70** | 0.86 |
| | SFT only | 17.28 | 0.86 |
| | Pretraining + SFT | 17.14 | 0.86 |
| Cantonese → Traditional Chinese | Base model | 27.38 | 0.85 |
| | **SFT only** | **32.81** | 0.85 |
| | Pretraining + SFT | 32.16 | 0.85 |
| English → Cantonese | **Base model** | **21.18** | **0.80** |
| | SFT only | 18.92 | 0.78 |
| | Pretraining + SFT | 20.75 | 0.79 |
| Mandarin → Cantonese | Base model | 21.89 | 0.82 |
| | SFT only | 34.10 | **0.83** |
| | **Pretraining + SFT** | **34.62** | **0.83** |
| Traditional Chinese → Cantonese | Base model | 25.00 | 0.83 |
| | SFT only | 30.29 | **0.84** |
| | **Pretraining + SFT** | **31.05** | **0.84** |

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is structured as follows:
- `/`: Contains the main scripts for training, evaluation, and inference.
- `adapters/`: Contains the adapters produced during training.
- `cantonese_translator/`: Contains the classes for the project.
- `configs/`: Contains configuration files for training.
- `data/`: Contains the data used for training and evaluation.
- `models/`: Contains the models used for training and evaluation.
- `results/`: Contains the results of translation and evaluation.

## Usage

### Training

To train the model on parallel data, specify the paths to the data in `configs/train_data_paths.txt` and run the following command:

```bash
python train_sft.py --base_model /path/to/base/model --data_paths configs/train_data_paths.txt
```

Use `--quant 4bit` or `--quant 8bit` to quantize the model during training.

The code for training on monolingual data will be updated soon. The old code can be found in the `train_mono.py` script.

The training scripts produce adapter models in the `adapters` directory. To merge them with a base model, run the following command:

```bash
python merge_adapter.py --base_model /path/to/base/model --adapter adapters/path/to/adapter --model_name merge_sample_model
``` 

### Inference

To test the model with an interactive prompt, run the `single_line_translation.py` script. The following command translates a sample sentence from Cantonese to English:

```bash
python single_line_translation.py --base_model /path/to/base/model --lang "English" --from_cantonese
```

If the --from_cantonese flag is not specified, the model will translate from Cantonese to the language specified in the --lang flag. Otherwise, the model will translate from the language specified in the --lang flag to Cantonese.

Use `--quant 4bit` or `--quant 8bit` to quantize the model during training.



To run a batch translation job from a file, run the `dataset_single_language_translation.py` script. The following command translates a sample of sentences from Cantonese to Traditional Chinese:

```bash
python dataset_single_language_translation.py --base_model /path/to/base/model --adapter path/to/adapter  --file /path/to/sentences --lang "Traditional Chinese" --sample_size 32 --batch_size 4 --from_cantonese --quant 8bit
```

To run a batch translation job from the full FLORES dataset, save the `dev` and `devtest` files from the FLORES+ dataset to the `data/flores+` directory and run the `dataset_all_language_translation.py` script. The following command translates all translation pairs to and from Cantonese in the FLORES+ dataset:

```bash
python dataset_all_language_translation.py --base_model path/to/base/model --adapter path/to/adapter --dataset data/flores+ --quant 8bit
```

### Evaluation

To evaluate the model on the FLORES+ test set, combine samples of the same language from the `dev` and `devtest` files from the FLORES+ dataset and name them `English.txt`, `Cantonese.txt`,  `Simplified_Chinese.txt` and `Traditional_Chinese.txt`. Place them in a separate directory and run the `model_eval.py` script:

```bash
python model_eval.py --truth_dir /flores/combined/samples/directory --model_output_dir path/to/model/outputs  --src_lang "Simplified Chinese" --tgt_lang Cantonese
```

## License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

While I am happy to share my approach and code, I encourage fellow students to use this as inspiration for approaching AI projects and to develop your own unique solution to demonstrate your learning and skills. If you find my code and report useful and would like to use it in your own work, please cite it using the following BibTeX entry:

```bibtex
@article{cantonese-translator,
  title={Eﬀicient LLM Fine-tuning for Cantonese Text Translation},
  author={Hon Kwan Shun Quinson},
  year={2024}
}
```

## Acknowledgements

I would like to thank my course instructer, Prof. Yu Li, for his guidance and support throughout the project. I would also like to thank the teaching assistants of AIST4010 for their help and feedback.

## Contact

Please visit my [personal website](https://hksquinson.github.io/profile) for more information about me and my projects.

