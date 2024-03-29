# Arithmetic-Based Pretraining -- Improving Numeracy of Pretrained Language Models

This repository provides the code for our paper [Arithmetic-Based Pretraining -- Improving Numeracy of Pretrained Language Models](https://arxiv.org/pdf/2205.06733.pdf). It is an experimental software and is published for the sole purpose of giving additional background details on the publication. This is a cleaned-up and condensed version of what was used to conduct the experiments in the paper (please get in touch if you find errors).

## Citation


Please reference our work as follows:
```
@inproceedings{petrak-etal-2023-arithmetic,
    title = "Arithmetic-Based Pretraining -- Improving Numeracy of Pretrained Language Models",
    author = "Petrak, Dominic  and
      Moosavi, Nafise Sadat  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the The 12th Joint Conference on Lexical and Computational Semantics (*SEM 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.starsem-1.42",
    pages = "477--493",
    abstract = "State-of-the-art pretrained language models tend to perform below their capabilities when applied out-of-the-box on tasks that require understanding and working with numbers (usually referred to as numeracy). Recent work suggests two main reasons for this: (1) popular tokenisation algorithms have limited expressiveness for numbers, and (2) common pretraining objectives do not target numeracy. Approaches that address these shortcomings usually require architectural changes or pretraining from scratch. In this paper, we propose a new extended pretraining approach called Arithmetic-Based Pretraining that jointly addresses both in one extended pretraining step without requiring architectural changes or pretraining from scratch. Arithmetic-Based Pretraining combines contrastive learning to improve the number representation, and a novel extended pretraining objective called Inferable Number Prediction Task to improve numeracy. Our experiments show the effectiveness of Arithmetic-Based Pretraining in three different tasks that require improved numeracy, i.e., reading comprehension in the DROP dataset, inference-on-tables in the InfoTabs dataset, and table-to-text generation in the WikiBio and SciGen datasets.",
}
```

## Project Description


State-of-the-art pretrained language models tend to perform below their capabilities when applied out-of-the-box on tasks that require
reasoning over numbers. Recent work sees two main reasons for this: (1) popular tokenisation algorithms are optimized for common words, and therefore have limited expressiveness for numbers, and (2) common pretraining objectives do not target numerical reasoning or understanding numbers at all. Recent approaches usually address them separately and mostly by proposing architectural changes or pretraining models from scratch. 

__In this work__, we propose a new extended pretraining approach called <ins>Arithmetic-Based Pretraining</ins> that jointly addresses both in one extended pretraining step without requiring architectural changes or pretraining from scratch. Arithmetic-Based Pretraining combines <ins>contrastive learning</ins> to improve the number representation, and a novel extended pretraining objective called <ins>Inferable Number Prediction Task</ins> to improve numeracy.

We evaluate our approach on three different tasks that require numerical reasoning, including (a) reading comprehension in the <ins>DROP</ins> dataset, (b) inference-on-tables in the
<ins>InfoTabs</ins> dataset, and (c) table-to-text generation in <ins>WikiBio</ins> and <ins>SciGen</ins> datasets. Our results on DROP and InfoTabs show that our approach improves the accuracy by 9.6 and 33.9 points on these datasets, respectively. Our human evaluation on SciGen and WikiBio shows that our approach improves the factual correctness on all datasets.

<ins>Why Character-Level Tokenisation?</ins>

Recent work has shown that character-level tokenization is more expressive for representing numbers because it does not rely on frequently observed patterns, as do subword-based tokenization algorithms, but considers each character individually.

## How To

This section briefly describes how to setup an environment for working with our code. Please don't hesitate to contact us if you find any errors or something does not work as expected. 

### Setup

The code was developed for Python 3.8.3. We recommend to create a new virtual environment first. Then:

1. We suggest to first install the [pytorch version](https://pytorch.org/get-started/locally/) that suits your environment. This might prevent errors by installing the provided _requirements.txt_ file.
2. The provided _requirements.txt_ file contains all libraries needed. Installing this file should be sufficient. If you have just installed pytorch, you should remove the corresponding lines from the file. 
3. After installing the requirements, you should install the package itself (<code>pip install -e .</code>)to make everything accessible to the environment.

If you want to use BLEURT for evaluation, please stick to [their](https://github.com/google-research/bleurt) installation instructions. The same applies to PARENT ([installation instructions](https://github.com/KaijuML/parent)).

### Dataset Creation

For make the original datasets usable with our code, please follow the instructions given in the README of the _scripts_ folder.

We provide the dataset splits that were used for our experiments with the inferable number prediction task in the _datasets_ folder. However, if you want to recreate them from the original datasets, you can also use the scripts provided in the _scripts_ folder.

### Run Experiments

Our code can be started via command line. The minimal command is:
```bash
python trainer.py --do_train --do_predict --model_name_or_path=facebook/bart-large --output_dir=/path/to/output/dir --data_dir=/path/to/data/dir --masked_number_prediction_contrastive --em_score --char_level_representation
```
This would train a bart-large model on the inferable number prediction task using contrastive learning and em score as validation metric. The following command starts finetuning of such a pretrained model:

```bash
python trainer.py --do_train --do_predict --model_name_or_path=facebook/bart-large --output_dir=/path/to/output/dir --data_dir=/path/to/data/dir --finetuning --mover_score --checkpoint_model=/path/to/pretrained/model --char_level_representation
```

In our experiments, we used EM score as validation metric for reasoning-aware pretraining, and for finetuning in case of DROP and InfoTabs. For WikiBio and SciGen, we use MoverScore for finetuning. A detailed description of each argument along with default values can be found in _args.py_.

If you want to use a GPU, you have to directly target it using _CUDA_VISIBLE_DEVICES_.

### Evaluation

Please follow the instructions given in the README of the _evaluation_ folder.

### Contact Persons

- Dominic Petrak (<petrak@ukp.informatik.tu-darmstadt.de>)
  
### Links

[UKP Lab Homepage](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt Website](https://www.tu-darmstadt.de/index.en.jsp)
