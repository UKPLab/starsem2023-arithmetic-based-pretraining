import argparse
import logging
import os
import time
from typing import List
from pathlib import Path
from collections import defaultdict

import torch
import glob
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import sys
from lightning_base import BaseTransformer, generic_train
from callbacks import get_checkpoint_callback, get_early_stopping_callback
from metrics import f1_score, em_score, em_score_drop, f1_score_drop
from torch.nn import CrossEntropyLoss
from losses import MNRLoss as mnr_loss, masked_loss
from datasets import SciGenMaskedEntity, BatchSampler, BartDataset
from utils import visualize_embeddings
from args import add_args

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class CustomTrainer(BaseTransformer):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.metrics_save_path = Path(self.hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.hparams.output_dir) / "hparams.pkl"
        self.step_count = 0
        self.count_valid_epoch = 0
        self.count_test_epoch = 0
        self.metrics = defaultdict(list)
        self.embeds_1, self.embeds_2 = None, None

        # initialize loss functions 
        self.contrastive_loss = torch.tensor([[0.0]], requires_grad=True)\
            .to('cuda' if self.hparams.n_gpu >= 1 else 'cpu')
        self.loss_function = CrossEntropyLoss()
        self.masked_loss = None

        # initialize dataset parameter
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            scientific_notation=self.hparams.scientific_notation,
            char_level_representation=self.hparams.char_level_representation,
            verbalized = self.hparams.verbalized,
            masked_number_prediction=True if \
                args.masked_number_prediction else False,
            masked_number_prediction_contrastive=True \
                if args.masked_number_prediction_contrastive else False
        )

        # finally print parameters
        logger.info("parameters %s", hparams)

    def _step(self, batch, return_contrastive_embeds=False):
                
        device = 'cuda' if self.hparams.n_gpu >= 1 else 'cpu'
        outputs, lm_labels = self.model_pass(batch)
        loss = outputs[0]       

        if self.dataset_kwargs['masked_number_prediction'] or\
            self.dataset_kwargs['masked_number_prediction_contrastive']:

            # Calculate masked loss
            if self.masked_loss is not None:
                self.masked_loss = self.masked_loss.clone().detach()
            self.masked_loss = masked_loss(batch, lm_labels, outputs[1],
                self.masked_loss, 1)

            if self.dataset_kwargs['masked_number_prediction_contrastive']:
                embeds_char_tok = torch.stack([torch.mean(outputs['encoder_last_hidden_state'][i][span[0]:span[1]], dim=0) for i in range(0, len(batch['source_splitted_indices']), 2) for span in batch['source_splitted_indices'][i]]).to(device)
                embeds_classic = torch.stack([torch.mean(outputs['encoder_last_hidden_state'][i][span[0]:span[1]+1], dim=0) for i in range(1, len(batch['source_splitted_indices']), 2) for span in batch['source_splitted_indices'][i]]).to(device)
                contrastive_loss = mnr_loss(embeds_char_tok, embeds_classic)

                inp_loss = (1/2) * self.masked_loss + (1/2) * contrastive_loss

                overall_loss = (1/2) * loss + (1/2) * inp_loss
                to_return = {'overall_loss': overall_loss,                     
                            'masked_loss': self.masked_loss,
                            'contrastive_loss': contrastive_loss}
            else:                
                loss += (1/2) * loss + (1/2) * self.masked_loss
                to_return = {'overall_loss': loss,                     
                            'masked_loss': self.masked_loss}
        else:
            to_return = {'overall_loss': loss}

        if return_contrastive_embeds:
            return to_return, embeds_char_tok, embeds_classic
        else:
            return to_return
                
    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log_dict_prefix(metrics, 'train')
        return {"loss": metrics['train_overall_loss']}

    def validation_step(self, batch, batch_idx):        

        outputs = self._generation(batch)            
        
        # necessary for early stopping
        metrics = self._step(batch)
        self.log_dict_prefix(metrics, 'val')
        
        return outputs
        
    def test_step(self, batch, batch_idx):

        outputs = self._generation(batch) 

        if self.dataset_kwargs['masked_number_prediction'] or\
            self.dataset_kwargs['masked_number_prediction_contrastive']:        

            if self.dataset_kwargs['masked_number_prediction_contrastive']:
                metrics, embeds_1, embeds_2 = self._step(batch, 
                        return_contrastive_embeds=True)

                self.embeds_1 = embeds_1 if self.embeds_1 is None\
                    else torch.cat((self.embeds_1, embeds_1))
                self.embeds_2 = embeds_2 if self.embeds_2 is None\
                    else torch.cat((self.embeds_2, embeds_2))
            else:
                metrics = self._step(batch)
                    
        else:
            metrics = self._step(batch)

        self.log_dict_prefix(metrics, 'test')
        
        return outputs

    def test_epoch_end(self, outputs):
        if self.dataset_kwargs['masked_number_prediction_contrastive']:        
            #visualize_embeddings(self.embeds_1, self.embeds_2)            
            pass
        self._val_epoch_end(outputs, 'test')
        
    def validation_epoch_end(self, outputs):
        self.step_count += 1
        self.count_valid_epoch += 1
        self._val_epoch_end(outputs, 'val')
    
    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:

        logger.info('loading dataloader...')
        self.dataset_kwargs['type_path'] = type_path

        if self.dataset_kwargs['masked_number_prediction'] or \
            self.dataset_kwargs['masked_number_prediction_contrastive']:
            dataset = SciGenMaskedEntity(self.tokenizer, 
                self.dataset_kwargs['masked_number_prediction_contrastive'],
                **self.dataset_kwargs
            )

            if self.dataset_kwargs['masked_number_prediction_contrastive']:
                sampler = BatchSampler(SequentialSampler(dataset), 
                    batch_size=batch_size, shuffle=shuffle)
                return DataLoader(dataset, batch_sampler=sampler, 
                    collate_fn=dataset.collate_fn, num_workers=0)
    
        else:
            dataset = BartDataset(self.tokenizer, **self.dataset_kwargs)
        
        return DataLoader(dataset, batch_size=batch_size, 
            collate_fn=dataset.collate_fn, shuffle=shuffle, num_workers=0)

    def train_dataloader(self) -> DataLoader:   
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)        
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        
        return dataloader

    def val_dataloader(self) -> DataLoader:        
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size, shuffle=False)       

    def test_dataloader(self) -> DataLoader:        
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size, shuffle=False)

def main(args):

    if [args.char_level_representation, args.scientific_notation].count(True) > 1:
        logger.info('You can either use scientific notation or char-level tokenization!')
        exit(1)

    val_metric = "val_mover_mean" if args.mover_score else "val_em_score"

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    model = CustomTrainer(args)

    if args.checkpoint_model:
        print('>>>>>>OOOOOO<<<<<<')
        model = model.load_from_checkpoint(args.checkpoint_model, hparams=args)
        logger.info("args.data_dir: %s", args.data_dir)
        model.dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            scientific_notation=args.scientific_notation,
            char_level_representation=args.char_level_representation,
            verbalized = args.verbalized,
            masked_number_prediction=True if \
                args.masked_number_prediction else False,
            masked_number_prediction_contrastive=True \
                if args.masked_number_prediction_contrastive else False
        )
        new_tokens = {'additional_special_tokens': ['<T>', '<ST>', '<DESC>', '<H>']}
        print('LENGTH: ', len(model.tokenizer))
        model.tokenizer.add_special_tokens(new_tokens)
        model.model.resize_token_embeddings(len(model.tokenizer))
        print('LENGTH: ', len(model.tokenizer))
        print('We have added new special tokens: %s' % model.tokenizer.additional_special_tokens)

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    trainer = generic_train(model, args, 
                   checkpoint_callback=get_checkpoint_callback(args.output_dir, val_metric), early_stopping_callback=es_callback)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:

        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        if args.checkpoint_model:
            trainer.test(model)
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
            if checkpoints:
                
                print('Loading weights from {}'.format(checkpoints[-1]))
                model = model.load_from_checkpoint(checkpoints[-1], 
                    hparams=args)
                model.dataset_kwargs: dict = dict(
                    data_dir=args.data_dir,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,   
                    scientific_notation=args.scientific_notation,
                    char_level_representation=args.char_level_representation,
                    verbalized = args.verbalized,
                    masked_number_prediction=True if \
                        args.masked_number_prediction else False,
                    masked_number_prediction_contrastive=True \
                        if args.masked_number_prediction_contrastive else False
                )
               
            trainer.test(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser) #, os.getcwd())    
    args = parser.parse_args()
    main(args)
