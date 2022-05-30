import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, SequentialSampler
import torch
import sys
from typing import Dict
from callbacks import LoggingCallback
from losses import MNRLoss as mnr_loss
from datasets import SciGenMaskedEntity, BatchSampler
from scientific_notation import convert_text_scientific_notation
from metrics import em_score_drop, eval_meteor, eval_mover_score, f1_score_drop
from args import add_args
import pprint

from transformers import (
    AdamW,
    AutoConfig,    
    AutoModelWithLMHead,
    AutoTokenizer    
)


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        "Initialize a model."

        super().__init__()

        self.pp = pprint.PrettyPrinter(indent=4)

        hparams = vars(hparams)
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            **({}),
            cache_dir=cache_dir)                
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir
        )

        new_tokens = {'additional_special_tokens': ['<R>', '<C>', '<CAP>', '<EXP>', '[EXP]', '[F]', '<T>', '<ST>', '<DESC>', '<H>']}
        self.tokenizer.add_special_tokens(new_tokens)

        logger.info('We have added new special tokens: %s' % self.tokenizer.additional_special_tokens)        
        self.model = AutoModelWithLMHead.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
            )            
        logger.info('We have added new special tokens: %s' % self.tokenizer.additional_special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids=None, attention_mask=None,  
        decoder_input_ids=None, inputs_embeds=None, labels=None, return_dict=False, output_hidden_states=False):

        if inputs_embeds is not None:
            # labels are not passed since loss calculation is not possible in this
            # scenario (https://github.com/huggingface/transformers/issues/12475).
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
                decoder_input_ids=decoder_input_ids, return_dict=return_dict, 
                output_hidden_states = output_hidden_states)
        else:
            return self.model(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, 
                labels=labels, return_dict=return_dict, output_hidden_states = output_hidden_states)            
       
    def model_pass(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100       

        outputs = self(source_ids, attention_mask=source_mask, 
            decoder_input_ids=y_ids, labels=lm_labels, return_dict=True, output_hidden_states=True)
        
        return outputs, lm_labels

    def log_dict_prefix(self, metrics:Dict, prefix:str):       
        for key in list(metrics):
            metrics[prefix + '_' + key] = metrics.pop(key)
        self.log_dict(metrics)

    def _generation(self, batch):

        skip_special_tokens = False if self.hparams.scientific_notation\
            else True
        source_ids, source_mask, y = SciGenMaskedEntity.trim_seq2seq_batch(batch, self.tokenizer.pad_token_id)            
        device = 'cuda' if self.hparams.n_gpu >= 1 else 'cpu'
        
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            max_length=self.hparams.max_target_length,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )

        preds =\
            [self.tokenizer.decode(g, skip_special_tokens=skip_special_tokens,  
                clean_up_tokenization_spaces=True) for g in generated_ids]
        target =\
            [self.tokenizer.decode(t, skip_special_tokens=skip_special_tokens, 
                clean_up_tokenization_spaces=True) for t in y]
        
        pred_tokens, target_tokens = [], []        
        if 'masked_ids_target' in batch.keys():
            for i in range(0, len(batch['masked_ids_target']), 2):
                pred_tokens\
                    .append(self.tokenizer.decode([generated_ids[i][pos].item() 
                        if pos < len(generated_ids[i]) else 1
                        for pos in batch['masked_ids_target'][i]], skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True).strip())
                target_tokens\
                    .append(self.tokenizer.decode([batch['target_ids'][i][pos]
                        .item() for pos in batch['masked_ids_target'][i]], skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True).strip())

        return {'predictions': preds,
                'targets': target,
                'predicted_tokens': pred_tokens,
                'target_tokens': target_tokens}

    def _val_epoch_end(self, outputs, prefix):

        metrics = None

        if "predictions" in outputs[0]:
            output_predictions_file = os.path.join(self.hparams.output_dir, prefix + "_predictions_" + str(self.count_valid_epoch) + ".txt")
            output_targets_file = os.path.join(self.hparams.output_dir, prefix + "_targets_" + str(self.count_valid_epoch) + ".txt")

            output_predicted_tokens = os.path.join(self.hparams.output_dir, prefix + "_predicted_masked_tokens_" + str(self.count_valid_epoch) + ".txt")
            output_targets_tokens = os.path.join(self.hparams.output_dir, prefix + "_target_masked_tokens_" + str(self.count_valid_epoch) + ".txt")                                                        

            # write predictions and targets for later rouge evaluation.
            with open(output_predictions_file, "w") as p_writer, open(output_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    preds = [s + "\n" for s in output_batch["predictions"]] if not self.hparams.scientific_notation \
                        else [convert_text_scientific_notation(s, self.tokenizer.all_special_tokens) + '\n' for s in output_batch["predictions"]]
                    tars = [s + "\n" for s in output_batch["targets"]] if not self.hparams.scientific_notation \
                        else [convert_text_scientific_notation(s, self.tokenizer.all_special_tokens)  for s in output_batch["targets"]]
                    p_writer.writelines(preds)
                    t_writer.writelines(tars)
                p_writer.close()
                t_writer.close()

        if 'pred_masked_tokens' in outputs[0]:
            with open(output_predicted_tokens, "w") as p_writer, open(output_targets_tokens, "w") as t_writer:
                for output_batch in outputs:                    
                    preds = [s + "\n" for s in output_batch["pred_masked_tokens"]] if not self.hparams.scientific_notation \
                        else [convert_text_scientific_notation(s, self.tokenizer.all_special_tokens) + '\n' for s in output_batch["preds"]]
                    tars = [s + "\n" for s in output_batch["target_masked_tokens"]] if not self.hparams.scientific_notation \
                        else [convert_text_scientific_notation(s, self.tokenizer.all_special_tokens) + '\n' for s in output_batch["target"]]
                    p_writer.writelines(preds)
                    t_writer.writelines(tars)
                p_writer.close()
                t_writer.close()                

        metrics = {}
        metrics["step_count"] = self.step_count

        if self.hparams.em_score:
            metrics["{}_em_score".format(prefix)] =\
                em_score_drop(output_predictions_file, output_targets_file)
            metrics["{}_f1_score".format(prefix)] =\
                f1_score_drop(output_predictions_file, output_targets_file)
        else:
            moverScore = eval_mover_score(output_targets_file, 
                output_predictions_file)                
            meteorScore = eval_meteor(output_targets_file, 
                output_predictions_file)
            metrics["{}_mover_mean".format(prefix)] = float(moverScore[0])
            metrics["{}_mover_median".format(prefix)] = float(moverScore[1])
            metrics["{}_meteor_mean".format(prefix)] = float(meteorScore)
        
        if metrics is not None:
            if prefix == 'val':
                self.pp.pprint('VALIDATION METRICS:')
                self.pp.pprint(metrics)
            self.log_dict(metrics, sync_dist=True)

    def is_logger(self):
        return True

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, 
        on_tpu=None, using_native_amp=None, using_lbfgs=None):
         #if self.trainer.use_tpu:
        #    xm.optimizer_step(optimizer)
        #else:
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_progress_bar_dict(self):
                
        running_train_loss = self.trainer.fit_loop.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

def generic_train(model: BaseTransformer, args: argparse.Namespace,
            early_stopping_callback=False,  checkpoint_callback=None,
            ):
    # init model
    set_seed(args)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not checkpoint_callback:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=-1
    )

    if not early_stopping_callback:
        callbacks = [LoggingCallback(), checkpoint_callback]
    else:
        callbacks = [LoggingCallback(), checkpoint_callback, early_stopping_callback]

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,        
        gradient_clip_val=args.max_grad_norm,        
        callbacks=callbacks,
        flush_logs_every_n_steps=1,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True,
        log_every_n_steps=1
    )

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    if args.val_check_interval is not None:
        train_params['val_check_interval'] = int(args.val_check_interval)

    if args.limit_val_batches is not None:
        train_params['limit_val_batches'] = int(args.limit_val_batches)

    tb_logger = pl_loggers.TensorBoardLogger(args.output_dir + '/tensorboard/')
    trainer = pl.Trainer(logger=tb_logger, **train_params)

    if args.do_train:
        trainer.fit(model)

    return trainer
