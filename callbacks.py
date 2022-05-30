import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics            

            # tensorboard logging
            if "log" in metrics.keys():
                self.log_dict(metrics['log'])

            # console logging
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}".format(key, str(metrics[key])))
                        writer.write("{} = {}".format(key, str(metrics[key])))

def get_checkpoint_callback(output_dir, metric):
    """Saves the best model by validation ROUGE2 score."""

    mode = "min" if 'loss' in metric else "max"   
    print('MODE OF CHECKPOINT CALLBACK: %s' % mode)    

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="{metric:.3f}-{step_count}",
        monitor=f"{metric}",
        mode=mode,
        save_top_k=5,
        every_n_epochs= 1,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback

def get_early_stopping_callback(metric, patience):
    mode = "min" if 'loss' in metric else "max"    
    return EarlyStopping(monitor=f"{metric}", mode=mode, patience=patience, verbose=True,)

