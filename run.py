import numpy as np
import pandas as pd
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from pprint import pprint

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger

from sklearn.model_selection import train_test_split

from model import TransformerTagger

# global parameters

TRAIN_PATH = './SemEval2020_Task10_Emphasis_Selection/train_dev_data/train.txt'
VAL_PATH = './SemEval2020_Task10_Emphasis_Selection/train_dev_data/dev.txt'
TEST_PATH = './SemEval2020_Task10_Emphasis_Selection/test_data/test_data.txt'


def set_seed(hparams):
    random.seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    torch.manual_seed(hparams.random_seed)
    if hparams.n_gpu > 0:
        torch.cuda.manual_seed_all(hparams.random_seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1)
    # parser.add_argument('--fp16', action='store_true')
    # parser.add_argument('--fp16_opt_level', type=str, default='O1')
    # parser.add_argument('--n_tpu_cores', type=int, default=0)

    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--model_name_or_path', type=str, default='bert-large-cased')
    parser.add_argument('--config_name', type=str, default='')
    parser.add_argument('--tokenizer_name', type=str, default='')
    parser.add_argument('--max_seq_length', type=int, default=70)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--label_mode', type=str, default='major', help='separate, major, ann_N, test')
    parser.add_argument('--label_type', type=str, default='token', help='token, prob')
    parser.add_argument('--do_lower_case', action='store_true')

    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--data_cache_dir', type=str, default='cached_data')
    parser.add_argument('--output_dir', type=str, default='')

    parser.add_argument('--train_path', type=str, default=TRAIN_PATH)
    parser.add_argument('--val_path', type=str, default=VAL_PATH)
    parser.add_argument('--test_path', type=str, default=TEST_PATH)

    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--frac_warmup_steps', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--early_stop_monitor', type=str, default='val_score_mean')
    parser.add_argument('--early_stop_patience', type=int, default=1)

    parser.add_argument('--random_seed', type=int, default=19)

    hparams = parser.parse_args()

    if hparams.early_stop_monitor == 'val_loss':
        hparams.early_stop_mode = 'min'
    elif hparams.early_stop_monitor in ['val_acc', 'val_f1', 'val_score_mean']:
        hparams.early_stop_mode = 'max'
    else:
        raise ValueError('Choose early_stop_monitor from val_loss, val_acc, val_f1')

    # set random seed
    set_seed(hparams)

    # callbacks
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stop_monitor,
        mode=hparams.early_stop_mode,
        patience=hparams.early_stop_patience,
        min_delta=0.0,
        verbose=True,
    )

    # callback on validation score
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=f'{hparams.model_name_or_path}_bs{hparams.train_batch_size}_acc{hparams.accumulate_grad_batches}'
    #          f'_lr{hparams.lr}_labmode-{hparams.label_mode}_ckpt',
    #     monitor='val_score_mean',
    #     verbose=True,
    #     save_top_k=1,
    #     save_weights_only=False,
    #     mode='max',
    #     period=1,
    # )

    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=f'{hparams.model_name_or_path}_bs{hparams.train_batch_size}_acc{hparams.accumulate_grad_batches}'
             f'_lr{hparams.lr}_labmode-{hparams.label_mode}',
    )

    pprint(vars(hparams))

    # define Trainer
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        early_stop_callback=early_stop_callback,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        gradient_clip_val=hparams.gradient_clip_val,
        max_epochs=hparams.max_epochs,
        val_check_interval=hparams.val_check_interval,
        gpus=hparams.n_gpu,
        progress_bar_refresh_rate=1,
    )

    model = TransformerTagger(hparams)
    trainer.fit(model)
