from pathlib import Path
from argparse import Namespace
import random
import pandas as pd
import numpy as np
import logging
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    XLNetConfig,
    XLNetForTokenClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import read_examples_from_file, convert_examples_to_features, match_m, write_csv


MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetForTokenClassification, XLNetTokenizer),
}

FIELDNAMES = [
    'model_name_or_path',
    'label_mode',
    'train_batch_size',
    'accumulate_grad_batches',
    'max_epochs',
    'current_epoch',
    'lr',
    'frac_warmup_steps',
    'weight_decay',
    'random_seed',
    'val_loss',
    'val_acc',
    'val_f1',
    'val_score1',
    'val_score2',
    'val_score3',
    'val_score4',
    'val_score_mean',
]

logger = logging.getLogger(__name__)


class TransformerTagger(pl.LightningModule):
    """
    Parts of class:
        `__init__` - The model/system definition (REQUIRED)

        `forward` - The model/system computations (REQUIRED)
        `training_step`, `training_end` - What happens in the training loop (`training_step` REQUIRED)

        `validation_step`, `validation_end` - What happens in the validation loop (OPTIONAL)
        `test_step`, `test_end` - What happens in the test loop (OPTIONAL)

        `configure_optimizers` - What optimizers to use (REQUIRED)

        `train_dataloader`, `val_dataloader`, `test_dataloader` - What data to use (`train_dataloader` REQUIRED)
    """

    def __init__(self, hparams: Namespace):
        """
        Initialize model and parameters
        """
        super(TransformerTagger, self).__init__()
        self.hparams = hparams
        self.labels = ['O', 'I']
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.hparams.model_type]

        config = config_class.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            num_labels=self.hparams.num_labels,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )
        model = model_class.from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool('.ckpt' in self.hparams.model_name_or_path),
            config=config,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None,
        )

        self.config, self.tokenizer, self.model = config, tokenizer, model
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

        self.prepare_data()

        # get total train steps - for lr scheduler, idk how to do better for now without double loading
        self.total_train_steps = self.get_total_train_steps()
        logger.info('Total training steps: %s', self.total_train_steps)

        # init predictions
        self.preds = {
            'val': defaultdict(dict),
            'test': defaultdict(dict),
        }

        self.model_id_name = (f'{hparams.model_name_or_path}_bs-{hparams.train_batch_size}'
                              f'_accum-{hparams.accumulate_grad_batches}'
                              f'_lr-{hparams.lr}_labmode-{hparams.label_mode}'
                              f'_maxep-{hparams.max_epochs}')

    def configure_optimizers(self):
        """
        Return: any of these 3 options:
            Single optimizer
            List or Tuple - List of optimizers
            Two lists - The first list has multiple optimizers, the second a list of learning-rate schedulers
        """

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_eps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.frac_warmup_steps * self.total_train_steps),
            num_training_steps=self.total_train_steps,
        )
        self.lr_scheduler = scheduler

        # don't return [optimizer], [scheduler] because we'll override optimizer_step
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        """
        Steps for optimizer and lr scheduler
        """
        # if self.trainer.use_tpu:
        #     xm.optimizer_step(optimizer)
        # else:
        #     optimizer.step()
        optimizer.step()
        optimizer.zero_grad()

        # need to define here to step after each batch (by default each epoch for some reason)
        self.lr_scheduler.step()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        Forward pass
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification.forward
        # loss is first element in returned tupled if labels is non None
        # (loss), logits, (hidden_states), (attentions)
        # loss, logits = outputs[:2]
        # if labels is None return logits first

        if labels is not None:
            loss, logits = outputs[:2]
        else:
            logits = outputs[0]
            loss = None

        return loss, logits

    def training_step(self, batch, batch_nb):
        """
        Returns output dict with loss key and optional log, progress keys
        """
        # batch
        input_ids, attention_mask, token_type_ids, labels, emph_probs = batch
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        # XLM and RoBERTa don't use segment_ids
        if self.hparams.model_type != 'distilbert':
            inputs['token_type_ids'] = (
                token_type_ids if self.hparams.model_type in ['bert', 'xlnet'] else None
            )

        # forward and loss
        loss, _ = self.forward(**inputs)

        # logs
        logs = {
            'train_loss': loss,
            'lr': self.lr_scheduler.get_last_lr()[-1],
        }

        # output dict
        output = {
            'loss': loss,
            'progress_bar': logs,
            'log': logs
        }
        return output

    def validation_step(self, batch, batch_nb, dataloader_idx):
        """
        This is the validation loop. It is called for each batch of the validation set.
        Whatever is returned from here will be passed in as a list on validation_end.
        In this step you’d normally generate examples or calculate anything of interest such as accuracy.
        """

        # batch
        input_ids, attention_mask, token_type_ids, labels, emph_probs = batch
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        # XLM and RoBERTa don't use segment_ids
        if self.hparams.model_type != 'distilbert':
            inputs['token_type_ids'] = (
                token_type_ids if self.hparams.model_type in ['bert', 'xlnet'] else None
            )

        # forward and loss
        loss, logits = self.forward(**inputs)

        output = {
            'val_loss_b': loss.detach().cpu(),
            'logits_b': logits.detach().cpu().numpy(),
            'labels_b': labels.detach().cpu().numpy(),
            'emph_probs_b': emph_probs.detach().cpu().numpy(),
        }
        return output

    def validation_epoch_end(self, outputs):
        """
        Outputs has the appended output after each validation step.
        """
        # outputs contain data for 2 dataloaders - val and test
        for out_idx, outputs_i in enumerate(outputs):
            # validation loss
            val_loss_mean = torch.stack([x['val_loss_b'] for x in outputs_i]).mean()

            # predicts
            logits = np.concatenate([x['logits_b'] for x in outputs_i], axis=0)
            labels = np.concatenate([x['labels_b'] for x in outputs_i], axis=0)
            emph_probs = np.concatenate([x['emph_probs_b'] for x in outputs_i], axis=0)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=2, keepdims=True)
            preds = np.argmax(logits, axis=2)

            label_map = {i: label for i, label in enumerate(self.labels)}
            out_label_list = [[] for _ in range(labels.shape[0])]
            preds_list = [[] for _ in range(labels.shape[0])]
            probs_list = [[] for _ in range(labels.shape[0])]
            emph_probs_list = [[] for _ in range(labels.shape[0])]

            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i, j] != self.pad_token_label_id:
                        out_label_list[i].append(label_map[labels[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        probs_list[i].append(probs[i][j][1])
                        emph_probs_list[i].append(emph_probs[i][j])

            # for validation
            if out_idx == 0:
                # show random example
                rand_idx = np.random.randint(0, len(out_label_list))
                logger.info('True: %s', out_label_list[rand_idx])
                logger.info('Pred: %s', preds_list[rand_idx])
                logger.info('Emph: %s', emph_probs_list[rand_idx])
                logger.info('Prob: %s', probs_list[rand_idx])

                # validation score
                val_score = match_m(probs_list, emph_probs_list)
                val_score_mean = np.mean(list(val_score.values()))

                # validation accuracy
                val_acc = accuracy_score(out_label_list, preds_list)

                # validation f1 score
                val_f1 = f1_score(out_label_list, preds_list)

                # score_logs
                score_logs = {
                    'val_score1': val_score[1],
                    'val_score2': val_score[2],
                    'val_score3': val_score[3],
                    'val_score4': val_score[4],
                    'val_score_mean': val_score_mean,
                }

                # logs
                logs = {
                    'val_loss': val_loss_mean.detach().numpy(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_score_mean': val_score_mean,
                }

                # output dict
                output = {
                    'val_loss': val_loss_mean,
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'progress_bar': logs,
                    'log': {**logs, **score_logs},
                }

                # write validation results to csv
                csv_dict = {k: v for k, v in vars(self.hparams).items() if k in FIELDNAMES}
                csv_dict.update(score_logs)
                csv_dict.update(logs)

                csv_dict['current_epoch'] = self.current_epoch
                csv_dict['global_step'] = self.global_step

                write_csv('results.csv', csv_dict)
                self.preds['val'][self.current_epoch][self.global_step] = probs_list

            # save prediction for test
            if out_idx == 1:
                self.preds['test'][self.current_epoch][self.global_step] = probs_list

        # write to file all val and test predictions
        with open('predicts/' + self.model_id_name + '.pkl', 'wb') as f:
            pickle.dump(self.preds, f)

        return output

    def training_epoch_end(self):
        print('Done', self.global_step)

    ################################################################################
    # Data processing related methods

    def _feature_file(self, mode, label_mode):
        """
        Get filename of cached file.
        """
        return os.path.join(
            self.hparams.data_cache_dir,
            'cached_{}_{}_{}_{}'.format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split('/'))).pop(),
                str(self.hparams.max_seq_length),
                label_mode,
            ),
        )

    def prepare_data(self):
        """
        Called to initialize data. Use the call to construct features
        """
        if not os.path.exists(self.hparams.data_cache_dir):
            os.mkdir(self.hparams.data_cache_dir)
        for mode, filepath in zip(['train', 'val', 'test'],
                                  [self.hparams.train_path, self.hparams.val_path, self.hparams.test_path]):
            if mode == 'train':
                label_mode = self.hparams.label_mode
            else:
                label_mode = 'major'
            cached_features_file = self._feature_file(mode, label_mode)

            if not os.path.exists(cached_features_file):
                logger.info('Creating features from dataset file at %s', filepath)
                examples = read_examples_from_file(filepath, mode, label_mode)
                features = convert_examples_to_features(
                    examples,
                    self.labels,
                    self.hparams.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.hparams.model_type in ['xlnet']),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.hparams.model_type in ['xlnet'] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=bool(self.hparams.model_type in ['roberta']),
                    pad_on_left=bool(self.hparams.model_type in ['xlnet']),
                    pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if self.hparams.model_type in ['xlnet'] else 0,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info('Saving features into cached file %s', cached_features_file)
                torch.save(features, cached_features_file)

    def get_dataloader(self, mode, label_mode, batch_size):
        """
        Load datasets. Called after prepare data.
        """
        cached_features_file = self._feature_file(mode, label_mode)
        logger.info('Loading features from cached file %s', cached_features_file)
        features = torch.load(cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_emph_probs = torch.tensor([f.emph_probs for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_emph_probs)

        if mode == 'train':
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        return dataloader

    def get_total_train_steps(self):
        dataloader = self.get_dataloader(
            mode='train',
            label_mode=self.hparams.label_mode,
            batch_size=self.hparams.train_batch_size
        )
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        return t_total

    def train_dataloader(self):
        dataloader = self.get_dataloader(
            mode='train',
            label_mode=self.hparams.label_mode,
            batch_size=self.hparams.train_batch_size
        )
        return dataloader

    def val_dataloader(self):
        dataloader_val = self.get_dataloader(
            mode='val',
            label_mode='major',  # for val and test always major mode (without separating by annotators)
            batch_size=self.hparams.eval_batch_size
        )
        dataloader_test = self.get_dataloader(
            mode='test',
            label_mode='major',  # for val and test always major mode (without separating by annotators)
            batch_size=self.hparams.eval_batch_size
        )
        return [dataloader_val, dataloader_test]

    # def test_dataloader(self):
    #     dataloader = self.get_dataloader(
    #         mode='test',
    #         label_mode='major',  # for val and test always major mode (without separating by annotators)
    #         batch_size=self.hparams.eval_batch_size
    #     )
    #     return dataloader

    # def predict(self, batch):
    #     """
    #     Return prediction and probabilities
    #     """
    #     # batch
    #     self.model.eval()
    #     input_ids, attention_mask, token_type_ids, _ = batch

    #     # forward
    #     _, logits = self.forward(input_ids, attention_mask, token_type_ids)
    #     _, y_hat = torch.max(logits, dim=1)

    #     # probs
    #     probs = nn.Softmax(dim=1)(logits.view(-1, self.model.num_labels))

    #     return y_hat, probs
