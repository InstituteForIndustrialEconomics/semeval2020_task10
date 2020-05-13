# Heavily based on from https://github.com/huggingface/transformers/blob/master/examples/ner/utils_ner.py

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Union
import logging
import os
import numpy as np
import csv


logger = logging.getLogger(__name__)


def get_df(filepath: str, is_test: bool = False) -> pd.DataFrame:
    """
    Reads data from given file

    Returns:
        pd.Dataframe
    """
    col_names = ['word_ids', 'words', 'BIO_annotations', 'BIO_frequencies', 'emphasis_probabilities', 'pos_tags']
    usecols = list(range(6))  # set number of columns because of tabs at line ends
    if is_test:
        col_names = col_names[:2]
        usecols = list(range(2))

    df = pd.read_csv(filepath, sep='\t', header=None, names=col_names, usecols=usecols, na_filter=False)
    df['sent_id'] = df['word_ids'].apply(lambda x: x.rsplit('_', 1)[0])

    # group by sentences
    df = pd.DataFrame([df.groupby('sent_id', sort=False)[col].apply(list) for col in col_names]).T.reset_index()

    # full sentence as string
    df['sentence'] = df['words'].apply(lambda x: ' '.join(x))

    # sentence source
    df['sent_source'] = df['sent_id'].apply(lambda x: x.split('_', 1)[0])

    return df


def get_separate_annotations(ann_lst: list, map_dict: Union[None, dict] = {'B': 'I'}) -> list:
    """
    Separate annotations by different annotators.
    Optionally map tags to given values.
    """
    output = []
    for annotator_idx in range(9):
        annotator_output = []
        for word_anns in ann_lst:
            annotator_tag = word_anns.split('|')[annotator_idx]
            # map tags, by default B -> I
            if map_dict:
                annotator_tag = map_dict.get(annotator_tag, annotator_tag)
            annotator_output.append(annotator_tag)
        output.append(annotator_output)
    return output


def get_major_vote_annotation(ann_lst: list, map_dict: Union[None, dict] = {'B': 'I'}) -> list:
    """
    Return major vote annotation.
    """
    output = []
    for word_anns in ann_lst:
        word_tags = defaultdict(int)
        for annotator_tag in word_anns.split('|'):
            if map_dict:
                annotator_tag = map_dict.get(annotator_tag, annotator_tag)
            word_tags[annotator_tag] += 1
        # by default we map B -> I, so only two tags: I, O and there are always major vote for 9 annotators
        major_tag = max(word_tags, key=word_tags.get)
        output.append(major_tag)
    return output


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: list. The labels for each word of the sequence. This should be
        specified for train and val examples, but not for test examples.
        emph_probs: list. List of token emphasis probabilites
    """
    guid: str
    words: list
    labels: list
    emph_probs: list


@dataclass
class InputFeatures():
    """
    A single set of features of data.
    """
    input_ids: list
    input_mask: list
    segment_ids: list
    label_ids: list
    emph_probs: list


def read_examples_from_file(filepath: str, mode: str, label_mode: str):
    """
    Read and convert examples.

    Args:
        filepath
        mode:
            'train'
            'val'
            'test'
        label_mode:
            'separate'
            'major'
            'ann_N' (n in 0..8)
            'test'
    """

    if mode == 'test':
        is_test = True
        label_mode = 'test'
    else:
        is_test = False

    # read to df
    df = get_df(filepath, is_test=is_test)

    if mode != 'test':
        if label_mode == 'separate':
            # in this case labels are list of lists and for each sentence we have 9 different examples of labels
            df['labels'] = df['BIO_annotations'].apply(get_separate_annotations)
            # in this case separate probabilities are just 1s and 0s
            df['probabilities'] = df['BIO_annotations'].apply(
                lambda x: get_separate_annotations(x, map_dict={'B': 1, 'I': 1, 'O': 0})
            )
        elif label_mode == 'major':
            # just one list of major annotation
            df['labels'] = df['BIO_annotations'].apply(get_major_vote_annotation)
            df['probabilities'] = df['emphasis_probabilities']
        elif label_mode in [f'ann_{i}' for i in range(9)]:
            # just one list of specific annotator labels
            df['separate_annotations'] = df['BIO_annotations'].apply(get_separate_annotations)
            df['separate_probabilities'] = df['BIO_annotations'].apply(
                lambda x: get_separate_annotations(x, map_dict={'B': 1, 'I': 1, 'O': 0})
            )
            ann_idx = int(label_mode.split('_')[-1])
            df['labels'] = df['separate_annotations'].apply(lambda x: x[ann_idx])
            df['probabilities'] = df['separate_probabilities'].apply(lambda x: x[ann_idx])
        else:
            raise ValueError('Set proper label_mode')
    # fill with the same labels for test
    elif mode == 'test':
        df['labels'] = [['O'] * len(sent) for sent in df.words]
        df['probabilities'] = [[0] * len(sent) for sent in df.words]

    examples = []

    for id_, words, labels, probs in zip(df['sent_id'], df['words'], df['labels'], df['probabilities']):
        if label_mode == 'separate':
            for i, (labels_i, probs_i) in enumerate(zip(labels, probs)):
                example = InputExample(guid=f'{mode}-{id_}-{i}', words=words, labels=labels_i, emph_probs=probs_i)
                examples.append(example)
        else:
            example = InputExample(guid=f'{mode}-{id_}', words=words, labels=labels, emph_probs=probs)
            examples.append(example)

    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token='[CLS]',
    cls_token_segment_id=1,
    sep_token='[SEP]',
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,


):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        emph_probs = []
        for word, label, prob in zip(example.words, example.labels, example.emph_probs):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            emph_probs.extend([prob] + [prob] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            emph_probs = emph_probs[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        emph_probs += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            emph_probs += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            emph_probs += [0]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            emph_probs = [0] + emph_probs

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            emph_probs = ([0] * padding_length) + emph_probs
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            emph_probs += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(emph_probs) == max_seq_length

        if ex_index < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s', example.guid)
            logger.info('tokens: %s', ' '.join([str(x) for x in tokens]))
            logger.info('input_ids: %s', ' '.join([str(x) for x in input_ids]))
            logger.info('input_mask: %s', ' '.join([str(x) for x in input_mask]))
            logger.info('segment_ids: %s', ' '.join([str(x) for x in segment_ids]))
            logger.info('label_ids: %s', ' '.join([str(x) for x in label_ids]))
            logger.info('emph_probs: %s', ' '.join([str(x) for x in emph_probs]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                          label_ids=label_ids, emph_probs=emph_probs)
        )
    return features


# evaluation functions without logging from SemEval2020_Task10_Emphasis_Selection


def average(lst):
    return sum(lst) / float(len(lst))


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def match_m(all_scores, all_labels):
    """
    This function computes match_m.
    :param all_scores: submission scores
    :param all_labels: ground_truth labels
    :return: match_m dict
    """
    # print("[LOG] computing Match_m . . .")
    top_m = [1, 2, 3, 4]
    match_ms = {}
    for m in top_m:
        # print("[LOG] computing m={} in match_m".format(m))
        intersects_lst = []
        # ****************** computing scores:
        score_lst = []
        for s in all_scores:
            # the length of sentence needs to be more than m:
            if len(s) <= m:
                continue
            s = np.array(s)
            ind_score = np.argsort(s)[-m:]
            score_lst.append(ind_score.tolist())
        # ****************** computing labels:
        label_lst = []
        for l in all_labels:
            # the length of sentence needs to be more than m:
            if len(l) <= m:
                continue
            # if label list contains several top values with the same amount we consider them all
            h = m
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label.tolist())

        for i in range(len(score_lst)):
            # computing the intersection between scores and ground_truth labels:
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/float((min(m, len(score_lst[i])))))
        # taking average of intersects for the current m:
        match_ms[m] = average(intersects_lst)

    return match_ms


def write_csv(filepath, values_dict):
    """
    Write params and scores
    """
    if not os.path.exists(filepath):
        with open(filepath, 'a') as csvfile:
            score_writer = csv.DictWriter(csvfile, values_dict.keys())
            score_writer.writeheader()

    with open(filepath, 'a') as csvfile:
        score_writer = csv.DictWriter(csvfile, values_dict.keys())
        score_writer.writerow(values_dict)
