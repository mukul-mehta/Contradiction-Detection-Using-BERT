#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import torch
import wget
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from constants import (BATCH_SIZE, DATASET_LABELS, MAX_LEN,
                       PREPROCESSED_FOLDER, SNLI_DATASET_URL, SNLI_FILE_NAMES,
                       SPECIAL_TOKENS)
from utils import LogUtils

LOG = LogUtils.setup_logger(__name__)


class SNLIDataset(object):
    """
    Class to handle datasets and
    preprocess them in order to pass them on
    to the model for training/testing
    """

    def __init__(
        self, tokenizer, data_folder="./data", batch_size=BATCH_SIZE, use_padding=True,
        dataset_labels=DATASET_LABELS, download_dataset=False
    ):

        self.data_folder = data_folder
        self.batch_size = batch_size
        self.use_padding = use_padding
        self.preproccesed_folder = PREPROCESSED_FOLDER
        self.tokenizer = tokenizer

        self.dataset_labels = dataset_labels

        if download_dataset:
            self.download_dataset()

        train_data = os.path.join(self.data_folder, SNLI_FILE_NAMES["train"])
        validation_data = os.path.join(
            self.data_folder, SNLI_FILE_NAMES["validation"])
        test_data = os.path.join(self.data_folder, SNLI_FILE_NAMES["test"])

        self.raw_train_data = pd.read_csv(train_data, index_col=1)
        self.raw_validation_data = pd.read_csv(validation_data, index_col=1)
        self.raw_test_data = pd.read_csv(test_data, index_col=1)

    def download_dataset(self):

        LOG.info("Downloading SNLI Dataset in CSV format")

        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        download_location = self.data_folder

        train_data_url = SNLI_DATASET_URL["train"]
        validation_data_url = SNLI_DATASET_URL["validation"]
        test_data_url = SNLI_DATASET_URL["test"]

        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["train"])):
            LOG.info("Test Data already present")
        else:
            train_data = wget.download(train_data_url, out=download_location)

        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["validation"])):
            LOG.info("Validation Data already present")
        else:
            validation_data = wget.download(
                validation_data_url, out=download_location)

        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["test"])):
            LOG.info("Test Data already present")
        else:
            test_data = wget.download(test_data_url, out=download_location)

        LOG.info(f"Dataset downloaded successfully in {download_location}")
        return

    def preprocess_dataset_util(self, dataset_df):

        tokenizer = self.tokenizer
        labels_dict = self.dataset_labels

        def tokenize_sentence(tokenizer, input_sentence):
            return tokenizer.tokenize(input_sentence)

        sentence_A = dataset_df.sentence1.to_numpy()
        sentence_B = dataset_df.sentence2.to_numpy()
        labels = dataset_df.gold_label.to_numpy()

        sentence_A_tokens = []
        sentence_B_tokens = []
        processed_labels = []

        for i, j, k in zip(sentence_A, sentence_B, labels):
            try:
                if k == '-':
                    continue

                t1 = tokenize_sentence(tokenizer, i)
                t2 = tokenize_sentence(tokenizer, j)

                sentence_A_tokens.append(t1)
                sentence_B_tokens.append(t2)

                label = labels_dict[k]
                processed_labels.append(label)

            except Exception as e:
                LOG.error(e, exc_info=True)
                continue

        sentence_tokens = []
        input_ids = []
        token_lengths = []

        CLS_TOKEN = SPECIAL_TOKENS['CLS']
        SEP_TOKEN = SPECIAL_TOKENS['SEP']

        for i, j in zip(sentence_A_tokens, sentence_B_tokens):
            sentence = CLS_TOKEN + i + SEP_TOKEN + j + SEP_TOKEN
            token_ids = tokenizer.convert_tokens_to_ids(sentence)

            sentence_tokens.append(sentence)
            token_lengths.append(len(token_ids))
            input_ids.append(token_ids)

        return np.array(sentence_tokens), np.array(input_ids), np.array(token_lengths), np.array(processed_labels)

    def preprocess_dataset(self, d_partition="train"):

        if d_partition.lower() not in ["train", "dev", "validation", "test"]:
            raise BaseException(
                "d_partition must be train, dev, validation or test")

        if not os.path.exists(self.preproccesed_folder):
            os.mkdir(self.preproccesed_folder)

        preprocessed_location = self.preproccesed_folder

        file_name_base = os.path.join(preprocessed_location, d_partition + "_")
        if os.path.exists(file_name_base + "tokens.npy"):
            LOG.info("Retrieving tokens from .npy files!")
            tokens = np.load(file_name_base + "tokens.npy", allow_pickle=True)
            ids = np.load(file_name_base + "token-ids.npy", allow_pickle=True)
            lengths = np.load(
                file_name_base + "token-lengths.npy", allow_pickle=True)
            labels = np.load(file_name_base + "labels.npy", allow_pickle=True)

        else:
            if d_partition.lower() == "train":
                dataset_df = self.raw_train_data
            elif d_partition.lower() in ["dev", "validation"]:
                dataset_df = self.raw_validation_data
            else:
                dataset_df = self.raw_test_data

            tokens, ids, lengths, labels = self.preprocess_dataset_util(
                dataset_df)

            np.save(file_name_base + "tokens.npy", tokens)
            np.save(file_name_base + "token-ids.npy", ids)
            np.save(file_name_base + "token-lengths.npy", lengths)
            np.save(file_name_base + "labels.npy", labels)

        return (tokens, ids, lengths, labels)

    def pad_and_create_attention_masks(self, input_ids):
        if self.use_padding:
            input_ids = pad_sequences(
                input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

        attention_masks = []
        for sentence in input_ids:
            attention_mask = [int(token_id > 0) for token_id in sentence]
            attention_masks.append(attention_mask)

        return input_ids, attention_masks

    def convert_data_to_tensor_dataset(self, tokens, attention_masks, labels):
        batch_size = self.batch_size

        tokens = torch.tensor(tokens)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels)

        data = TensorDataset(tokens, attention_masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return (data, sampler, dataloader)
