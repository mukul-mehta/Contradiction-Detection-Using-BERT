#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import wget
from constants import (SNLI_DATASET_URL,
                    SNLI_FILE_NAMES,
                    BATCH_SIZE,
                    DATASET_LABELS,
                    PREPROCESSED_FOLDER,
                    SPECIAL_TOKENS)


class SNLIDataset(object):
    """
    Class to handle datasets and 
    preprocess them in order to pass them on 
    to the model for training/testing
    """

    def __init__(self, tokenizer, data_folder = "./data", batch_size = BATCH_SIZE, use_padding = True, dataset_labels = DATASET_LABELS, download_dataset = False):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.use_padding = use_padding
        self.preproccesed_folder = PREPROCESSED_FOLDER
        self.tokenizer = tokenizer

        self.dataset_labels = dataset_labels
        
        if download_dataset:
            self.download_dataset()
        
        train_data = os.path.join(self.data_folder, SNLI_FILE_NAMES["train"])
        validation_data = os.path.join(self.data_folder, SNLI_FILE_NAMES["validation"])
        test_data = os.path.join(self.data_folder, SNLI_FILE_NAMES["test"])

        self.raw_train_data = pd.read_csv(train_data, index_col = 1)
        self.raw_validation_data = pd.read_csv(validation_data, index_col = 1)
        self.raw_test_data = pd.read_csv(test_data, index_col = 1)

    def download_dataset(self):
        
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        download_location = self.data_folder
    
        train_data_url = SNLI_DATASET_URL["train"]
        validation_data_url = SNLI_DATASET_URL["validation"]
        test_data_url = SNLI_DATASET_URL["test"]

        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["train"])):
            print("Train Data Found!")
        else:
            train_data = wget.download(train_data_url, out = download_location)
        
        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["validation"])):
            print("Dev Data Found!")
        else:
            validation_data = wget.download(validation_data_url, out = download_location)
        
        if os.path.exists(os.path.join(download_location, SNLI_FILE_NAMES["test"])):
            print("Test Data Found!")
        else:
            test_data = wget.download(test_data_url, out = download_location)
        
        print("\nDownload Data successfully!")
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
                print(e)
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

        return np.array(sentence_tokens, dtype=object), np.array(input_ids, dtype=object), np.array(token_lengths, dtype=object)

    def preprocess_dataset(self, d_partition = "train"):

        if d_partition.lower() not in ["train", "dev", "validation", "test"]:
            raise BaseException("d_partition must be train, dev, validation or test")

        if not os.path.exists(self.preproccesed_folder):
            os.mkdir(self.preproccesed_folder)
        
        preprocessed_location = self.preproccesed_folder

        file_name_base = os.path.join(preprocessed_location, d_partition + "_")
        if os.path.exists(file_name_base + "tokens.npy"):
            print("Retrieving NPY files!")
            tokens = np.load(file_name_base + "tokens.npy", allow_pickle = True)
            ids = np.load(file_name_base + "token-ids.npy", allow_pickle = True)
            lengths = np.load(file_name_base + "token-lengths.npy", allow_pickle = True)

        else:
            if d_partition.lower() == "train":
                dataset_df = self.raw_train_data
            elif d_partition.lower() in ["dev", "validation"]:
                dataset_df = self.raw_validation_data
            else:
                dataset_df = self.raw_test_data
            
            tokens, ids, lengths = self.preprocess_dataset_util(dataset_df)
            
            np.save(file_name_base + "tokens.npy", tokens)
            np.save(file_name_base + "token-ids.npy", ids)
            np.save(file_name_base + "token-lengths.npy", lengths)
        
        return (tokens, ids, lengths)