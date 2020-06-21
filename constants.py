#!/usr/bin/env python
# -*- coding: utf-8 -*-

SNLI_DATASET_URL = {
    "train": "https://cs-datasets-metamehta.s3.ap-south-1.amazonaws.com/snli_1.0_train.csv",
    "validation": "https://cs-datasets-metamehta.s3.ap-south-1.amazonaws.com/snli_1.0_dev.csv",
    "test": "https://cs-datasets-metamehta.s3.ap-south-1.amazonaws.com/snli_1.0_test.csv" 
}

SNLI_FILE_NAMES = {
    "train": "snli_1.0_train.csv",
    "validation": "snli_1.0_dev.csv",
    "test": "snli_1.0_test.csv"
}

BATCH_SIZE = 32

DATASET_LABELS = {
    'contradiction' : 0,
    'entailment' : 1, 
    'neutral': 2
}

PREPROCESSED_FOLDER = "preprocessed_data"

SPECIAL_TOKENS = {
    'CLS': ['[CLS]'],
    'SEP': ['[SEP]'] 
}