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
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

PREPROCESSED_FOLDER = "preprocessed_data"

SPECIAL_TOKENS = {
    'CLS': ['[CLS]'],
    'SEP': ['[SEP]']
}

MAX_LEN = 128
EPOCHS = 4

SEED_VALUE = 42

DEFAULT_MODEL_PARAMS = {
    'learning_rate': 2e-5,
    'epsilon': 1e-8,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.0,
    'correct_bias': True
}

BATCH_PRINT_FREQ = 500

GRAD_CLIP_VALUE = 1.0

SAVED_MODEL_LOCATION = "./saved-model"
