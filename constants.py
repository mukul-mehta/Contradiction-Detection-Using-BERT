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

SPECIAL_TOKENS = {
    'CLS': ['[CLS]'],
    'SEP': ['[SEP]']
}
