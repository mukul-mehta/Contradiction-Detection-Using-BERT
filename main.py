import argparse
import json
import os

import numpy as np
import torch
from transformers import BertTokenizer

from build_dataset import SNLIDataset
from evaluate import evaluate_model_on_test_set
from models.bert import BERTModel
from train import train_and_evaluate_bert
from utils import LogUtils, read_config

LOG = LogUtils.setup_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='data',
                    help="Directory to download dataset")
parser.add_argument('--preprocessed_dir', default='preprocessed_data',
                    help="Directory to save preprocessed datasets")
parser.add_argument('--model_dir', default="saved-model",
                    help="Directory to save model after training")

parser.add_argument('--config', default="config.ini",
                    help="INI file for model configuration")

parser.add_argument('--epochs', help="Number of epochs to train for")
parser.add_argument('--batch_size',
                    help="Batch Size for Training/Validation/Testing on dataset")

parser.add_argument('--download_dataset', action='store_true',
                    help="If set, download the dataset to a specified location")
parser.add_argument('--bert_size', default="base", choices=["base", "large"],
                    help="Size of pretrained BERT to use (base or large)")
parser.add_argument('--cased', default=False,
                    help="Use argument when you think casing is important")
parser.add_argument('--num_labels',
                    help="Number of labels to classify for given dataset")

parser.add_argument('--output_attentions', action='store_true',
                    help="Output attention values from BERT Model")
parser.add_argument('--output_hidden_states', action='store_true',
                    help="Output embeddings generated from BERT layers")
parser.add_argument('--gpu', action='store_true',
                    help="Use GPU for training if available")
parser.add_argument('--predict_on_test', action='store_true',
                    help="Load model from file and run on test set")
parser.add_argument('--no_train', action='store_true',
                    help="Don't train model, load model and quit")

if __name__ == "__main__":
    args = parser.parse_args()

    filename = args.config
    sections = ["hyperparameters", "locations", "dataset_labels", "misc"]

    hyperparameters = read_config(filename=filename, section="hyperparameters")

    EPOCHS = int(hyperparameters["epochs"]
                 ) if args.epochs is None else int(args.epochs)
    BATCH_SIZE = int(hyperparameters["batch_size"]) if args.batch_size is None else int(
        args.batch_size)
    MODEL_PARAMS = {
        'learning_rate': float(hyperparameters["learning_rate"]),
        'epsilon': float(hyperparameters["epsilon"]),
        'beta1': float(hyperparameters["beta1"]),
        'beta2': float(hyperparameters["beta2"]),
        'weight_decay': float(hyperparameters["weight_decay"]),
        'correct_bias': True if hyperparameters["correct_bias"] == "True" else False
    }

    locations = read_config(filename="config.ini", section="locations")

    DATASET_LOCATION = locations["dataset_location"] \
        if not args.data_dir else args.data_dir
    PREPROCESSED_LOCATION = locations["preprocessed_data_location"] \
        if not args.preprocessed_dir else args.preprocessed_dir
    SAVED_MODEL_LOCATION = locations["saved_model_location"] \
        if not args.model_dir else args.model_dir

    dataset_labels = read_config(
        filename="config.ini", section="dataset_labels")

    DATASET_LABELS = {}
    for k, v in dataset_labels.items():
        DATASET_LABELS[k] = int(v)

    misc = read_config(filename="config.ini", section="misc")

    GRAD_CLIP_VALUE = float(misc["grad_clip_value"])
    BATCH_PRINT_FREQ = int(misc["batch_print_freq"])
    MAX_LEN_TOKENS = int(misc["max_len_tokens"])
    SEED_VALUE = int(misc["seed_value"])

    DOWNLOAD_DATASET = False
    if args.download_dataset:
        DOWNLOAD_DATASET = True

    MODEL_SIZE = "base"
    if args.bert_size.lower() == "large":
        MODEL_SIZE = "large"

    CASED = "uncased"
    if args.cased:
        CASED = "cased"

    NUM_LABELS = len(DATASET_LABELS) if not args.num_labels else int(
        args.num_labels)

    OUTPUT_ATTENTIONS = args.output_attentions
    OUTPUT_HIDDEN_STATES = args.output_hidden_states

    USE_GPU = args.gpu

    RUN_PREDICTIONS = args.predict_on_test
    NO_TRAIN = args.no_train

    model_name = f"bert-{MODEL_SIZE}-{CASED}"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    dataset = SNLIDataset(
        tokenizer=tokenizer,
        data_folder=DATASET_LOCATION,
        preprocessed_folder=PREPROCESSED_LOCATION,
        batch_size=BATCH_SIZE,
        max_len_tokens=MAX_LEN_TOKENS,
        dataset_labels=DATASET_LABELS,
        download_dataset=DOWNLOAD_DATASET,
        use_padding=True
    )

    train_tokens, train_input_ids, train_token_lenghts, train_labels = dataset.preprocess_dataset(
        d_partition="train")
    validation_tokens, validation_input_ids, validation_token_lenghts, validation_labels = dataset.preprocess_dataset(
        d_partition="validation"
    )

    LOG.info("Padding inputs and creating attention masks")

    train_input_ids, train_attention_masks = dataset.pad_and_create_attention_masks(
        train_input_ids)
    validation_input_ids, validation_attention_masks = dataset.pad_and_create_attention_masks(
        validation_input_ids)

    LOG.info("Converting dataset to PyTorch TensorDataset")

    train_data, train_sampler, train_dataloader = dataset.convert_data_to_tensor_dataset(
        train_input_ids, train_attention_masks, train_labels)
    validation_data, validation_sampler, validation_dataloader = dataset.convert_data_to_tensor_dataset(
        validation_input_ids, validation_attention_masks, validation_labels)

    LOG.info("Loading BERT Model, optimizer and scheduler")

    BERTModel = BERTModel(
        train_dataloader=train_dataloader,
        num_labels=NUM_LABELS,
        model_size=MODEL_SIZE,
        cased=CASED,
        output_attentions=OUTPUT_ATTENTIONS,
        output_hidden_states=OUTPUT_HIDDEN_STATES,
        optimizer="AdamW",
        lr=MODEL_PARAMS["learning_rate"],
        eps=MODEL_PARAMS["epsilon"],
        beta1=MODEL_PARAMS["beta1"],
        beta2=MODEL_PARAMS["beta2"],
        weight_decay=MODEL_PARAMS["weight_decay"],
        correct_bias=MODEL_PARAMS["correct_bias"],
        epochs=EPOCHS
    )

    if NO_TRAIN:
        LOG.info("Bye")
        exit()

    model, metrics = train_and_evaluate_bert(
        BERTModel=BERTModel,
        validation_dataloader=validation_dataloader,
        use_gpu=USE_GPU,
        seed_value=SEED_VALUE,
        batch_print_freq=BATCH_PRINT_FREQ,
        grad_clip_value=GRAD_CLIP_VALUE
    )

    output_dir = SAVED_MODEL_LOCATION

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    LOG.info(f"Saving model to {output_dir}")

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "model-metrics.json"), "w") as f:
        model_metrics = json.dumps(metrics)
        f.write(model_metrics)

    if not RUN_PREDICTIONS:
        LOG.info("Model saved successfully! Quitting")
        exit()

    test_tokens, test_input_ids, test_labels, test_token_lenghts = dataset.preprocess_dataset(
        d_partition="test")

    test_input_ids, test_attention_masks = dataset.pad_and_create_attention_masks(
        test_input_ids)

    test_data, test_sampler, test_dataloader = dataset.convert_data_to_tensor_dataset(
        test_input_ids, test_attention_masks, test_labels)

    evaluations = evaluate_model_on_test_set(
        prediction_dataloader=test_dataloader,
        dataset_labels=DATASET_LABELS,
        saved_model_location=SAVED_MODEL_LOCATION,
        use_gpu=USE_GPU,
        batch_print_freq=BATCH_PRINT_FREQ
    )

    with open(SAVED_MODEL_LOCATION + "model-evaluation.json", "w") as f:
        output = json.dumps(evaluations)
        f.write(output)
