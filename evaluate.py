import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from transformers import BertForSequenceClassification, BertTokenizer

from constants import BATCH_PRINT_FREQ, SAVED_MODEL_LOCATION
from utils import LogUtils, format_time

LOG = LogUtils.setup_logger(__name__)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_model_from_file(saved_model_location=SAVED_MODEL_LOCATION):

    model_weight_dir = saved_model_location

    if not os.path.exists(model_weight_dir):
        raise Exception(f"Directory {saved_model_location} doesn't exist")

    model = BertForSequenceClassification.from_pretrained(
        model_weight_dir,
        output_hidden_states=True
    )

    tokenizer = BertTokenizer.from_pretrained(model_weight_dir)

    return (model, tokenizer)


def run_model_on_test_set(prediction_dataloader, saved_model_location=SAVED_MODEL_LOCATION, use_gpu=False):

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        LOG.info(f"There are {torch.cuda.device_count()} GPU(s) available")
        LOG.info(f"Using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        LOG.info("Using CPU")

    model, _ = load_model_from_file(saved_model_location)
    model.cuda()

    predictions, true_labels = [], []
    start_time = time.time()

    for idx, batch in enumerate(prediction_dataloader):
        if idx % BATCH_PRINT_FREQ == 0 and idx != 0:
            LOG.info(f"Done with batch {i} of {len(prediction_dataloader)}")
            elapsed = time.time() - start_time
            time_per_batch = elapsed / idx
            time_remaining = (len(prediction_dataloader) -
                              idx) * time_per_batch
            LOG.info(
                f"\nBatch {idx} of {len(prediction_dataloader)}. Elapsed: {format_time(elapsed)}")
            LOG.info(
                f"Time left: {format_time(time_remaining)}")

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        LOG.info('Done with Predictions!')

        return (predictions, true_labels)


def evaluate_model_on_test_set(prediction_dataloader, num_labels,
                               saved_model_location=SAVED_MODEL_LOCATION, use_gpu=False
                               ):

    predictions, true_labels = run_model_on_test_set(
        prediction_dataloader=prediction_dataloader,
        saved_model_location=saved_model_location,
        use_gpu=use_gpu
    )
    pred_labels = np.argmax(predictions, axis=1)

    exact_match_score = accuracy_score(true_labels, pred_labels)
    hamming_score = hamming_loss(true_labels, pred_labels)

    labels = [0, 1, 2]
    target_names = ["contradiction", "entailment", "neutral"]

    prec_recall_report = classification_report(
        true_labels, pred_labels, labels=labels, target_names=target_names, output_dict=True)

    evaluation = {}
    evaluation["exact_march_score"] = exact_match_score
    evaluation["hamming_score"] = hamming_score
    evaluation["prec_recall_report"] = prec_recall_report

    with open(saved_model_location + "model-evaluation.json", "w") as f:
        output = json.dumps(evaluation)
        f.write(output)

    return evaluation

