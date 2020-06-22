"""Train the model"""

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import trange

from constants import (BATCH_PRINT_FREQ, BATCH_SIZE, DEFAULT_MODEL_PARAMS,
                       EPOCHS, GRAD_CLIP_VALUE, SAVED_MODEL_LOCATION,
                       SEED_VALUE)
from evaluate import flat_accuracy
from models.bert import BERTModel
from utils import LogUtils, format_time

LOG = LogUtils.setup_logger(__name__)


def train_and_evaluate_bert(train_dataloader, validation_dataloader, num_labels, use_gpu=True, model_size="base",
                            cased=False, output_attentions=False, output_hidden_states=False, optimizer="AdamW",
                            params=DEFAULT_MODEL_PARAMS, epochs=EPOCHS,
                            save_model=True, save_location=SAVED_MODEL_LOCATION):
    """
    Train the BERT Model

    Args:
        train_dataloder: (torch.utils.data.TensorDataset) Dataloder for the Training Set
        validation_dataloder: (torch.utils.data.TensorDataset) Dataloder for the Validation Set
        num_labels: (int) Number of output labels
        use_gpu: (bool) Use GPU if available for training
        model_size: (str) Size of BERT Model ["base", "large"]
        cased: (bool) Use cased or uncased BERT model
        output_attentions: (bool) Output attention values from BERT Model
        output_hidden_states: (bool) Output embeddings generated from BERT layers
        optimizer: (str) Name of the Optimizer ["AdamW", "AdamWeightDecay"]
        params: (dict) Hyperparameters for the model
        epochs: (int) Number of epochs to run the model.
        save_model: (bool) Save model after training is complete
        save_location: (str) Location to save model after training
    """

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        LOG.info(f"There are {torch.cuda.device_count()} GPU(s) available")
        LOG.info(f"Using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        LOG.info("Using CPU")

    BertModel = BERTModel(
        train_dataloader=train_dataloader,
        num_labels=num_labels,
        model_size=model_size,
        output_attentions=output_attentions,
        output_hidden_states=output_attentions,
        optimizer=optimizer,
        lr=params["learning_rate"],
        eps=params["epsilon"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        weight_decay=params["weight_decay"],
        correct_bias=params["correct_bias"],
        epochs=epochs
    )

    model = BERTModel.model()
    optimizer = BERTModel.optimizer()
    scheduler = BERTModel.scheduler()

    LOG.info("Model, Optimizer and Scheduler setup successfully!")

    model.cuda()

    seed_val = SEED_VALUE
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []
    metrics = []

    LOG.info("Training Started!")

    t = trange(epochs)
    for epoch in t:
        epMetric = {}
        LOG.info(f"==== Epoch {epoch + 1} / {epochs} ====")
        epMetric["epoch"] = epoch + 1
        start_time = time.time()
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % BATCH_PRINT_FREQ == 0 and not step == 0:
                elapsed = time.time() - start_time
                time_per_batch = elapsed / step
                time_remaining = (len(train_dataloader) -
                                  step) * time_per_batch
                LOG.info(
                    f"\nBatch {step} of {len(train_dataloader)}. Elapsed: {format_time(elapsed)}")
                LOG.info(
                    f"Time left in this epoch: {format_time(time_remaining)}")

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

            loss = output[0]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        p_time = time.time()
        time_taken_train = format_time(p_time - start_time)

        epMetric["time_taken_train"] = time_taken_train
        epMetric["epoch_avg_loss"] = avg_train_loss

        LOG.info(f"\n==== Average Training Loss: {avg_train_loss} ====")
        LOG.info(f"==== Training Epoch Time: {time_taken} ====\n")

        LOG.info("Validation time!")

        start_time = time.time()
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        p_time = time.time()
        time_taken_validation = format_time(p_time - start_time)
        validation_accuracy = eval_accuracy / nb_eval_steps

        epMetric["time_taken_validation"] = time_taken_validation
        epMetric["validation_accuracy"] = validation_accuracy

        LOG.info(f"==== Accuracy: {validation_accuracy} ====")
        LOG.info(f"==== Validation Took: {time_taken_validation}")

        metrics.append(epMetric)

    LOG.info("Training Complete!")

    output_dir = save_location
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    LOG.info(f"Saving model to {output_dir}")

    with open(os.path.join(output_dir, "model-metrics.json"), "w") as f:
        model_metrics = json.dumps(metrics)
        f.write(model_metrics)

    return (model, metrics)
