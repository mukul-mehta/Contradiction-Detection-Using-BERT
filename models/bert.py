from transformers import (AdamW, AdamWeightDecay,
                          BertForSequenceClassification,
                          get_linear_schedule_with_warmup)


class BERTModel:

    def __init__(self, train_dataloader, num_labels, model_size, cased,
                 output_attentions, output_hidden_states,
                 optimizer, lr, eps, beta1, beta2,
                 weight_decay, correct_bias, epochs):
        """
            Args:
            train_dataloder: (torch.utils.data.TensorDataset) Dataloder for the Training Set
            num_labels: (int) Number of output labels
            model_size: (str) Size of BERT Model ["base", "large"]
            cased: (bool) Use cased or uncased BERT model
            output_attentions: (bool) Output attention values from BERT Model
            output_hidden_states: (bool) Output embeddings generated from BERT layers
            optimizer: (str) Name of the Optimizer ["AdamW", "AdamWeightDecay"]
            lr: (float) Learning Rate for the optimizer
            eps: (float) Epsilon value for optimizer
            beta1: (float) Beta1 value for Adam optimizer
            beta2: (float) Beta2 value for Adam optimizer
            weight_decay: (float) Weight Decay value for Adam optimizer
            correct_bias: (float) Correct for bias terms in Adam Optimizer, default = True
            epochs: (int) Number of epochs to run the model.
        """

        self.num_labels = num_labels
        self.model_size = model_size
        self.cased = cased
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.correct_bias = correct_bias

        self._train_dataloader = train_dataloader
        self._epochs = epochs

        if model_size.lower() == "base":
            if cased:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-base-cased',
                    num_labels=num_labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
            else:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=num_labels,
                    output_hidden_states=output_attentions,
                    output_attentions=output_attentions
                )
        else:
            if cased:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-large-cased',
                    num_labels=num_labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
            else:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-large-uncased',
                    num_labels=num_labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )

        if optimizer == "AdamWeightDecay":
            optimizer = AdamWeightDecay(model.parameters(
            ), learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=eps, weight_decay_rate=weight_decay)
        else:
            optimizer = AdamW(model.parameters(), lr=lr, eps=eps, betas=(
                beta1, beta2), weight_decay=weight_decay, correct_bias=correct_bias)

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        self._optimizer = optimizer
        self._model = model
        self._scheduler = scheduler

    def model(self):
        return self._model

    def optimizer(self):
        return self._optimizer

    def scheduler(self):
        return self._scheduler

    def epochs(self):
        return self._epochs

    def train_dataloader(self):
        return self._train_dataloader
