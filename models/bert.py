from transformers import (AdamW, AdamWeightDecay,
                          BertForSequenceClassification,
                          get_linear_schedule_with_warmup)

train_dataloder = []
epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


class BERTModel:

    def __init__(self, train_dataloader, num_labels, model_size="base", cased=False,
                 output_attentions=False, output_hidden_states=False,
                 optimizer="AdamW", lr=2e-5, eps=1e-8, beta1=0.9, beta2=0.999,
                 weight_decay=0.0, correct_bias=True, epochs=4):
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

        self.train_dataloader = train_dataloader
        self.epochs = epochs

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
