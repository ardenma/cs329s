import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
    Trainer,
    TrainingArguments
)

from util import get_device

class MisinformationClassifier():
    BASE_MODEL = "distilbert-base-uncased"
    NUM_CLASSES = 6

    tokenizer = None
    model = None
    classifier = None

    def __init__(self):
        """
        Creates and returns a text classifier (which is trained in the train()
        function to classify text as one of six labels: pants-fire, false,
        barely-true, half-true, mostly-true, and true). The labels are taken
        from the LIAR dataset and are unchanged.
        (https://arxiv.org/pdf/1705.00648v1.pdf)

        Returns: classifier: https://huggingface.co/docs/transformers/main_classes/pipelines
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.BASE_MODEL,
            num_labels=self.NUM_CLASSES
        )
        self.model = self.model.to(get_device())

        self.classifier = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if get_device() == "cuda:0" else -1
        )

    LEARNING_RATE = 2e-5
    LEARNING_RATE_DECAY = 0.01
    TRAINING_BATCH_SIZE = 16
    EVALUATION_BATCH_SIZE = 16
    NUM_EPOCHS = 5

    def train(self, dataset):
        # Used during training to batch examples.
        # Also dynamically pads text to the length of the longest element in each
        # batch, so that they are of uniform length. This implementation is more
        # efficient than setting padding=True in the tokenizer function,
        # as per the Hugging Face documentation (https://huggingface.co/docs/transformers/tasks/sequence_classification)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=self.LEARNING_RATE,
            per_device_train_batch_size=self.TRAINING_BATCH_SIZE,
            per_device_eval_batch_size=self.EVALUATION_BATCH_SIZE,
            num_train_epochs=self.NUM_EPOCHS,
            weight_decay=self.LEARNING_RATE_DECAY
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
