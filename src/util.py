from datasets import load_dataset
import torch

def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def load_liar_dataset(tokenizer):
    dataset = load_dataset("liar")

    # Drop unused columns from our dataset
    dataset = dataset.remove_columns([
        'id',
        'subject',
        'speaker',
        'job_title',
        'state_info',
        'party_affiliation',
        'barely_true_counts',
        'false_counts',
        'half_true_counts',
        'mostly_true_counts',
        'pants_on_fire_counts',
        'context'
    ])

    # Tokenize our text and truncate statements
    def preprocess_function(examples):
        return tokenizer(examples["statement"], truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    return tokenized_dataset

def compute_accuracy(predictions, reals):
    num_correct = 0

    for i in range(len(predictions)):
        pred = predictions[i]['label']

        # Predictions are of the form LABEL_0, LABEL_1, etc.
        # whereas reals are of the form 0, 1, etc. Here, we
        # trim the first part of the prediction to extract
        # just the number, so that it matches the same format
        # as the real.
        pred = int(pred.split("_", 1)[1])

        real = int(reals[i]['label'])

        if pred == real:
            num_correct += 1

    accuracy = num_correct / len(predictions) * 100
    return accuracy
