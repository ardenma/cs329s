"""
script: train.py

Trains the MisinformationClassifier model on the LIAR dataset.
Saves the trained model's weights. During runtime later,
we will load these weights and use them so that we don't have to
re-train the model every time we run.
"""

import os
import pathlib

from model import MisinformationClassifier
from util import load_liar_dataset

SAVED_MODELS_DIR_NAME = "saved_models"

# Get path of directory to save model parameters in.
# Create the directory if it does not exist already.
cwd = pathlib.Path(__file__).parent.resolve()
saved_models_path = os.path.join(cwd, SAVED_MODELS_DIR_NAME)
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)

classifier = MisinformationClassifier()

dataset = load_liar_dataset(classifier.tokenizer)
classifier.train(dataset)

classifier.save_pretrained(saved_models_dir)
