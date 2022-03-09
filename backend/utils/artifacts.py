import os
import pathlib
import glob
import logging

import wandb


def download_index_artifact(artifact_name: str) -> str:
    artifact_path = download_artifact(artifact_name)

    # Extract index path
    index_paths = glob.glob(os.path.join(artifact_path, "*.index"))
    assert len(index_paths) == 1, "artifact should have only one index..."
    index_path = index_paths[0]

    return index_path


def download_model_artifact(artifact_name: str) -> str:
    artifact_path = download_artifact(artifact_name)

    # Extract model path
    model_paths = glob.glob(os.path.join(artifact_path, "*.pt"))
    assert len(model_paths) == 1, "artifact should have only one model checkpoint..."
    model_path = model_paths[0]

    return model_path


def download_artifact(artifact_name: str) -> str:
    # Create artifacts dir if it doesn't already exist
    artifacts_dir = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "artifacts"
    )
    if not os.path.exists(artifacts_dir):
        os.mkdir(artifacts_dir)

    # Setup wandb, parse model name, and generate a path to save the artifact
    api = wandb.Api()
    artifact = api.artifact(
        f"cs329s/cs329s/{artifact_name}", type="distilbert-embedding-model"
    )
    model_name = artifact_name
    artifact_path = os.path.join(artifacts_dir, model_name)

    # Download model path if it doesn't exist
    if not os.path.exists(artifact_path):
        logging.info(f"Downloading artifact: '{artifact_name}' to '{artifact_path}'")
        artifact.download(artifact_path)
    else:
        logging.info(f"Using cached artifact: '{artifact_name}' at '{artifact_path}'")

    return artifact_path
