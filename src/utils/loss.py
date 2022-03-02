import torch
import logging
logging.getLogger().setLevel(logging.DEBUG)

def contrastive_loss(embeddings: torch.tensor, labels: torch.tensor, num_labels: int, same_label_multiplier: int=2):
    batch_size = len(labels)
    max_label_distance = num_labels - 1

    # B X EMB_SIZE
    normalized_embs = torch.nn.functional.normalize(embeddings, p=2.0, dim=-1, eps=1e-12)
    logging.debug(f"normalized_embs: {normalized_embs}")

    # B X B
    similarity_matrix = normalized_embs @ normalized_embs.T
    logging.debug(f"similarity matrix: {similarity_matrix}")

    # B X B (upper right triangular)
    label_distances = torch.zeros(batch_size, batch_size).to(normalized_embs)

    for row in range(batch_size):
        for col in range(row, batch_size):
            label_distances[row, col] = torch.abs(labels[row] - labels[col])
    
    label_distances = label_distances * -1
    label_distances += max_label_distance

    # Make 0 distance elements large (since want same label embeddiungs to have small dist)
    for row in range(batch_size):
        for col in range(row, batch_size):
            if row != col and label_distances[row, col] == 0:
                label_distances[row, col] = same_label_multiplier  # arbitrary

    logging.debug(f"label_distances: {label_distances}")

    # Cosine similarities weighted by their label distances
    similarity_matrix = similarity_matrix * label_distances
    logging.debug(f"label weighted similarity matrix: {similarity_matrix}")

    # Loss is sum of label weighted distances
    loss = torch.sum(similarity_matrix) 
    logging.debug(f"loss: {loss}")

    # Normalize to make loss closer to 1
    normalizer = (((batch_size * batch_size) - batch_size) / 2.0) * (max_label_distance)
    same_label_normalizer_offset = batch_size * same_label_multiplier
    normalized_loss = loss / (normalizer + same_label_normalizer_offset)
    logging.debug(f"normalizer: {normalizer}, same_label_normalizer: {same_label_normalizer_offset}, loss (before): {loss}, loss (after): {normalized_loss}")
    
    return normalized_loss