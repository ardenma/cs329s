import torch
import logging

logging.getLogger().setLevel(logging.DEBUG)


def contrastive_loss(
    embeddings: torch.tensor,
    labels: torch.tensor,
    num_labels: int,
    same_label_multiplier: int = 2,
):
    batch_size = len(labels)
    max_label_distance = num_labels - 1

    # B X EMB_SIZE
    normalized_embs = torch.nn.functional.normalize(
        embeddings, p=2.0, dim=-1, eps=1e-12
    )
    logging.debug(f"normalized_embs: {normalized_embs}")

    # B X B
    similarity_matrix = normalized_embs @ normalized_embs.T
    logging.debug(f"similarity matrix: {similarity_matrix}")

    # B X B (upper right triangular), using label distances as "importance" weights
    label_distances = torch.zeros(batch_size, batch_size).to(normalized_embs)
    for row in range(batch_size):
        for col in range(row, batch_size):
            label_distances[row, col] = torch.abs(labels[row] - labels[col])

    # label_distances = max_label_distance - label_distances

    # If same label, want to maximize similarity so need to make contribution negative
    # Ignore self-similarity
    same_label_pair_count = 0
    for row in range(batch_size):
        for col in range(row, batch_size):
            if label_distances[row, col] == 0 and row != col:
                label_distances[row, col] = -same_label_multiplier  # arbitrary
                same_label_pair_count += 1

    logging.debug(f"label_distances: {label_distances}")

    # Cosine similarities weighted by their label distances
    similarity_matrix = similarity_matrix * label_distances
    logging.debug(f"label weighted similarity matrix: {similarity_matrix}")

    # Loss is sum of label weighted distances
    loss = torch.sum(similarity_matrix)
    logging.debug(f"loss: {loss}")

    # Perform max-min normalization on loss
    num_active = ((batch_size * batch_size) - batch_size) / 2.0
    diff_label_pair_count = num_active - same_label_pair_count
    max_loss = diff_label_pair_count * (max_label_distance)
    min_loss = -(same_label_pair_count * same_label_multiplier)
    normalized_loss = (loss - min_loss) / (max_loss - min_loss)
    logging.debug(
        f"max_loss: {max_loss}, min_loss: {min_loss}, loss (before): {loss}, loss (after): {normalized_loss}"
    )

    return normalized_loss
