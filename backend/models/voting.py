from typing import List
from collections import Counter

import numpy as np


class MajorityVoter:
    def __call__(self, batch_votes: List[List[int]]) -> List[int]:
        return [self.vote_func(votes) for votes in batch_votes]

    def vote_func(self, votes: List[int]) -> int:
        vote_counter = Counter(votes)
        labels = list(vote_counter.keys())
        vote_counts = list(vote_counter.values())
        max_votes = max(vote_counts)

        # Randomly break ties
        candidates = []
        for i in range(len(vote_counts)):
            if vote_counts[i] == max_votes:
                candidates.append(labels[i])

        return np.random.choice(candidates)


class WeightedMajorityVoter:
    def __call__(
        self, batch_votes: List[List[int]], batch_similarities: np.array
    ) -> List[int]:
        return [
            self.vote_func(votes, similarities)
            for votes, similarities in zip(batch_votes, batch_similarities)
        ]

    def vote_func(self, votes: List[int], similarities: np.array) -> int:
        scores = {}
        for vote, similarity in zip(votes, similarities):
            if vote not in scores:
                scores[vote] = similarity
            else:
                scores[vote] += similarity

        return max(scores, key=scores.get)
