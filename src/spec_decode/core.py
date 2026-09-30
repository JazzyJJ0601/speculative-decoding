import numpy as np


class DraftModel:
    """Draft model using n-gram statistics to propose tokens."""

    def __init__(self, ngram_order=2, vocab_size=100):
        self.ngram_order = ngram_order
        self.vocab_size = vocab_size
        self._history = []
        self._counts = {}

    def train(self, tokens):
        """Build n-gram statistics from a sequence of tokens."""
        for i in range(len(tokens) - self.ngram_order):
            ctx = tuple(tokens[i : i + self.ngram_order])
            nxt = tokens[i + self.ngram_order]
            self._counts[ctx] = self._counts.get(ctx, {})
            self._counts[ctx][nxt] = self._counts[ctx].get(nxt, 0) + 1

    def propose(self, num_candidates=5):
        """Propose candidate tokens using n-gram predictor."""
        if len(self._history) < self.ngram_order:
            return np.random.randint(0, self.vocab_size, size=num_candidates)
        ctx = tuple(self._history[-self.ngram_order:])
        probs = self._counts.get(ctx, {})
        if probs:
            candidates = []
            for _ in range(num_candidates):
                token = np.random.choice(list(probs.keys()), p=[probs[t] / sum(probs.values()) for t in probs])
                candidates.append(token)
            out = np.array(candidates)
        else:
            out = np.random.randint(0, self.vocab_size, size=num_candidates)
        self._history.extend(out)
        return out


class TargetModel:
    """Target model interface that verifies candidates via logit comparison."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self._data = np.random.randn(1000, vocab_size)  # synthetic logits bank

    def get_logits(self, context_size=1):
        """Return logits for verification."""
        return self._data[:context_size]


def rejection_sampling(draft_model, target_model, max_steps=10):
    """
    Perform rejection sampling: draft proposes tokens, target verifies them.
    Returns accepted tokens.
    """
    accepted = []
    for _ in range(max_steps):
        candidates = draft_model.propose(num_candidates=1)
        if candidates.size == 0:
            continue
        token = int(candidates[0])
        logits = target_model.get_logits(1)
        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
        draft_q = 1.0 / draft_model.vocab_size
        target_p = probs[token]
        alpha = min(1.0, target_p / draft_q)
        if np.random.rand() < alpha:
            accepted.append(token)
    return np.array(accepted)
