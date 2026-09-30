import numpy as np
from src.spec_decode.core import DraftModel, TargetModel, rejection_sampling


def test_draft_model():
    model = DraftModel(vocab_size=50)
    tokens = model.propose(3)
    assert len(tokens) == 3
    assert np.all((tokens >= 0) & (tokens < 50))


def test_target_model():
    model = TargetModel(vocab_size=50)
    logits = model.get_logits(5)
    assert logits.shape == (5, 50)


def test_rejection_sampling():
    draft = DraftModel(vocab_size=50)
    target = TargetModel(vocab_size=50)
    # Seed for reproducibility
    np.random.seed(42)
    output = rejection_sampling(draft, target, max_steps=5)
    assert isinstance(output, np.ndarray)
    assert output.shape[0] <= 5


def test_training():
    model = DraftModel(ngram_order=2, vocab_size=20)
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    model.train(seq)
    assert len(model._counts) > 0
