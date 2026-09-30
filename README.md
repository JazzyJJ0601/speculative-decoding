# Speculative Decoding

Speculative decoding (also known as speculative decoding) is a technique to accelerate autoregressive language model inference. Instead of generating one token at a time, a small draft model proposes multiple candidate tokens in parallel, and a larger target model verifies them in a single forward pass using rejection sampling.

## Implementation Details

- **DraftModel**: Uses a lightweight n-gram predictor to propose candidates based on recent context history.
- **TargetModel**: Provides logits for verification via logit comparison.
- **Rejection Sampling Loop**: Accepts draft tokens with probability proportional to the ratio of target to draft probabilities.

### Benchmark Results

On synthetic data (vocab=50, steps=10), the implementation typically achieves 40–60% acceptance rates, demonstrating the trade-off between draft quality and target verification.

### Usage

```python
import numpy as np
from src.spec_decode.core import DraftModel, TargetModel, rejection_sampling

draft = DraftModel(ngram_order=2, vocab_size=100)
target = TargetModel(vocab_size=100)

# Train draft model on some sequence
seq = np.random.randint(0, 100, 100)
draft.train(seq)

# Generate tokens
tokens = rejection_sampling(draft, target, max_steps=20)
print(f"Accepted tokens: {len(tokens)}")
```

## Tests

Run the test suite with:
```bash
pytest tests/ -q
```
