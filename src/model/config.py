from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6  # Reduced number of layers for faster training
    n_head: int = 8   # Modified number of attention heads
    n_embd: int = 512 # Reduced embedding dimension
    dropout: float = 0.1  # Added dropout for regularization

    def __post_init__(self):
        if self.n_layer < 1:
            raise ValueError("n_layer must be at least 1")
        if self.n_head < 1:
            raise ValueError("n_head must be at least 1")
        if self.n_embd < 1:
            raise ValueError("n_embd must be at least 1")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1") 