import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

def create_small_gpt(vocab_size=50257, seq_len=128):
    """
    Create a small GPT-style model
    Small enough to run on most GPUs but big enough to stress memory
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_embd=512,      # hidden size
        n_layer=6,       # number of transformer blocks
        n_head=8,        # attention heads
        n_inner=2048,    # FFN intermediate size
    )
    
    model = GPT2LMHeadModel(config)
    
    # Count params (just for reference)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params/1e6:.2f}M parameters")
    
    return model


def create_larger_gpt(vocab_size=50257, seq_len=128):
    """
    Bigger model for stress testing
    This one will definitely hit memory limits faster
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_embd=768,      # bigger hidden
        n_layer=12,      # more layers
        n_head=12,
        n_inner=3072,
    )
    
    model = GPT2LMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Larger model created: {total_params/1e6:.2f}M parameters")
    
    return model


class SimpleDataset(torch.utils.data.Dataset):
    """
    Fake dataset for quick testing
    Don't need real data to measure memory usage
    """
    
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random tokens
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Labels are just shifted inputs (standard language modeling)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'labels': labels}
