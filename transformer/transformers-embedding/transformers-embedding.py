import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Create embedding layer
    embedding = nn.Embedding(vocab_size, d_model)
    
    # Initialize with Xavier/standard normal scaled by sqrt(d_model)
    # This is the standard initialization for transformer embeddings
    nn.init.normal_(embedding.weight, mean=0, std=1.0 / math.sqrt(d_model))
    
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Look up embeddings
    embeddings = embedding(tokens)
    
    # Scale by sqrt(d_model) as per Transformer paper
    scaled_embeddings = embeddings * math.sqrt(d_model)
    
    return scaled_embeddings