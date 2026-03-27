import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Add special tokens first (IDs 0, 1, 2, 3)
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for token in special_tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = self.vocab_size
                self.id_to_word[self.vocab_size] = token
                self.vocab_size += 1
        
        # Collect all unique words from texts
        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)
        
        # Add unique words to vocabulary (sorted for consistency)
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        Unknown words map to UNK token ID.
        """
        words = text.split()
        token_ids = []
        
        for word in words:
            # Use .get() to handle unknown words
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        """
        words = []
        
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)
        
        return " ".join(words)