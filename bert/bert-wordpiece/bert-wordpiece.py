from typing import List, Dict

class WordPieceTokenizer:
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab        = vocab
        self.unk_token    = unk_token
        self.max_word_len = max_word_len

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for word in text.lower().split():
            tokens.extend(self._tokenize_word(word))
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        if len(word) > self.max_word_len:
            return [self.unk_token]

        tokens = []
        start  = 0

        while start < len(word):
            end      = len(word)
            cur_tok  = None

            # Greedy longest-match from current position
            while start < end:
                substr = word[start:end]
                candidate = substr if start == 0 else "##" + substr
                if candidate in self.vocab:
                    cur_tok = candidate
                    break
                end -= 1

            # No subword found → whole word is UNK
            if cur_tok is None:
                return [self.unk_token]

            tokens.append(cur_tok)
            start = end

        return tokens