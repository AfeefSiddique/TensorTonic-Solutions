import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    if not candidate:
        return 0.0
    
    def get_ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams  = get_ngrams(reference, n)
        
        denom = sum(cand_ngrams.values())
        if denom == 0:
            return 0.0
        
        clipped = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in cand_ngrams.items())
        
        if clipped == 0:
            return 0.0
        
        precisions.append(clipped / denom)
    
    c, r = len(candidate), len(reference)
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    
    return bp * geo_mean