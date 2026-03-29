def word_count_dict(sentences):
    counts = {}
    for sentence in sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1
    return counts