import math
import json
import re
from collections import defaultdict

def load_unigrams(filepath):
    counts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit(None, 1)
            if len(parts) == 2:
                counts[parts[0]] = int(parts[1])
    return counts

# Top 10% with 150 min_count
def load_ngrams_to_dict(filepath, min_count=275, alphanumeric_only=True):
    counts = {}
    alnum_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().rsplit(None, 1)
            if len(parts) == 2:
                ngram_tokens = parts[0].split()
                count = int(parts[1])
                
                if count <= min_count:
                    continue

                if alphanumeric_only and not all(alnum_pattern.match(tok) for tok in ngram_tokens):
                    continue

                counts[tuple(ngram_tokens)] = count
    return counts


# 2x2 contingency table calculation
def calculate_llr(c11, c12, c21, c22):
    def x_log_x(x):
        return x * math.log(x) if x > 0 else 0
    
    N = c11 + c12 + c21 + c22
    if N == 0: return 0.0
        
    return 2 * (x_log_x(N) - x_log_x(c11 + c12) - x_log_x(c21 + c22) - x_log_x(c11 + c21) - x_log_x(c12 + c22) + x_log_x(c11) + x_log_x(c12) + x_log_x(c21) + x_log_x(c22))


unigram_counts = load_unigrams('unigrams.txt')
bigram_counts = load_ngrams_to_dict('bigrams.txt')
trigram_counts = load_ngrams_to_dict('trigrams.txt')
quadgram_counts = load_ngrams_to_dict('quadgrams.txt')

N = sum(unigram_counts.values())

all_scores = defaultdict(dict)

print("Process bigrams...")
for (w1, w2), c_w1_w2 in bigram_counts.items():
    ngram_str = f"{w1} {w2}"
    c_w1 = unigram_counts.get(w1, 0)
    c_w2 = unigram_counts.get(w2, 0)

    # PMI
    if c_w1 > 0 and c_w2 > 0 and c_w1_w2 > 0:
        pmi = math.log2((c_w1_w2 * N) / (c_w1 * c_w2))
        if pmi >= 2:
            all_scores['bigram_pmi'][ngram_str] = pmi

    # t-score
    if c_w1_w2 > 0:
        expected = (c_w1 * c_w2) / N
        t_score = (c_w1_w2 - expected) / math.sqrt(c_w1_w2)
        all_scores['bigram_t_score'][ngram_str] = t_score

    # LLR
    c12 = c_w1 - c_w1_w2
    c21 = c_w2 - c_w1_w2
    c22 = N - (c_w1_w2 + c12 + c21)
    all_scores['bigram_llr'][ngram_str] = calculate_llr(c_w1_w2, c12, c21, c22)

print("Process trigrams...")
for (w1, w2, w3), c_w1_w2_w3 in trigram_counts.items():
    ngram_str = f"{w1} {w2} {w3}"
    c_w1, c_w2, c_w3 = unigram_counts.get(w1, 0), unigram_counts.get(w2, 0), unigram_counts.get(w3, 0)
    
    # PMI
    if c_w1 > 0 and c_w2 > 0 and c_w3 > 0 and c_w1_w2_w3 > 0:
        pmi = math.log2((c_w1_w2_w3 * (N**2)) / (c_w1 * c_w2 * c_w3))
        if pmi >= 2:
            all_scores['trigram_pmi'][ngram_str] = pmi

    # t-score
    if c_w1_w2_w3 > 0:
        expected = (c_w1 * c_w2 * c_w3) / (N**2)
        t_score = (c_w1_w2_w3 - expected) / math.sqrt(c_w1_w2_w3)
        all_scores['trigram_t_score'][ngram_str] = t_score

    # LLR ((w1, w2) and w3)
    c_w1_w2 = bigram_counts.get((w1, w2), 0)
    c12 = c_w1_w2 - c_w1_w2_w3
    c21 = c_w3 - c_w1_w2_w3
    c22 = N - (c_w1_w2_w3 + c12 + c21)
    all_scores['trigram_llr'][ngram_str] = calculate_llr(c_w1_w2_w3, c12, c21, c22)

print("Process quadgrams...")
for (w1, w2, w3, w4), c_w1_w2_w3_w4 in quadgram_counts.items():
    ngram_str = f"{w1} {w2} {w3} {w4}"
    c_w1, c_w2, c_w3, c_w4 = [unigram_counts.get(w, 0) for w in (w1, w2, w3, w4)]

    # PMI
    if all(c > 0 for c in [c_w1, c_w2, c_w3, c_w4]) and c_w1_w2_w3_w4 > 0:
        pmi = math.log2((c_w1_w2_w3_w4 * (N**3)) / (c_w1 * c_w2 * c_w3 * c_w4))
        if pmi >= 2:
            all_scores['quadgram_pmi'][ngram_str] = pmi

    # t-score
    if c_w1_w2_w3_w4 > 0:
        expected = (c_w1 * c_w2 * c_w3 * c_w4) / (N**3)
        t_score = (c_w1_w2_w3_w4 - expected) / math.sqrt(c_w1_w2_w3_w4)
        all_scores['quadgram_t_score'][ngram_str] = t_score

    # LLR ((w1, w2) and (w3, w4))
    c_w1_w2 = bigram_counts.get((w1, w2), 0)
    c_w3_w4 = bigram_counts.get((w3, w4), 0)
    c12 = c_w1_w2 - c_w1_w2_w3_w4
    c21 = c_w3_w4 - c_w1_w2_w3_w4
    c22 = N - (c_w1_w2_w3_w4 + c12 + c21)
    all_scores['quadgram_llr'][ngram_str] = calculate_llr(c_w1_w2_w3_w4, c12, c21, c22)

for key, scores_dict in all_scores.items():
    with open(f"{key}_scores.json", 'w', encoding='utf-8') as f:
        json.dump(scores_dict, f, indent=4)

print("Done.")