import re
from collections import Counter
from datasets import load_dataset

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()
quadgrams = Counter()

dataset = load_dataset(
    "HuggingFaceFW/fineweb", 
    name="sample-10BT",
    split="train",
    streaming=True
)

doc_count = 0

for example in dataset:
    text = example['text']
    tokens = re.findall(r'\b\w+\b', text.lower())
    n = len(tokens)
    
    unigrams.update(tokens)
    
    if n >= 2:
        bigrams.update([f"{tokens[i]} {tokens[i+1]}" for i in range(n - 1)])
    
    if n >= 3:
        trigrams.update([f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(n - 2)])
    
    if n >= 4:
        quadgrams.update([f"{tokens[i]} {tokens[i+1]} {tokens[i+2]} {tokens[i+3]}" for i in range(n - 3)])
    
    doc_count += 1
    if doc_count % 10000 == 0:
        print(f"Processed {doc_count:,} documents...")


unigram_tokens = sum(unigrams.values())
bigram_tokens = sum(bigrams.values())
trigram_tokens = sum(trigrams.values())
quadgram_tokens = sum(quadgrams.values())

with open('total_unigrams.txt', 'w') as f:
    f.write(str(unigram_tokens))

with open('total_bigrams.txt', 'w') as f:
    f.write(str(bigram_tokens))

with open('total_trigrams.txt', 'w') as f:
    f.write(str(trigram_tokens))

with open('total_quadgrams.txt', 'w') as f:
    f.write(str(quadgram_tokens))


print("Saving unigrams...")
with open('unigrams.txt', 'w') as f:
    for word, count in unigrams.most_common():
        f.write(f"{word}\t{count}\n")

print("Saving bigrams...")
with open('bigrams.txt', 'w') as f:
    for bigram, count in bigrams.most_common():
        f.write(f"{bigram}\t{count}\n")

print("Saving trigrams...")
with open('trigrams.txt', 'w') as f:
    for trigram, count in trigrams.most_common():
        f.write(f"{trigram}\t{count}\n")

print("Saving quadgrams...")
with open('quadgrams.txt', 'w') as f:
    for quadgram, count in quadgrams.most_common():
        f.write(f"{quadgram}\t{count}\n")

print("Done.")
