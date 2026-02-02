#!/usr/bin/env python3
import re
import gc
import time
from collections import Counter
from datasets import load_dataset


quadgrams = Counter()
total_quadgrams = 0

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
    
    if n >= 4:
        new_quadgrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]} {tokens[i+3]}" 
                        for i in range(n - 3)]
        quadgrams.update(new_quadgrams)
        total_quadgrams += len(new_quadgrams)
    
    doc_count += 1
    if doc_count % 10000 == 0:
        print(f"Processed {doc_count:,} documents")
    
    if doc_count % 50000 == 0:
        print(f"Removing quadgrams with count = 1...")
        before = len(quadgrams)
        
        quadgrams = Counter({k: v for k, v in quadgrams.items() if v > 1})
        
        after = len(quadgrams)
        print(f"Removed {before - after:,} quadgrams")
        
        gc.collect()


time.sleep(10)
with open('total_quadgrams.txt', 'w') as f:
    f.write(str(total_quadgrams))


with open('quadgrams.txt', 'w') as f:
    for quadgram, count in quadgrams.items():
        if count > 20:
            f.write(f"{quadgram}\t{count}\n")

print("Done!")