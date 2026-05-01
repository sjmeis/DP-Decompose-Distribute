import argparse
from gensim.models import Word2Vec
import pandas as pd
import re
import csv
import os

from chunker import ChunkGenerator
from distributor import BudgetDistributor
from MLDP import MLDP
#from Diffractor import Diffractor, Lists, Algorithm

TOKEN_RE = re.compile(r"(\w+|[^\w\s])")


def reconstruct_with_template(original_tokens, privatized_chunks, all_chunks):
    content_to_chunk = []
    for chunk_idx, chunk in enumerate(all_chunks):
        words_in_chunk = chunk.split('_')
        for _ in words_in_chunk:
            content_to_chunk.append(chunk_idx)
    
    result = []
    content_idx = 0
    inserted_chunks = set()
    
    for token in original_tokens:
        if token.isalnum():
            # Find which chunk this content token belongs to
            if content_idx < len(content_to_chunk):
                chunk_idx = content_to_chunk[content_idx]
                
                # Insert chunk only once (when we hit the first word of the chunk)
                if chunk_idx not in inserted_chunks:
                    result.append(privatized_chunks[chunk_idx])
                    inserted_chunks.add(chunk_idx)
                
                content_idx += 1
        else:
            # Punctuation - keep as is
            result.append(token)
    
    return result

def get_chunked_budgets(clean_text, chunker, distributor, epsilon):
    # Chunk text
    chunked_sentences = chunker.chunk_text(clean_text)
    all_chunks = [chunk for sentence in chunked_sentences for chunk in sentence]
    
    # Distribute budgets
    word_budgets_list = distributor.distribute(clean_text, epsilon)
    word_budget_iterator = iter(word_budgets_list)
    final_chunk_budgets = []

    # Combine both
    for chunk in all_chunks:
        words_in_chunk = chunk.split('_')
        num_words = len(words_in_chunk)
        chunk_budget = 0.0
        for _ in range(num_words):
            _, budget = next(word_budget_iterator)
            chunk_budget += budget

        final_chunk_budgets.append((chunk, chunk_budget))

    return final_chunk_budgets, all_chunks


def privatize_text(chunked_budgets, word_vectors, mechanism):
    privatized_chunks = []

    #print("\nPrivatizing chunks...")
    for chunk, epsilon in chunked_budgets:
 
        if epsilon > 0 and chunk in word_vectors:
            privatized_chunk = mechanism.replace_word(chunk, epsilon)
            privatized_chunks.append(privatized_chunk)
            #print(f"  '{chunk}' (epsilon: {epsilon:.2f}) -> '{privatized_chunk}'")
        else:
            privatized_chunks.append(chunk)
    
    return privatized_chunks


def privatize_dataset(document, mechanism, text_chunker, budget_distributor, word_vectors, epsilon):
    text = document.replace("'", " ")
    all_tokens = TOKEN_RE.findall(text)
    
    content_tokens = [t for t in all_tokens if t.isalnum()]
    clean_text = ' '.join(content_tokens)

    chunked_budgets, all_chunks = get_chunked_budgets(clean_text, text_chunker, budget_distributor, epsilon)
    privatized_chunks = privatize_text(chunked_budgets, word_vectors, mechanism)
    
    privatized_chunks = privatize_text(chunked_budgets, word_vectors, mechanism)

    final_tokens = reconstruct_with_template(all_tokens, privatized_chunks, all_chunks)

    privatized_text = ' '.join(final_tokens).replace('_', ' ')
    privatized_text = re.sub(r'\s+([,.!?;:%])', r'\1', privatized_text)
    for left, right in [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]:
        privatized_text = privatized_text.replace(f"{left} ", left)
        privatized_text = privatized_text.replace(f" {right}", right)

    return privatized_text


def main():
    parser = argparse.ArgumentParser(description="Privatize a dataset using MLDP and specific budget distribution.")
    
    # Required Arguments
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV dataset')
    parser.add_argument('-c', '--chunker', required=True, help='Chunking method (pmi, llr, pos, wordnet, etc.)')
    parser.add_argument('-m', '--model', required=True, help='Path to the Word2Vec model')
    parser.add_argument('-d', '--distributor', default='baseline', 
                        help='Distribution method (keybert, yake, attention, ic, baseline)')
    parser.add_argument('-e', '--epsilon', type=float, 
                        help='Privacy budget. If omitted, uses dataset-specific defaults.')
    parser.add_argument('-col', '--column', 
                        help='The column name containing text to privatize.')

    args = parser.parse_args()

    print(f"Initialize {args.chunker} text chunker...")
    text_chunker = ChunkGenerator(args.chunker)
    
    print(f"Load {args.model} Word2Vec model...")
    w2v_model = Word2Vec.load(args.model)
    word_vectors = w2v_model.wv

    print(f"Initialize {args.distributor} distributor...")
    budget_distributor = BudgetDistributor(args.distributor)

    print(f"Initialize MLDP mechanism...")
    mechanism = MLDP.MultivariateCalibrated(embedding_matrix=word_vectors)

    dataset_path = args.input
    df = pd.read_csv(dataset_path)
    
    column = args.column
    epsilon = args.epsilon

    if not column:
        column = "text" # Default fallback

    if epsilon is None:
        epsilon = 50 # Default fallback

    input_dir, input_filename = os.path.split(dataset_path)
    privatized_column_name = f'privatized_{column}'
    output_filename = os.path.join(input_dir, f'privatized_{args.distributor}_{args.chunker}_{epsilon}_{input_filename}')

    print(f"Running privatization on '{column}' with epsilon={epsilon}...")
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(df.columns.tolist() + [privatized_column_name])
        
        for index, row in df.iterrows():
            original_text = str(row[column])
            if not original_text.strip():
                privatized_text = ""
            else:
                if (index + 1) % 100 == 0:
                    print(f"  Privatized {index+1} / {len(df)}...")
                
                privatized_text = privatize_dataset(
                    original_text, 
                    mechanism, 
                    text_chunker, 
                    budget_distributor, 
                    word_vectors, 
                    epsilon
                )
            
            new_row = row.values.tolist() + [privatized_text]
            writer.writerow(new_row)

    print(f"Success! Results saved to: {output_filename}")

if __name__ == '__main__':
    main()
