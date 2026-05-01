# DP Text Decomposition and Privacy Budget Distribution
In this repository, you will find all implementation and evaluation scripts as detailed in the PrivateNLP 2026 submission: *A Systematic Exploration of Text Decomposition and Budget Distribution in Differentially Private Text Obfuscation*

In this work, we implement a pipeline for privatizing text datasets under metric Differential Privacy. It particular, we utilize various NLP techniques to identify chunks (using PMI, LLR, POS tagging, etc.) and distribute a privacy budget ($\epsilon$) across these chunks to replace sensitive terms with private alternatives. The above paper conducts systematic investigations at the intersection of these two, i.e., text decomposition and privacy budget distribution.

## Project Structure

| File                    | Description                                                                                                                         |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `privatizer.py`           | Main Entry Point. Orchestrates the privatization of CSV text datasets.                                                                   |
| `chunker.py`              | Handles text decomposition into units (words or n-grams) based on various methods.                                                   |
| `distributor.py`          | Distributes the privacy budget ($\epsilon$) across tokens using strategies like KeyBERT, Attention weights, or Information Content. |
| `ngram_counter.py`        | Scrapes the HuggingFace FineWeb dataset to count unigrams through quadgrams.                                                        |
| `association_measures.py` | Calculates statistical scores (PMI, LLR, T-Score) for n-grams to identify  collocations.                                      |
| `tag_chunker.py`          | A helper class for Part-of-Speech (POS) based chunking using a Bigram Tagger.                                                       |
| `train_word2vec.py`       | Trains custom Word2Vec models on the Fineweb corpus, used for DP chunk replacement.                                                 |
| `quadgram_counter.py`     | A memory-optimized counter specifically for 4-grams.                                                                                |


## Workflow

1. Data Preparation & Statistics

   First, generate the basis for text decomposition. This requires counting ngrams from FineWeb, and then calculating the resulting association measures.
   - Run `ngram_counter.py` to generate frequency lists from the FineWeb dataset. 
   - Run `association_measures.py` to compute association scores. This creates JSON files in `measure_scores/` which the `ChunkGenerator` uses to decide which words belong together (e.g., "New_York").

2. Model Training

    Train a Word2Vec model on the chunked corpus. This ensures that the privatization mechanism understands the context of multi-word expressions.

```bash
python train_word2vec.py -i fineweb --chunker pmi --methods pmi,llr
```

3. Privatization
   
   Use the `privatizer.py` script to apply Differential Privacy to your target text dataset. The script allows you to specify the decomposotion method, the distribution method, the privacy budget ($\epsilon$), and the target text column directly.

**General Usage:**

```bash
python privatizer.py -i <dataset.csv> -c <chunker> -m <model_path> -d <distributor> -e <epsilon> -col <column_name>
```

**Examples:**
 - POS Chunking with KeyBERT distribution:
```bash
python privatizer.py -i yelp_sample.csv -c pos -m word2vec_pos.model -d keybert -e 50 -col review
```

 - LLR Chunking with Attention-based distribution:

```
python privatizer.py -i enron_sample.csv -c llr -m word2vec_llr.model -d attention -e 25 -col text
```

### Command Line Arguments for `privatizer.py`

| Argument    | Flag | Description                                                                        |
|-------------|------|------------------------------------------------------------------------------------|
| Input       | -i   | Path to the input CSV file.                                                        |
| Chunker     | -c   | The method used to group words (pmi, llr, pos, wordnet, t_score).                  |
| Model       | -m   | Path to the trained Word2Vec model file.                                           |
| Distributor | -d   | Budget distribution strategy (keybert, yake, attention, ic, gradients, baseline).  |
| Epsilon     | -e   | The total privacy budget ($\epsilon$). If omitted, uses dataset-specific defaults. |
| Column      | -col | The specific column in the CSV to privatize.                                       |

### Note
we note that `privatizer.py` uses the MADLIB (multivariate calibrated noise) approach of Feyistean et al. (2020). Adapting our framework to other similar mechanisms can be done by swapping out the mechanism in this script.

## Evaluation & Metrics

The `eval_scripts/` directory contains a suite of tools to evaluate text privatization, as implemented in the paper.

1. Privacy

    To measure privacy leakage, we simulate an "attacker" attempting to identify sensitive attributes (Author ID or Gender) from the text. All attackers use a `microsoft/deberta-v3-base` model.
   - `baseline_attacker.py`: Establishes the "upper bound" by training and testing on non-privatized (original) data.
   - `static_attacker.py`: The attacker model is trained on original data but attempts to classify privatized data.
   - `adaptive_attacker.py`: The attacker model is trained directly on privatized data, allowing it to "learn" and potentially bypass the noise introduced by the mechanism.

2. Utility

    These scripts measure how much "useful" information is lost during privatization.
     - `calculate_cs.py` (Cosine Similarity):
        - Measures semantic drift between original and privatized sentences.
        - Uses an ensemble of three models: all-MiniLM-L12-v2, all-mpnet-base-v2, and gte-small.
     - `calculate_perplexity.py` (Text Coherence):
        - Measures the "naturalness" or fluency of the privatized text using GPT-2.
        - Lower perplexity indicates that the privatized text still follows the rules of the English language.
     - `sentiment.py` (Downstream Task Performance):
        - Evaluates if the sentiment of a review is preserved after the text is privatized.
        - High F1 here proves the privatization is "utility-preserving".

### Running Evaluations
The evaluation scripts are designed to automatically scan the `privatized_datasets/` folder for csv files and process them.

**Usage examples:**

```bash
python eval_scripts/calculate_cs.py

python eval_scripts/adaptive_attacker.py

python eval_scripts/sentiment.py
```

**Output files**

Results are appended to auto-generated csv files for easy comparison and analysis. These take the form of `MODULE_results.csv`.

## Citation
If you find this work or the provided tools useful, please consider citing the published work:

```
Coming soon!
```
