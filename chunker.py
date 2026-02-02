import string
import nltk
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import conll2000, stopwords
from tag_chunker import TagChunker
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from datasets import load_dataset
import multiprocessing
import re
import time
import os

worker_chunker = None

def init_worker(chunker_method):
    global worker_chunker
    worker_chunker = ChunkGenerator(chunker_method)


def chunk_document_worker(doc_text):
    global worker_chunker
    if worker_chunker:
        return list(worker_chunker.chunk_text(doc_text))
    return []


class FinewebCorpusGenerator:    
    def __init__(self, chunker, sample_size=None, num_processes=4, chunksize=50):
        self.chunker = chunker
        self.sample_size = sample_size
        self.num_processes = num_processes
        self.chunksize = chunksize
    
    def __iter__(self):

        if self.sample_size:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb", 
                name="sample-10BT",
                split="train",
                streaming=True
            ).take(self.sample_size)

        else:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb", 
                name="sample-10BT",
                split="train",
                streaming=True
            )
        doc_generator = (doc['text'] for doc in dataset)

        processed_docs = 0
        start_time = time.perf_counter()

        with multiprocessing.Pool(self.num_processes, initializer=init_worker, initargs=(self.chunker,)) as pool:
            for list_of_sentences in pool.imap_unordered(chunk_document_worker, doc_generator, chunksize=self.chunksize):
                processed_docs += 1

                if processed_docs % 1_000_000 == 0:
                    elapsed = time.perf_counter() - start_time
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = elapsed % 60
                    print(f"Processed {processed_docs:,} documents in {hours}h {minutes}m {seconds:.2f}s")

                for sentence in list_of_sentences:
                    yield sentence


class ChunkGenerator:

    def __init__(self, method):
        self.method = method
        self.detokenizer = TreebankWordDetokenizer()
        self.punct = set(string.punctuation)
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        self.contractions = {
            "i m", "i ve", "i ll", "i d", "amn t", "ain t", "aren t", "can t", "could ve", "couldn t", "daren t", "daresn t", "dasn t", "didn t", "don t", "doesn t", "e er", "everyone s", "gon t", "hadn t", "hasn t", "haven t", "he ve", "he s", "he ll", "he d", "here s", "how re", "how d", "how s", "how ll", "isn t", "it s", "it ll", "it d", "let s", "ma am", "may ve", "mayn t", "might ve", "mightn t", "must ve", "mustn t", "needn t", "ne er", "o clock", "oughtn t", "o er", "shan t", "shalln t", "she s", "she ll", "she d", "should ve", "shouldn t", "so ve", "so s", "somebody s", "someone s", "something s", "that re", "that s", "that ll", "that d", "there re", "there s", "there ll", "there d", "these re", "they re", "they ve", "they ll", "they d", "they d ve", "this s", "this ll", "this d", "those re", "to ve", "wasn t", "we re", "we ve", "we ll", "we d", "weren t", "what re", "what d", "what ve", "what s", "what ll", "when ve", "when s", "where re", "where d", "where ve", "where s", "which s", "who re", "who ve", "who s", "who ll", "who d", "why re", "why d", "why ve", "why s", "will ve", "won t", "would ve", "wouldn t", "y all", "you re", "you ve", "you ll", "you d"
        }

        if self.method == "pmi":
            self.bigrams, self.trigrams, self.quadgrams = self._load_files("pmi") 
        if self.method == "llr":
            self.bigrams, self.trigrams, self.quadgrams = self._load_files("llr") 
        if self.method == "t_score":
            self.bigrams, self.trigrams, self.quadgrams = self._load_files("t_score") 
        if self.method == "pos":
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('conll2000', quiet=True)
            train_sents = conll2000.chunked_sents('train.txt')
            self.pos_chunker = TagChunker(train_sents)
        if self.method == "wordnet":
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)


    def _load_files(self, measure):
        with open(f"measure_scores/bigram_{measure}_scores.json", 'r') as f:
            bigrams = set(json.load(f).keys())
        with open(f"measure_scores/trigram_{measure}_scores.json", 'r') as f:
            trigrams = set(json.load(f).keys())
        with open(f"measure_scores/quadgram_{measure}_scores.json", 'r') as f:
            quadgrams = set(json.load(f).keys())

        return bigrams, trigrams, quadgrams
    

    def _process_ngram(self, ngram_words):
        start_idx = 0
        while start_idx < len(ngram_words) and ngram_words[start_idx].lower() in self.stopwords:
            start_idx += 1

        end_idx = len(ngram_words) - 1
        while end_idx >= 0 and ngram_words[end_idx].lower() in self.stopwords:
            end_idx -= 1

        if start_idx > end_idx:
            return ngram_words

        leading_stopwords = ngram_words[:start_idx]
        core_chunk = "_".join(ngram_words[start_idx : end_idx + 1])
        trailing_stopwords = ngram_words[end_idx + 1:]

        processed_tokens = []
        processed_tokens.extend(leading_stopwords)
        processed_tokens.append(core_chunk)
        processed_tokens.extend(trailing_stopwords)

        return processed_tokens


    def _match_ngrams(self, bigrams, trigrams, quadgrams, text):
        
        for sent in nltk.sent_tokenize(text):
            tokens = re.findall(r'\b\w+\b', sent.lower())
            if not tokens:
                continue

            sentence_collocations = []
            i = 0
            while i < len(tokens):

                # Check for contractions first
                if i + 1 < len(tokens):
                    potential_contraction = f"{tokens[i]} {tokens[i+1]}".lower()
                    if potential_contraction in self.contractions:
                        sentence_collocations.append(f"{tokens[i]}_{tokens[i+1]}")
                        i += 2
                        continue

                # Check for Quadgram
                if i + 3 < len(tokens):
                    ngram_str = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]} {tokens[i+3]}".lower()
                    if ngram_str in quadgrams:
                        processed_list = self._process_ngram(tokens[i:i+4])
                        sentence_collocations.extend(processed_list)
                        i += 4
                        continue

                # Check for Trigram
                if i + 2 < len(tokens):
                    ngram_str = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".lower()
                    if ngram_str in trigrams:
                        processed_list = self._process_ngram(tokens[i:i+3])
                        sentence_collocations.extend(processed_list)
                        i += 3
                        continue

                # Check for Bigram
                if i + 1 < len(tokens):
                    ngram_str = f"{tokens[i]} {tokens[i+1]}".lower()
                    if ngram_str in bigrams:
                        processed_list = self._process_ngram(tokens[i:i+2])
                        sentence_collocations.extend(processed_list)
                        i += 2
                        continue

                sentence_collocations.append(tokens[i])
                i += 1

            yield sentence_collocations


    def pos_conll(self, text): 
        for sent in nltk.sent_tokenize(text):
            sentence_collocations = []
            tokens = re.findall(r'\b\w+\b', sent.lower()) 
            tagged_tokens = pos_tag(tokens)
            tree = self.pos_chunker.parse(tagged_tokens)

            i = 0
            flat_tokens = []

            for subtree in tree:
                if isinstance(subtree, nltk.Tree):
                    original_words = [word for word, _ in subtree.leaves()]
                    processed_tokens = self._process_ngram(original_words)
                    flat_tokens.extend(processed_tokens)
                else:
                    word, _ = subtree
                    flat_tokens.append(word)

            # merge contractions
            i = 0
            while i < len(flat_tokens):
                if i + 1 < len(flat_tokens):
                    potential_contraction = f"{flat_tokens[i]} {flat_tokens[i+1]}".lower()
                    if potential_contraction in self.contractions:
                        sentence_collocations.append(f"{flat_tokens[i]}_{flat_tokens[i+1]}")
                        i += 2
                        continue
                sentence_collocations.append(flat_tokens[i])
                i += 1

            yield sentence_collocations


    def wordnet_collocations(self, text):
        for sent in nltk.sent_tokenize(text):
            tokens = re.findall(r'\b\w+\b', sent.lower())

            sentence_collocations = []
            i = 0
            while i < len(tokens):
                # Check for contractions
                if i + 1 < len(tokens):
                    potential_contraction = f"{tokens[i]} {tokens[i+1]}".lower()
                    if potential_contraction in self.contractions:
                        sentence_collocations.append(f"{tokens[i]}_{tokens[i+1]}")
                        i += 2
                        continue

                if i + 3 < len(tokens) and wn.synsets(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}_{tokens[i+3]}"):
                    processed_list = self._process_ngram(tokens[i:i+4])
                    sentence_collocations.extend(processed_list)
                    i += 4
                    continue

                if i + 2 < len(tokens) and wn.synsets(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"):
                    processed_list = self._process_ngram(tokens[i:i+3])
                    sentence_collocations.extend(processed_list)
                    i += 3
                    continue

                if i + 1 < len(tokens) and wn.synsets(f"{tokens[i]}_{tokens[i+1]}"):
                    processed_list = self._process_ngram(tokens[i:i+2])
                    sentence_collocations.extend(processed_list)
                    i += 2
                    continue
                
                sentence_collocations.append(tokens[i])
                i += 1
            yield sentence_collocations


    def chunk_text(self, text):
        if self.method in ["pmi", "llr", "t_score"]:
            sentences = self._match_ngrams(self.bigrams, self.trigrams, self.quadgrams, text)
        elif self.method == "pos":
            sentences = self.pos_conll(text)
        elif self.method == "wordnet":
            sentences = self.wordnet_collocations(text)        
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")
        
        # Remove punctuation from the sentences and yield entire sentence
        for sentence in sentences:  
            sentence_collocations = []
            for token in sentence:
                if token and all(c.isalnum() or c in '-_' for c in token):
                    sentence_collocations.append(token)
            yield sentence_collocations


    def load_document(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                document = f.read()
                yield from self.chunk_text(document)
        except Exception as e:
            print(e)


    def load_fineweb_dataset(self, sample_size=None):
        if sample_size:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb", 
                name="sample-10BT",
                split="train",
                streaming=True
            ).take(sample_size)

        else:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb", 
                name="sample-10BT",
                split="train",
                streaming=True
            )

        for doc_dict in dataset:
            doc = doc_dict['text']
            yield from self.chunk_text(doc)


    def load_from_csv(self, filepath, column_name):
        import pandas as pd
        df = pd.read_csv(filepath)

        for text in df[column_name].dropna():
            if isinstance(text, str):
                yield from self.chunk_text(text)