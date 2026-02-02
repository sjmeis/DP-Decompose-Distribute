import argparse
import re
import string

import numpy as np
import torch
import yake
from captum.attr import IntegratedGradients
from keybert import KeyBERT
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn, wordnet_ic
from nltk.corpus.reader.wordnet import information_content
from nltk.wsd import lesk
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import transformers.models.bert.modeling_bert


class BudgetDistributor:
    def __init__(self, distributor):
        self.distributor = str.lower(distributor)
        self.stop_words = set(stopwords.words('english'))

        if self.distributor == "keybert":
            self.kw_extractor = KeyBERT()
        elif self.distributor == "yake":
            self.yake_extractor = yake.KeywordExtractor(n=1, stopwords=None)
        elif self.distributor == "attention":
            model_name = "bert-base-uncased"
            self.model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif self.distributor == "gradients":
            model_name = "bert-base-uncased"
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = "cpu"
            self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif self.distributor == "ic":
            nltk.download('wordnet_ic')
            ic_files = ['ic-brown.dat', 'ic-semcor.dat', 'ic-bnc.dat', 'ic-shaks.dat', 'ic-treebank.dat']
            self.ics = []
            for name in ic_files:
                self.ics.append(wordnet_ic.ic(name))


    def distribute(self, text, epsilon):
        if self.distributor == "keybert":
            budgets = self.keybert(text, epsilon)
        elif self.distributor == "yake":
            budgets = self.yake_extract(text, epsilon)
        elif self.distributor == "gradients":
            budgets = self.extract_integrated_gradients(text, epsilon)
        elif self.distributor == "attention":
            budgets = self.extract_attention_weights(text, epsilon)
        elif self.distributor == "ic":
            budgets = self.get_information_content(text, epsilon)
        elif self.distributor == "pos":
            budgets = self.pos_scores(text, epsilon)
        elif self.distributor == "baseline":
            budgets = self.baseline(text, epsilon)

        return budgets


    def keybert(self, text, epsilon):
        vectorizer = CountVectorizer(token_pattern=r'\b\w+\b').fit([text])
        num_tokens = len(vectorizer.get_feature_names_out())
        scored_tokens = self.kw_extractor.extract_keywords(text, top_n=num_tokens, vectorizer=vectorizer)
        output = self._convert_scores(scored_tokens, text, epsilon, invert=True)
        return output


    def yake_extract(self, text, epsilon):
        num_tokens = len(re.findall(r'\b\w+\b', text.lower()))
        self.yake_extractor.top = num_tokens
        scored_tokens = self.yake_extractor.extract_keywords(text)
        output = self._convert_scores(scored_tokens, text, epsilon, invert=False)
        return output


    # Based on https://github.com/jessevig/bertviz sample code
    def extract_attention_weights(self, text, epsilon):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attention = outputs.attentions
        all_attentions = torch.stack(attention)
        avg_attention = torch.mean(all_attentions, dim=(0, 2)).squeeze(0)
        token_importance = torch.mean(avg_attention, dim=0)
        scores = token_importance.tolist()
        scored_tokens = list(zip(tokens, scores))
        scored_tokens = self._combine_bert_tokens(scored_tokens)
        
        filter_tokens = {'[CLS]', '[SEP]', "[PAD]"} | set(string.punctuation)
        scored_tokens = [(token, score) for token, score in scored_tokens if token not in filter_tokens]
        
        output = self._convert_scores(scored_tokens, text, epsilon)
        return output

    # https://captum.ai/tutorials/Bert_SQUAD_Interpret
    # https://captum.ai/tutorials/Bert_SQUAD_Interpret2
    def extract_integrated_gradients(self, text, epsilon):
        input_ids = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        input_ids = input_ids.to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])

        def forward(input_embeds):
            outputs = self.model(inputs_embeds=input_embeds)
            return outputs.last_hidden_state.mean(dim=1).sum(dim=-1)

        input_embeddings = self.model.embeddings(input_ids=input_ids["input_ids"])
        gradients = IntegratedGradients(forward)
        attributions = gradients.attribute(input_embeddings)
        scores = attributions.norm(dim=-1).squeeze().tolist()
        
        scored_tokens = list(zip(tokens, np.abs(scores)))
        scored_tokens = self._combine_bert_tokens(scored_tokens)
        
        filter_tokens = {'[CLS]', '[SEP]', "[PAD]"} | set(string.punctuation)
        scored_tokens = [(token, score) for token, score in scored_tokens if token not in filter_tokens]
        
        output = self._convert_scores(scored_tokens, text, epsilon)
        return output


    def get_information_content(self, text, epsilon):
        tokens = re.findall(r'\b\w+\b', text.lower())
        tagged = pos_tag(tokens)

        token_ic = []
        wn_tagged_tokens = self._convert_to_wordnet_pos(tagged)
        for tok, wn_pos in wn_tagged_tokens:
            syn = lesk(tokens, tok, pos=wn_pos) if wn_pos else lesk(tokens, tok)
            
            if syn:
                vals = []
                for ic in self.ics:
                    try:
                        vals.append(information_content(syn, ic))
                    except Exception:
                        continue
                ic_val = float(np.mean(vals)) if vals else 1.0
            else:
                ic_val = 1.0

            token_ic.append((tok, ic_val))

        scored_tokens = [(t, v) for t, v in token_ic if t.isalnum()]
        return self._convert_scores(scored_tokens, text, epsilon)
    

    def baseline(self, text, epsilon):
        tokens = re.findall(r'\b\w+\b', text.lower())
        non_stop_tokens = [t for t in tokens if t not in self.stop_words]
        len_nonstop = len(non_stop_tokens)

        if len_nonstop == 0:
            return [(t, 0.0) for t in tokens]

        per_token_eps = epsilon / len_nonstop

        result = []
        for t in tokens:
            if t in self.stop_words:
                result.append((t, 0.0))
            else:
                result.append((t, per_token_eps))

        return result


    def _convert_scores(self, token_score_map, text, epsilon, invert=True):
        original_word_sequence = re.findall(r'\b\w+\b', text.lower())

        provided_scores_map = {token.lower(): score for token, score in token_score_map}

        scores = []
        for word in original_word_sequence:
            score = provided_scores_map.get(word, 0.0)
            if word in self.stop_words:
                score = 0.0
            scores.append(score)
        
        scores = np.array(scores, dtype=float)

        non_zero_mask_for_shift = scores != 0
        if np.any(non_zero_mask_for_shift):
            min_non_zero_score = np.min(scores[non_zero_mask_for_shift])
            if min_non_zero_score < 0:
                scores[non_zero_mask_for_shift] += abs(min_non_zero_score)
        
        final_scores = np.zeros_like(scores, dtype=float)
        
        if invert:
            non_zero_mask = scores > 0
            if np.any(non_zero_mask):
                non_zero_scores = scores[non_zero_mask]
                
                min_val = np.min(non_zero_scores)
                max_val = np.max(non_zero_scores)
                
                if max_val == min_val:
                    inverted_non_zero = np.ones_like(non_zero_scores)
                else:
                    inverted_non_zero = (max_val + min_val) - non_zero_scores

                total_inverted_score = np.sum(inverted_non_zero)
                if total_inverted_score > 0:
                    distributed_scores = (inverted_non_zero / total_inverted_score) * epsilon
                    final_scores[non_zero_mask] = distributed_scores
        else: # not invert
            total_score = np.sum(scores)
            if total_score > 0:
                final_scores = (scores / total_score) * epsilon
                
        output = list(zip(original_word_sequence, final_scores))
        
        return output


    def _convert_to_wordnet_pos(self, tagged_tokens):
        wordnet_pos_tagged_tokens = []
        for token, tag in tagged_tokens:
            if tag.startswith('N'):
                wordnet_pos_tagged_tokens.append((token, wn.NOUN))
            elif tag.startswith('V'):
                wordnet_pos_tagged_tokens.append((token, wn.VERB))
            elif tag.startswith('J'):
                wordnet_pos_tagged_tokens.append((token, wn.ADJ))
            elif tag.startswith('RB'):
                wordnet_pos_tagged_tokens.append((token, wn.ADV))
            else:
                wordnet_pos_tagged_tokens.append((token, None))
        return wordnet_pos_tagged_tokens


    def _combine_bert_tokens(self, scored_tokens):
        combined_words = []
        current_word = ""
        current_score = 0.0
        
        for token, score in scored_tokens:
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score
            else:
                if current_word:
                    combined_words.append((current_word, current_score))
                
                current_word = token
                current_score = score
        
        if current_word:
            combined_words.append((current_word, current_score))
        
        return combined_words
    

    def _unrank_stopwords(self, scored_tokens):
        for i, (token, _) in enumerate(scored_tokens):
            if token in self.stop_words:
                scored_tokens[i] = (token, 0)
        
        return scored_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('-d', '--distributor', help='budget distribution method')
    parser.add_argument('-e', '--epsilon', help='total privacy budget')
    args = parser.parse_args()
    
    epsilon = int(args.epsilon)
    document = ''
    try:
        with open(args.input) as f:
            document = f.read()
    except Exception as e:
        print(e)
        return

    scorer = BudgetDistributor(distributor=args.distributor)
    
    if args.distributor == "keybert":
        result = scorer.keybert(document, epsilon)
    elif args.distributor == "yake":
        result = scorer.yake_extract(document, epsilon)
    elif args.distributor == "attention":
        result = scorer.extract_attention_weights(document, epsilon)
    elif args.distributor == "gradients":
        result = scorer.extract_integrated_gradients(document, epsilon)
    elif args.distributor == "lrp":
        result = scorer.extract_lrp(document, epsilon)
    elif args.distributor == "ic":
        result = scorer.get_information_content(document, epsilon)
    elif args.distributor == "pos":
        result = scorer.pos_scores(document, epsilon)
    elif args.distributor == "bert":
        result = scorer.bert_mask_score(document, epsilon)
    elif args.distributor == "baseline" or args.distributor is None:
        result = scorer.baseline(document, epsilon)
    else:
        print(f"Error: Unknown distributor '{args.distributor}'")

    print(result)


if __name__ == "__main__":
    main()