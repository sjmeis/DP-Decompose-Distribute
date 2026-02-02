# tag_chunker.py
from nltk.chunk import ChunkParserI
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import BigramTagger

# https://nlp4everyone.medium.com/nlp-chunking-and-information-extraction-from-text-using-nltk-c34d2afe9fc7
# https://www.nltk.org/book/ch07.html subchapter 3
class TagChunker(ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = BigramTagger(train_data)


    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return conlltags2tree(conlltags)