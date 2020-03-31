# install opencc-python-reimplemented first
# for the conversion from simplified chinese to traditional chinese

import numpy as np
import pandas as pd
import jieba
from opencc import OpenCC
from gensim.models import Word2Vec, word2vec
from gensim.corpora import WikiCorpus

# load wiki dump
wiki_path = "zhwiki-20200220-pages-articles.xml.bz2"

wiki_corpus = WikiCorpus(wiki_path, dictionary={})  # 350000 articles
texts_num = 0

with open("wiki_texts.txt", 'w', encoding='utf-8') as output:
    for text in wiki_corpus.get_texts():
        output.write(' '.join(text))
        texts_num += 1
        if texts_num % 10000 == 0:
            print("{} articles finished.".format(texts_num))

# use jieba to segment and write segmented results to txt file
output = open("wiki_seg.txt", 'w', encoding='utf-8')
with open("wiki_zh_tw.txt", 'r', encoding='utf-8') as content:
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        words = jieba.cut(line)
        output.write(" ".join(words) + "\n")

        if (texts_num + 1) % 10000 == 0:
            print("finished {} lines.".format(texts_num + 1))

output.close()

# training
sentences = word2vec.LineSentence("wiki_seg.txt")
word2vec_model_1 = Word2Vec(sentences, size=250, sg=1, iter=5, hs=1)
word2vec_model_1.save("word2vec_1.model")  # save it for later use

# if you want to load pretrained model
# word2vec_model_1 = Word2Vec.load("word2vec_1.model")
