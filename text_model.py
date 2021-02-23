""""Class for text data."""
import string
import numpy as np
import torch

class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        # Maybe UNK means 'unkown'
        # word2id表示某个单词的id, 单词唯一标识, wordcount表示该单词出现次数
        self.word2id['<UNK>'] = 0
        self.wordcount['UNK'] = 9e9

        def tokenize_text(self, text):
            """
            1.过滤文本，将字符串处理成只存在ascii码，将标点映射为None
            2.小写化，去头尾空格
            3.根据空格将字符串分割为多个token
            """
            #  文本过滤， 将字符串text处理成只有‘ascii’字符
            text = text.encode('ascii', 'ignore').decode('ascii')
            # string.punctuation --> string内置的所有标点符号 -->!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
            # maketrans创建字符到字符的映射，translate根据该映射重写字符串
            # maketrans('', '', string.punctuation) 将标点符号映射为None
            # translate(str.maketrans('', '', string.punctuation)) --> 消除标点符号
            tokens = str(text).lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
            return tokens

        # 将文本中的单词添加到单词本，并更新单词id和相应单词计数
        def add_text_to_vocab(self, text):
            """
            将文本中的单词添加到单词本，并更新单词id和统计相应单词计数
            """
            tokens = self.tokenize_text(text)
            for token in tokens:
                if token not in self.word2id:
                    self.word2id[token] = len(self.word2id)
                    self.wordcount[token] = 0
                self.wordcount[token] += 1

        def threshold_rare_words(self, wordcount_threshold=5):
            """
            稀疏单词阈值设置，和根据阈值剔除稀疏单词，即将其id置为0
            """
            for w in self.word2id:
                if self.wordcount[w] < wordcount_threshold:
                    self.word2id[w] = 0

        def encode_text(self, text):
            """
            通过对应单词本中的数字编码，返回句子的对应的数字编码
            """
            tokens = self.tokenize_text(text)
            # dic.get(key, default=None) --> 返回key对应的value，没有该key则返回0
            x = [self.word2id.get(t, 0) for t in tokens]
            return x

        def get_size(self):
            return len(self.word2id)

    
class TextLSTMModel(torch.nn.Module):
    def __init__(self, 
                    texts_to_build_vocab, 
                    word_embed_dim=512, 
                    lstm_hidden_dim=512):
        super(TextLSTMModel, self).__init__()

        self.vocab = SimpleVocab()
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()