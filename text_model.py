""""Class for text data."""
import string
import numpy as np
import torch

class SimpleVocab(object):
    """
    词汇表类：输入多句子的篇章，使用add_text_to_vocab形成单词本\n
    然后输入句子使用encode_text将每个单词(字)编码为一个数字，最终返回句子对应的数字列表
    """

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
    """
    LSTM模型
    """
    def __init__(self, 
                    texts_to_build_vocab, 
                    word_embed_dim=512, 
                    lstm_hidden_dim=512):
        super(TextLSTMModel, self).__init__()

        self.vocab = SimpleVocab()
        # 建立单词本
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()

        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        # embedding第一个参数是单词本的大小，第二个是输出向量的维度
        # 特别的输入的不同单词数 < 单词本大小，不能大于和等于
        # 输入 --> 单词的id
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
        self.fc_output = torch.nn.Sequential(
            #  probability of an element to be zeroed. Default: 0.5
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
        )

    def forward(self, x):
        """input x: list of strings"""
        if type(x) is list:
            if type(x[0]) is str or type(x[0]) is unicode:
                # text是数字列表， x是数字列表的列表
                x = [self.vocab.encode_text(text) for text in x]
        assert type(x) is list
        assert type(x[0]) is list
        assert type(x[0][0]) is int 
        return self.forward_encoded_texts(x)

    # texts是数字列表的列表
    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        # 加'.long()'是为了使zeros矩阵从浮点类型变成整形
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        # 将texts放入torch.tensor的2维张量中，不足的地方用0占位
        for i in range(len(texts)):
            # itexts中每一列对应每个数字列表(即句子的编码)
            itexts[:lengths[i], i] = torch.tensor(texts[i])

        # embed words
        itexts = torch.autograd.Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)

        # get last output (using length)
        text_features = []
        for i in range(len(texts)):
            # 注意此处获取最终输出是用的 “lengths[i] - 1” , 而不是-1, 
            # 因为itexts中有0占位符, 所以应该取句子长度
            text_features.append(lstm_output[lengths[i] - 1, i, :])

        # output
        # torch.stack将tensor的列表链接成为 "tensor的tensor"
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        
        return text_features

    def forward_lstm_(self, etexts):
        return None
