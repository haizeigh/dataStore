# import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# https://blog.csdn.net/qq_28840013/article/details/89681499

class MyWord2Ver:
    def getVer(self, sentences, size, window):
        wordVerDic = {}
        # size：词向量的维度
        # window：是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
        # min_count：忽略出现次数低于min_count的词
        model = Word2Vec(sentences, size=size, window=window , min_count=0, workers=4)
        wordList = model.wv.index2word
        for item in wordList:
            wordVerDic[item] = model[item]

        return wordVerDic




# model.save("word_embedding")	#模型会保存到该 .py文件同级目录下，该模型打开为乱码
# model.wv.save_word2vec_format("word_embedding1")  #通过该方式保存的模型，能通过文本格式打开，也能通过设置binary是否保存为二进制文件。但该模型在保存时丢弃了树的保存形式（详情参加word2vec构建过程，以类似哈夫曼树的形式保存词），所以在后续不能对模型进行追加训练
