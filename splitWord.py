import jieba
import jieba.posseg as pseg
import jieba.analyse as anl

class MySpliter:

    def split(self, word):
        seg_list = jieba.cut(word, cut_all=False)
        stopWords = []
        for stopWord in open("/Users/xmly/PycharmProjects/test1/com/stopwords/hit_stopwords.txt", 'r'):
            stopWords.append(stopWord.strip())

        splitWordList = []
        for word in seg_list:
            if (word not  in stopWords ) and self.check_all_chinese(word):
                splitWordList.append(word)

        return splitWordList

    # 只要中文的
    def check_all_chinese(self, check_str):
        for ch in check_str:
            if ch < u'\u4e00' or ch > u'\u9fff':
                return False
        return True


# mySpliter = MySpliter()
# mySpliter.split("yu")

# text = "欧阳建国是创新办主任也是欢聚时代公司云计算方面的专家，建国是建设国家的有力帮手"

# jieba.add_word("云计算", 8000)

# jieba.cut() 方法接受两个输入参数:
# 需要分词的字符串
# cut_all 参数用来控制是否采用全模式

#
#
# # 精确模式，默认模式就是精确模式
# seg_list = jieba.cut(text, cut_all = False)
# print('Default Mode:\n' + '/' .join(seg_list))
#
# # 全模式
# seg_list = jieba.cut(text, cut_all = True)
# print( "Full Mode:\n" + '/' .join(seg_list))
#
# # jieba.cut_for_search() 方法接受一个参数：
# # 需要分词的字符串
# # 该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细
#
# # 搜索引擎模式
# seg_list = jieba.cut_for_search(text)
# print('Research Mode:\n' + '/'.join(seg_list))
#
# print(jieba.get_FREQ("云"))
# print(jieba.get_FREQ("计算"))
# print(jieba.get_FREQ("云计算"))
#
# # 显示词性
# words = pseg.cut(text)
# for w in words:
#     print("%s %s" %(w.word, w.flag))
#
# print("显示权重， TF-IDF ")
# tags = anl.extract_tags(text, topK=20, withWeight=True, allowPOS=())
# for w , weight in tags:
#     print("%s %s" %(w, weight))
#
# # 去除停顿词
# print("去除停顿词")
#
# stopWords = []
# for stopWord in  open("/Users/xmly/PycharmProjects/test1/com/dataStoreTest/stop_word", 'r'):
#     print("stopWord = " + stopWord)
#     stopWords.append(stopWord.strip())
#
# stayed_line = []
# words = jieba.cut(text)
# for word in words:
#     word = word.strip()
#     if word not in stopWords:
#         stayed_line.append(word)
# print(stayed_line)
#
