import  xlrd
from splitWord import MySpliter
from MyWord2Ver import MyWord2Ver
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split#将数据分为测试集和训练集


def getClasslabel(num):
    if(num < 1000):
        # 一千
        return 1
    if (num < 10000):
        # 一万
        return 2
    if (num < 100000):
        # 十万
        return 3
    if (num < 500000):
        # 五十万
        return 4
    if (num < 1000000):
        # 一百万
        return 5
    if (num < 2000000):
        # 二百万
        return 6
    if (num < 3000000):
        # 三百万
        return 7
    if (num < 4000000):
        # 四百万
        return 8
    if (num < 5000000):
        # 五百万
        return 9
    if (num < 6000000):
        # 六百万
        return 10
    if (num < 7000000):
        # 七百万
        return 11
    if (num < 8000000):
        # 八百万
        return 12
    if (num < 9000000):
        # 九百万
        return 13
    if (num < 10000000):
        # 一千万
        return 14
    if (num >=10000000):
        return 15


def getClasslabel100(num):
    # // "表示整数除法。
    return num//100000

def test():
    print(getClasslabel100(11200001))


def analysis():
    # 打开文件
    excel = xlrd.open_workbook("/Users/xmly/PycharmProjects/test1/com/data/sources/bilibili_ test.xlsx")
    sheet = excel.sheet_by_name("bilibili_")
    print("总行：" + str(sheet.nrows))
    print("总列：" + str(sheet.ncols))
    mySpliter = MySpliter()
    myWord2Ver = MyWord2Ver()
    # print(r)

    # 获取词向量
    sentences = []
    inputDic = []
    allWordsList = []
    # 存放播放量
    Y_input = []
    for rowNum in range(1, sheet.nrows -1 ):
    # for rowNum in range(1, 3000 ):
        sen = sheet.row_values(rowNum, 0, 6)[0]
        play_num = sheet.row_values(rowNum, 0, 6)[5]
        sentence = mySpliter.split(sen )
        # 只有一个词的排除
        if (len(sentence) <= 1):
            continue
        sentences.append(sentence)
        # Y_input.append(getClasslabel(play_num))
        Y_input.append(getClasslabel100(play_num))
        allWordsList.extend(sentence)
        print("第"+ str(rowNum) +"行句子分词：" + str(sentence) )
        # print(split)

    allWordsSet = set(allWordsList)
    print( "集合中总计单词数目：" +  str(len(allWordsSet)))

    wordsDimension = 200
    verDic  = myWord2Ver.getVer(sentences, wordsDimension  , 1)
    print( "单词维度数目：" +  str(wordsDimension) )

    X_inputVer = []
    for item in sentences:
        # print(item)
        everySentenceVer = np.empty( 0 )
        row = 0
        for word in item:
            everySentenceVer = np.concatenate( ( everySentenceVer, verDic[word] ), axis = 0 )
            row += 1
        everySentenceVer = everySentenceVer.reshape((row , wordsDimension)).sum(axis=0)
        # cc= everySentenceVer.sum(axis=0)
        X_inputVer.append(everySentenceVer.tolist())

    # print( str(X_inputVer) )

    # 划分数据
    X_train,X_test,y_train,y_test=train_test_split(X_inputVer,Y_input,test_size=0.2)#利用train_test_split进行将训练集和测试集进行分开，test_size占20%
    # 模型训练
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150 , 150 , 150 ),
                        random_state=1, verbose = False, early_stopping = True,
                        warm_start = True)
    clf.fit(X_train, y_train)
    print("迭代次数: " + str(clf.n_iter_) )
    print("隐层数: " + str(clf.hidden_layer_sizes) )

    # y_pred = clf.predict(X_test)
    score  = clf.score(X_test, y_test)
    print("R值(准确率) = " + str(score) )


# predictions = clf.predict(X_test)
# precision, recall, threshold = precision_recall_curve(y_true, y_scores)
# from sklearn.metricsimportclassification_report,confusion_matrix

if __name__ == '__main__':
    analysis()
    # test()







