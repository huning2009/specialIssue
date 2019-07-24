# load modules
import warnings
warnings.filterwarnings('ignore')
import re
import jieba
import numpy
import logging
import pyLDAvis.gensim
from pprint import pprint
from matplotlib import pyplot as plt
import jieba.posseg as pseg
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models import CoherenceModel, LdaModel

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# re format
pattern_noun = re.compile(r'^n')
pattern_adj = re.compile(r'^a')
pattern_verb = re.compile(r'^v')

# 导入自定义词典
jieba.load_userdict('/Users/lyndon/PycharmProjects/doubanShortComment/corpusAppendix/userdict.txt')

# 构建停用词列表
stopwords = list()
with open('/Users/lyndon/PycharmProjects/doubanShortComment/corpusAppendix/stopwords.txt', 'r', encoding='utf-8') as f:
    for item in f.readlines():
        stopwords.append(item.strip())
stopwords = list(set(stopwords))

# 分词并提取词性：a* 形容词类；n* 名词类；v* 动词类
documents = list()
with open('./allText.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        temp = list()
        words = pseg.cut(line.strip())
        for word, flag in words:
            if (pattern_noun.match(flag) != None) or (pattern_adj.match(flag) != None) or (pattern_verb.match(flag) != None):
                temp.append(word.strip())
        documents.append(' '.join(temp))

# 去除停用词，并筛除单词
texts = [[word for word in document.lower().split() if word not in stopwords and len(word) > 1] for document in documents]
print('\n' * 2, '>>>>>>>>>>筛除词语后的文档示例<<<<<<<<<<')
for i in texts[:5]:
    print(i)

# method 1: 利用词袋模型来提取文本特征
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print('\n' * 2, '>>>>>>>>>>文档词袋表示示例<<<<<<<<<<')
for i in list(corpus)[:5]:
    print(i)

# method 2: 向量表示转换
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]
print('\n' * 2, '>>>>>>>>>>文档tf-idf表示示例<<<<<<<<<<')
for i in corpus_tfidf[:5]:
    print(i)

# # method 3: 继续进行向量转换，从tfidf到lsi (bow--tfidf--lsi)
# lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
# corpus_lsi = lsi[corpus_tfidf]
# for doc in corpus_lsi[:5]:
#     print(doc)
# pprint(lsi.show_topics())
#
# # 层次狄利克雷过程（Hierarchical Dirichlet Process, HDP）
# # 第一遍：20个主题数
# hdp = models.HdpModel(corpus, id2word=dictionary)
# print('\n' * 2, '><' * 20)
# print(hdp)
# for topic in hdp.show_topics():
#     print(topic[0], topic[1])

########## 确定主题数目 ##########
# # 根据CoherenceModel中的两个指标：U_Mass Coherence, C_V Coherence，指标数值越大，说明主题模型效果越好
# numpy.random.seed(723)
# print('\n' * 2, '>>>>>>>>>>依据coherence来进行主题数目确定<<<<<<<<<<')
# for topic_num in range(2, 15):
#     lda_model = LdaModel(corpus=corpus_tfidf, id2word=dictionary, iterations=50, num_topics=topic_num)
#     u_mass = CoherenceModel(model=lda_model, corpus=corpus_tfidf, dictionary=dictionary, coherence='u_mass')
#     c_v = CoherenceModel(model=lda_model, corpus=corpus_tfidf, texts=texts, dictionary=dictionary, coherence='c_v')
#     print('主题数目:', topic_num, ', U_Mass:', u_mass.get_coherence(), ', C_V:', c_v.get_coherence())


# 选定 * 个主题
print('\n' * 2, '>>>>>>>>>>LDA模型<<<<<<<<<<')
pattern_word = re.compile(r'"(.*?)"')
model = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5, iterations=500, alpha='auto')
# print(model.show_topics(num_words=10))
for item in model.show_topics(num_words=10):
    print('主题编号：%s, 关键词为：%s' %(item[0], pattern_word.findall(item[1])))


# 主题模型可视化：1. 每个主题的意义？  2. 每个主题在总语料库的比重？  3. 主题之间的关联？
print('\n' * 2, '>>>>>>>>>>主题模型可视化<<<<<<<<<<')
prepared = pyLDAvis.gensim.prepare(model, corpus_tfidf, dictionary)
pyLDAvis.save_html(prepared, './temp.html')
with open('./lda.html', 'w', encoding='utf-8') as t:
    with open('./temp.html', 'r', encoding='utf-8') as f:
        html_1 = f.read().replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css', '/Users/lyndon/PycharmProjects/infotainment/abstract/LDA/appendix/ldavis.v1.0.0.css')
        html_2 = html_1.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js', '/Users/lyndon/PycharmProjects/infotainment/abstract/LDA/appendix/ldavis.v1.0.0.js')
        t.write(html_2)
