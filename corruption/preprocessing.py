# 合并training set
import codecs
import pandas as pd

trainingSet = pd.DataFrame(columns=['content', 'type'])

# 导入“不作为”部分的语料，并加入类型名称
buzuowei = codecs.open('./training_buzuowei.txt', 'r', encoding='utf-8')
idx = 0
for item in buzuowei.readlines():
    trainingSet.loc[idx, 'content'] = item.strip()
    trainingSet.loc[idx, 'type'] = '不作为'
    idx += 1
print(idx)

trainingSet = trainingSet

# 导入“贪腐”部分的语料，并加入类型名称
tanfu = codecs.open('./training_tanfu.txt', 'r', encoding='utf-8')
idx = 20767
for item in tanfu.readlines():
    trainingSet.loc[idx, 'content'] = item.strip()
    trainingSet.loc[idx, 'type'] = '贪腐'
    idx += 1
print(idx)

# 写出到csv中
trainingSet.to_csv('./trainingSet.csv', header=True, sep='|', index=None)

# 20767条不作为训练语料, 41014条贪腐训练语料
# 总计61781
