import jieba
from gensim.models import word2vec

filepath = u"corpus.txt"
filedonepath = u'corpuSegDone.txt'

fileTrain = []

with open(filepath) as fileReadRaw:
    for line in fileReadRaw:
        fileTrain.append(line)

def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i])

fileTrainSeg = []
for i in range(len(fileTrain)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrain[i][9:-11], cut_all=False)))])
    if i % 100 == 0:
        print(i)

with open(filedonepath, 'wb') as fileSeg:
    for i in range(len(fileTrainSeg)):
        fileSeg.write(fileTrainSeg[i][0].encode('utf-8'))
        fileSeg.write(('\r').encode('utf-8'))

sentences = word2vec.Text8Corpus(filedonepath)
model = word2vec.Word2Vec(sentences, size=300, hs=1, min_count=1, window=3)
print(model.vector_size)
print(u'与记者相近的词')
keys = model.similar_by_word(u'记者')
for key in keys:
    if(len(key[0])==2):
        print(key[0], key[1])

keys = model.similar_by_vector(u'中国')
for key in keys:
    print(key[0], key[1])

keys = model.most_similar(u'中国')
for key in keys:
    print(key[0], key[1])