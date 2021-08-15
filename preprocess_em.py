import numpy as np
# import Weibo_model
import codecs
import re

rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
vector_em = []
word2id_em = {}
id2word_em = {}
tag_id_em = {}
id_tag_em= {}
word_dim=100
num_steps=80


class Setting(object):
    def __init__(self):
        self.lr=0.001
        self.word_dim=100
        self.lstm_dim=120
        self.num_units=240
        self.num_heads=8
        self.num_steps=80
        self.keep_prob=0.7
        self.keep_prob1=0.7
        self.in_keep_prob=0.7
        self.out_keep_prob=0.6
        self.batch_size=20
        self.clip=5
        self.num_epoches=140
        self.adv_weight=0.06
        self.task_num=2
        self.ner_tags_num=9
        self.cws_tags_num=4


def load_embedding(setting):
    print('reading chinese word embedding.....')
    f = open('./embed.txt','r')
    f.readline()
    while True:
        content=f.readline()
        if content=='':
            break
        else:
            content=content.strip().split()
            if len(content[1:]) != 100:
                continue
            word2id_em[content[0]]=len(word2id_em)
            id2word_em[len(id2word_em)]=content[0]
            content=content[1:]
            # content=[float(i) for i in content]
            vector_em.append(np.asarray(content, dtype=float))
    f.close()
    word2id_em['padding']=len(word2id_em)
    word2id_em['unk']=len(word2id_em)
    vector_em.append(np.zeros(shape=setting.word_dim,dtype=np.float32))
    vector_em.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
    id2word_em[len(id2word_em)]='padding'
    id2word_em[len(id2word_em)]='unk'


def process_train_data(setting):
    print('reading train data.....')
    train_word=[]
    train_label=[]
    train_length=[]
    # f=open('./weiboNER.conll.train','r')
    train_word.append([])
    train_label.append([])
    train_max_len=0
    with open('./data/weiboNER_2nd_conll.train.txt', 'r') as f:
        while True:
            content=f.readline()
            if content=='':
                break
            elif content=='\n':
                length=len(train_word[len(train_word)-1])
                train_length.append(min(length,num_steps))
                if length>train_max_len:
                    train_max_len=length
                train_word.append([])
                train_label.append([])
            else:
                content=content.replace('\n','').replace('\r','').strip().split()
                if content[1]!='O':
                    label1=content[1].split('.')[0]
                    # label2=content[1].split('.')[1]
                    content[1]=label1
                    # if label2=='NOM':
                    #    content[1]='O'
                if content[0][0] not in word2id_em:
                    word2id_em[content[0][0]]=len(word2id_em)
                    vector_em.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
                    id2word_em[len(id2word_em)]=content[0][0]
                if content[1] not in tag_id_em:
                    tag_id_em[content[1]]=len(tag_id_em)
                    id_tag_em[len(id_tag_em)]=content[1]
                train_word[len(train_word)-1].append(word2id_em[content[0][0]])
                train_label[len(train_label)-1].append(tag_id_em[content[1]])

    if len(train_word[len(train_word)-1])!=0:
        train_length.append(min(len(train_word[len(train_word)-1]),num_steps))
    if [] in train_word:
        train_word.remove([])
    if [] in train_label:
        train_label.remove([])

    assert len(train_word)==len(train_label)
    assert len(train_word)==len(train_length)
    for i in range(len(train_word)):
        if len(train_word[i])<num_steps:
            for j in range(num_steps-train_length[i]):
                train_word[i].append(word2id_em['padding'])
                train_label[i].append(tag_id_em['O'])
        else:
            train_word[i]=train_word[i][:num_steps]
            train_label[i]=train_label[i][:num_steps]

    train_word = np.asarray(train_word)
    train_label = np.asarray(train_label)
    train_length = np.asarray(train_length)
    np.save('./data/weibo_train_word_em.npy',train_word)
    # np.save('./data/weibo_train_label.npy',train_label)
    # np.save('./data/weibo_train_length.npy', train_length)


def process_test_data(setting):
    print('reading test data.....')
    test_word=[]
    test_label=[]
    test_length=[]
    # f=open('./data/weiboNER.conll.test','r')
    test_word.append([])
    test_label.append([])
    test_max_len=0
    with open('./data/weiboNER_2nd_conll.test.txt', 'r') as f:
        while True:
            content=f.readline()
            # print(f'content: {content}')
            if content=='':
                break
            elif content=='\n':
                test_length.append(min(len(test_word[len(test_word)-1]),num_steps))
                if len(test_word[len(test_word)-1])>test_max_len:
                    test_max_len=len(test_word[len(test_word)-1])
                test_word.append([])
                test_label.append([])
            else:
                content = content.replace('\n', '').replace('\r', '').strip().split()
                if content[1]!='O':
                    label1=content[1].split('.')[0]
                    # label2=content[1].split('.')[1]
                    content[1]=label1
                    # if label2=='NOM':
                    #    content[1]='O'
                if content[0][0] not in word2id_em:
                    word2id_em[content[0][0]]=len(word2id_em)
                    vector_em.append(vector_em[word2id_em['unk']])
                    id2word_em[len(id2word_em)]=content[0][0]
                if content[1] not in tag_id_em:
                    tag_id_em[content[1]]=len(tag_id_em)
                    id_tag_em[len(id_tag_em)]=content[1]
                test_word[len(test_word)-1].append(word2id_em[content[0][0]])
                test_label[len(test_label)-1].append(tag_id_em[content[1]])

    if len(test_word[len(test_word)-1])!=0:
        test_length.append(len(test_word[len(test_word)-1]))
    if [] in test_word:
        test_word.remove([])
    if [] in test_label:
        test_label.remove([])
    assert len(test_word) == len(test_label)
    assert len(test_word) == len(test_length)
    for i in range(len(test_word)):
        if len(test_word[i]) < num_steps:
            for j in range(num_steps - test_length[i]):
                test_word[i].append(word2id_em['padding'])
                test_label[i].append(tag_id_em['O'])
        else:
            test_word[i]=test_word[i][:num_steps]
            test_label[i]=test_label[i][:num_steps]
    test_word = np.asarray(test_word)
    test_label = np.asarray(test_label)
    test_length = np.asarray(test_length)
    np.save('./data/weibo_test_word_em.npy',test_word)
    # np.save('./data/weibo_test_label.npy',test_label)
    # np.save('./data/weibo_test_length.npy', test_length)


def process_dev_data(setting):
    print('reading dev data.....')
    dev_word=[]
    dev_label=[]
    dev_length=[]
    # f=open('./data/weiboNER.conll.test','r')
    dev_word.append([])
    dev_label.append([])
    dev_max_len=0
    with open('./data/weiboNER_2nd_conll.dev.txt', 'r') as f:
        while True:
            content=f.readline()
            # print(f'content: {content}')
            if content=='':
                break
            elif content == '\n':
                dev_length.append(min(len(dev_word[len(dev_word) - 1]), num_steps))
                if len(dev_word[len(dev_word) - 1]) > dev_max_len:
                    dev_max_len = len(dev_word[len(dev_word) - 1])
                dev_word.append([])
                dev_label.append([])
            else:
                content = content.replace('\n', '').replace('\r', '').strip().split()
                if content[1] != 'O':
                    label1 = content[1].split('.')[0]
                    # label2=content[1].split('.')[1]
                    content[1] = label1
                    # if label2=='NOM':
                    #    content[1]='O'
                if content[0][0] not in word2id_em:
                    word2id_em[content[0][0]] = len(word2id_em)
                    vector_em.append(vector_em[word2id_em['unk']])
                    id2word_em[len(id2word_em)] = content[0][0]
                if content[1] not in tag_id_em:
                    tag_id_em[content[1]] = len(tag_id_em)
                    id_tag_em[len(id_tag_em)] = content[1]
                dev_word[len(dev_word) - 1].append(word2id_em[content[0][0]])
                dev_label[len(dev_label) - 1].append(tag_id_em[content[1]])

    if len(dev_word[len(dev_word)-1])!=0:
        dev_length.append(len(dev_word[len(dev_word)-1]))
    if [] in dev_word:
        dev_word.remove([])
    if [] in dev_label:
        dev_label.remove([])
    assert len(dev_word) == len(dev_label)
    assert len(dev_word) == len(dev_length)
    for i in range(len(dev_word)):
        if len(dev_word[i]) < num_steps:
            for j in range(num_steps - dev_length[i]):
                dev_word[i].append(word2id_em['padding'])
                dev_label[i].append(tag_id_em['O'])
        else:
            dev_word[i]=dev_word[i][:num_steps]
            dev_label[i]=dev_label[i][:num_steps]
    dev_word = np.asarray(dev_word)
    dev_label = np.asarray(dev_label)
    dev_length = np.asarray(dev_length)
    np.save('./data/weibo_dev_word_em.npy',dev_word)
    # np.save('./data/weibo_dev_label.npy',dev_label)
    # np.save('./data/weibo_dev_length.npy', dev_length)


if __name__ == '__main__':
    setting = Setting()
    load_embedding(setting)
    process_train_data(setting)
    process_test_data(setting)
    process_dev_data(setting)

    tag_to_id = open('tag_to_id_em.txt', 'w')
    tag_to_id.write(str(tag_id_em))
    tag_to_id.close()
    id_to_tag = open('id_to_tag_em.txt', 'w')
    id_to_tag.write(str(id_tag_em))
    id_to_tag.close()

    word_to_id = open('word_to_id_em.txt', 'w')
    word_to_id.write(str(word2id_em))
    word_to_id.close()
    id_to_word = open('id_to_word_em.txt', 'w')
    id_to_word.write(str(id2word_em))
    id_to_word.close()

    vector_em = np.asarray(vector_em)
    # print(vector_em.shape)    (16324, 100)
    np.save('./vector_em.npy', vector_em)