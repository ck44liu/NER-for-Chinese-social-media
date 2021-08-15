import numpy as np
import ast
import os
import preprocess_em
import preprocess_update
import utils
import model_update
import model_combined
import model_transformer
import random
import torch
from torch import nn

rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
vector = []
word2id = {}
id2word = {}
tag_id = {}
id_tag = {}
word_dim = 100
num_steps = 80

if __name__ == "__main__":
    np.random.seed(0)
    print('read word embedding......')
    # embedding = np.load('./data/weibo_vector.npy')
    setting = preprocess_update.Setting()

    print('read ner train data......')
    train_word = np.load('./data/weibo_train_word.npy')
    train_label = np.load('./data/weibo_train_label.npy')
    train_length = np.load('./data/weibo_train_length.npy')
    print('read ner test data......')
    test_word = np.load('./data/weibo_test_word.npy')
    test_label = np.load('./data/weibo_test_label.npy')
    test_length = np.load('./data/weibo_test_length.npy')
    print('read ner dev data......')
    dev_word = np.load('./data/weibo_dev_word.npy')
    dev_label = np.load('./data/weibo_dev_label.npy')
    dev_length = np.load('./data/weibo_dev_length.npy')

    print('read tag<->id dictionaries......')
    file_t_i = open('tag_to_id.txt', 'r')
    tag_id = ast.literal_eval(file_t_i.read())
    file_t_i.close()
    file_i_t = open('id_to_tag.txt', 'r')
    id_tag = ast.literal_eval(file_i_t.read())
    file_i_t.close()

    print('read word<->id dictionaries......')
    file_w_i = open('word_to_id.txt', 'r')
    word2id = ast.literal_eval(file_w_i.read())
    file_w_i.close()
    file_i_w = open('id_to_word.txt', 'r')
    id2word = ast.literal_eval(file_i_w.read())
    file_i_w.close()

    print('read embedding vector......')
    vector = np.load('./vector.npy')
    vector = vector.tolist()

    #------#
    print('read ner train_em data......')
    train_word_em = np.load('./data/weibo_train_word_em.npy')
    print('read ner test_em data......')
    test_word_em = np.load('./data/weibo_test_word_em.npy')
    print('read ner dev_em data......')
    dev_word_em = np.load('./data/weibo_dev_word_em.npy')

    print('read word<->id em dictionaries......')
    file_w_i = open('word_to_id_em.txt', 'r')
    word2id_em = ast.literal_eval(file_w_i.read())
    file_w_i.close()
    file_i_w = open('id_to_word_em.txt', 'r')
    id2word_em = ast.literal_eval(file_i_w.read())
    file_i_w.close()

    print('read embedding em vector......')
    vector_em = np.load('./vector_em.npy')
    vector_em = vector_em.tolist()

    print(train_word.shape)
    print(train_length[30:40])
    print(train_label.shape)
    print(len(word2id))  # 15701 (wiki_100 is 16691 from start)
                         # 24097 in new version

    tag_id_size = len(tag_id)
    print(tag_id)
    print(id_tag)
    # print(id2word[15504])    # padding
    # print(vector[15504])    # [0]*100
    # print(id2word[15505])    # unk
    # print(vector[15505])    # random 100-dim vector

    train = True
    # train = False
    if train:
        bilstm_crf = model_combined.model_train(train_word, train_word_em, train_length, train_label,
                                                dev_word, dev_word_em, dev_length, dev_label,
                                                test_word, test_word_em, test_length, test_label,
                                                setting, tag_id, id_tag, vector, vector_em, id2word)
        torch.save(bilstm_crf, './models/bilstm_crf_attn_v2.pth')
    else:
        bilstm_crf = torch.load('./models/bilstm_crf_attn_v2.pth')

    bilstm_crf.eval()

    print('start train prediction......')
    train_mask = torch.zeros(train_label.shape[0], train_label.shape[1], dtype=torch.uint8)  # (270, 80)
    for i in range(train_label.shape[0]):
        train_mask[i, 0:train_length[i]] = 1

    pred_label = []
    for i in range(train_label.shape[0]):
        sentence = torch.IntTensor(train_word[i].reshape(1, -1))
        sent_em = torch.IntTensor(train_word_em[i].reshape(1, -1))
        mask = train_mask[i, :].reshape(1, -1)
        pred = bilstm_crf.forward(sentence, sent_em, train_length[i], mask)
        pred_label.append(pred[0])

    sent_entity = [['O' for i in range(train_label.shape[1])] for j in range(train_label.shape[0])]
    pred_entity = [['O' for i in range(train_label.shape[1])] for j in range(train_label.shape[0])]

    for idx in range(train_label.shape[0]):
        for i in range(train_length[idx]):
            sent_entity[idx][i] = id_tag[train_label[idx, i]]
            pred_entity[idx][i] = id_tag[pred_label[idx][i]]

    print('start evaluation......')

    # for i in range(train_label.shape[0]):
    #     if i % 50 == 0:
    #         print(sent_entity[i])
    #         print(pred_entity[i])
    #         print()

    utils.entity_eval(sent_entity, pred_entity)

    ###
    print('start dev prediction......')
    dev_mask = torch.zeros(dev_label.shape[0], dev_label.shape[1], dtype=torch.uint8)  # (270, 80)

    for i in range(dev_label.shape[0]):
        dev_mask[i, 0:dev_length[i]] = 1

    pred_label = []
    for i in range(dev_label.shape[0]):
        sentence = torch.IntTensor(dev_word[i].reshape(1, -1))
        dev_em = torch.IntTensor(dev_word_em[i].reshape(1, -1))
        mask = dev_mask[i, :].reshape(1, -1)
        pred = bilstm_crf.forward(sentence, dev_em, dev_length[i], mask)
        pred_label.append(pred[0])

    sent_entity = [['O' for i in range(dev_label.shape[1])] for j in range(dev_label.shape[0])]
    pred_entity = [['O' for i in range(dev_label.shape[1])] for j in range(dev_label.shape[0])]

    for idx in range(dev_label.shape[0]):
        for i in range(dev_length[idx]):
            sent_entity[idx][i] = id_tag[dev_label[idx, i]]
            pred_entity[idx][i] = id_tag[pred_label[idx][i]]

    print('start evaluation......')

    for i in range(dev_label.shape[0]):
        if i % 20 == 0:
            print(sent_entity[i])
            print(pred_entity[i])
            print()

    utils.entity_eval(sent_entity, pred_entity)
