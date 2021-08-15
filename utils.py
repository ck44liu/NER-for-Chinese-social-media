import numpy as np
import torch
import ast
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


def get_char_pos(sents, sent_length, id_to_word: dict):
    char_pos = np.full_like(sents, 10, dtype=int)
    # max_index = 0
    for idx in range(sents.shape[0]):
        for i in range(sent_length[idx]):
            char = id_to_word[int(sents[idx, i])]
            digit = int(char[-1]) if str.isdigit(char[-1]) else 0
            char_pos[idx, i] = digit
            # if int(char[1]) > max_index:
            #     max_index = int(char[1])
    return char_pos


def get_trig_emb(batch_size, sent_length):
    sin_emb = np.zeros((batch_size, sent_length))
    cos_emb = np.zeros((batch_size, sent_length))
    for j in range(sent_length):
        sin_emb[:,j] = round(5 * np.sin(np.pi*j/6) + 5, 0)
        cos_emb[:,j] = round(5 * np.cos(np.pi*j/6) + 5, 0)
    return sin_emb, cos_emb


def get_pos_trig_emb(seq_len, hidden_dim):
    d_model = hidden_dim
    emb = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, int(d_model/2)):
            emb[pos, 2*i] = np.sin(pos / (10000 ** (2*i / d_model)))
            emb[pos, 2*i + 1] = np.cos(pos / (10000 ** (2*i / d_model)))
    return torch.FloatTensor(emb)


def token_eval(sent_label, sent_length, pred_label):
    # accu = 0
    TP = 0
    FP = 0
    FN = 0

    for idx in range(sent_label.shape[0]):
        for i in range(sent_length[idx]):
            label = sent_label[idx, i]
            pred = pred_label[idx, i]
            # 'O': 0 in tag_to_id
            if (label != 0) and (label == pred):
                TP += 1
            if (pred != 0) and ((label == 0) or (label != pred)):
                FP += 1
            if (label != 0) and ((pred == 0) or (label != pred)):
                FN += 1

    # accuracy = accu / np.sum(sent_length)
    precision = recall = f1_score = 0

    if TP+FP != 0 and TP+FN != 0:
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        f1_score = 2*precision*recall / (precision+recall)
    print(f"Precision: {precision} ({TP}/{TP+FP}), Recall: {recall} ({TP}/{TP+FN}), F1: {f1_score}")

    return precision, recall, f1_score


def entity_eval(sent_entity, pred_entity):
    # y_true = [['B-LOC', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-GPE', 'I-GPE', 'O']]
    # y_pred = [['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-GPE', 'I-GPE', 'O']]
    # f1 = f1_score(y_true, y_pred)
    # print(f1)
    # print(classification_report(y_true, y_pred))

    f1 = f1_score(sent_entity, pred_entity)
    print(f1)
    print(classification_report(sent_entity, pred_entity, digits=4))
    return f1


if __name__ == '__main__':
    label = np.asarray([[0,0,0,2,3,3,0,0,4,5,0]])
    pred  = np.asarray([[0,0,2,2,3,0,0,0,4,5,0]])
    length = np.asarray([11])

    file_i_t = open('id_to_tag.txt', 'r')
    id_tag = ast.literal_eval(file_i_t.read())
    file_i_t.close()

    # entity_eval(label, length, pred, id_tag)
