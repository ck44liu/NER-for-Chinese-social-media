import numpy as np
import os
import preprocess_em
import preprocess_update
import utils
import torch
from torch import nn
from torchcrf import CRF


class Bilstm_crf(nn.Module):
    def __init__(self, vocab_size, tag_to_id, id_to_tag, embedding_dim, hidden_dim,
                 pretrained_weight1, pretrained_weight2, batch_size, drop_out, num_steps,
                 id2word):
        super(Bilstm_crf, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_id = tag_to_id
        self.id_to_tag = id_to_tag
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = len(tag_to_id)
        self.batch_size = batch_size
        self.drop0 = nn.Dropout(p=drop_out/2)
        self.drop = nn.Dropout(p=drop_out)
        # self.relu = nn.ReLU()
        self.num_steps = num_steps
        self.id2word = id2word
        # self.batch_norm1 = nn.BatchNorm1d(num_features=120)
        # self.batch_norm2 = nn.BatchNorm1d(num_features=120)
        # self.batch_norm3 = nn.BatchNorm1d(num_features=120)
        self.batch_norm0 = nn.BatchNorm1d(num_features=270)
        self.batch_norm1 = nn.BatchNorm1d(num_features=512)
        # self.start_tag = '<START>'

        self.word_embeds1 = nn.Embedding.from_pretrained(pretrained_weight1, freeze=False)
        # self.word_embeds1.weight.requires_grad = True
        self.second_word_embeds1 = nn.Embedding.from_pretrained(pretrained_weight1, freeze=False)

        self.word_embeds2 = nn.Embedding.from_pretrained(pretrained_weight2, freeze=False)
        # self.word_embeds2.weight.requires_grad = True
        self.second_word_embeds2 = nn.Embedding.from_pretrained(pretrained_weight1, freeze=False)

        self.pos_embeds = nn.Embedding(num_embeddings=num_steps, embedding_dim=40)
        self.second_pos_embeds = nn.Embedding(num_embeddings=num_steps, embedding_dim=40)

        self.char_pos_embeds = nn.Embedding(num_embeddings=11, embedding_dim=30,
                                            padding_idx=10)
        self.second_char_pos_embeds = nn.Embedding(num_embeddings=11, embedding_dim=30,
                                                   padding_idx=10)

        # self.sin_embeds = nn.Embedding(num_embeddings=11, embedding_dim=25)
        # self.cos_embeds = nn.Embedding(num_embeddings=11, embedding_dim=25)

        # transformer-like position embedding
        self.trig_pos = utils.get_pos_trig_emb(num_steps, 200)
        self.trig_pos_embeds = nn.Embedding.from_pretrained(self.trig_pos, freeze=True)

        self.conv0 = nn.Conv1d(in_channels=embedding_dim + 270, out_channels=256,
                               kernel_size=(9,), padding=(4,))
        self.conv1 = nn.Conv1d(in_channels=embedding_dim + 270, out_channels=256,
                               kernel_size=(7,), padding=(3,))

        # self.conv2 = nn.Conv1d(in_channels=250, out_channels=250,
        #                        kernel_size=(9,), padding=(4,))

        self.rnn1 = nn.LSTM(input_size=embedding_dim + 70, hidden_size=200, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.multi_attn1 = nn.MultiheadAttention(embed_dim=(200+hidden_dim) * 2, num_heads=10, dropout=drop_out)
        self.multi_attn2 = nn.MultiheadAttention(embed_dim=(200+hidden_dim) * 2, num_heads=10, dropout=drop_out)

        # self.hidden1_2 = nn.Linear(hidden_dim*2, hidden_dim)
        # self.hidden2_tag = nn.Linear(hidden_dim, self.target_size)
        # self.attn1_tag = nn.Linear(hidden_dim * 2, self.label_size)
        # self.attn2_tag = nn.Linear(hidden_dim * 2, self.label_size)
        # self.rnn_tag = nn.Linear(hidden_dim * 2, self.label_size)
        self.batch_norm_ffnn = nn.BatchNorm1d(num_features=(200+hidden_dim) * 6)
        self.ffnn_tag3 = nn.Linear((200+hidden_dim) * 6, self.label_size * 9)
        self.tag3_tag = nn.Linear(self.label_size * 9, self.label_size)

        self.transitions = nn.Parameter(torch.full((self.label_size, self.label_size), -4.))
        # add initialized hard-coded transition values if needed
        for i in range(self.label_size):
            for j in range(self.label_size):
                if (i == 0 and j in [0, 1, 3, 4, 6]) \
                        or (i in [1, 2] and j in [0, 2]) \
                        or (i in [3, 8] and j in [0, 8]) \
                        or (i in [4, 5] and j in [0, 5]) \
                        or (i in [6, 7] and j in [0, 7]):
                    self.transitions.data[i, j] = 0.
                elif (i in [1, 3, 4, 6] and j in [1, 3, 4, 6]) \
                        or (i in [2, 5, 7, 8] and j in [1, 3, 4, 6]):
                    self.transitions.data[i, j] = 0.

        # self.start_trans = nn.Parameter(torch.full((self.label_size,), -100.))
        # for i in range(self.label_size):
        #     if i in [3,5,7,8]:
        #         self.start_trans.data[i] = 0.

        self.crf = CRF(num_tags=self.label_size, batch_first=True)
        self.crf.start_transitions = nn.Parameter(torch.zeros(self.label_size, ))
        self.crf.end_transitions = nn.Parameter(torch.zeros(self.label_size, ))
        self.crf.transitions = self.transitions

    def _get_lstm_attn_features(self, sentences, sent_em, lengths):
        embeds1 = self.word_embeds1(sentences)  # (batch, 80, 100)
        embeds2 = self.word_embeds2(sent_em)

        second_embeds1 = self.second_word_embeds1(sentences)
        second_embeds2 = self.second_word_embeds2(sent_em)

        pos = torch.IntTensor(np.asarray([np.arange(0, 80, 1)] * embeds1.shape[0]))
        pos_embeds = self.pos_embeds(pos)
        second_pos_embeds = self.second_pos_embeds(pos)

        trig_embeds = self.trig_pos_embeds(pos)

        char_idx = utils.get_char_pos(sentences, lengths, self.id2word)
        char_embeds = self.char_pos_embeds(torch.IntTensor(char_idx))
        second_char_embeds = self.second_char_pos_embeds(torch.IntTensor(char_idx))

        # sin_enc, cos_enc = utils.get_trig_emb(embeds1.shape[0], embeds1.shape[1])
        # sin_embeds = self.sin_embeds(torch.IntTensor(sin_enc))
        # cos_embeds = self.sin_embeds(torch.IntTensor(cos_enc))

        # print(f"pos embed shape: {pos_embeds.shape}")
        # pos_input = torch.transpose(pos_embeds, 0, 1)

        # input = torch.transpose(embeds, 0, 1)  # (80, batch, 100)
        # input = self.drop(input)

        # concat_input = torch.cat((pos_input, input), dim=2)
        # lstm_out, _ = self.lstm(self.drop(concat_input))  # lstm_out: (80, batch, 240)

        concat_embed0 = torch.cat((second_pos_embeds, second_char_embeds, second_embeds1, second_embeds2), dim=2)
        concat_embed0 = torch.transpose(concat_embed0, 1, 2)  # (batch, 320, 80)
        rnn_input1 = self.batch_norm0(concat_embed0)
        rnn_input1 = rnn_input1.permute(2, 0, 1)
        rnn_out1, _ = self.rnn1(self.drop0(rnn_input1))
        rnn_out1 = self.drop(rnn_out1)

        concat_embed1 = torch.cat((pos_embeds, trig_embeds, char_embeds,
                                  embeds1, embeds2), dim=2)
        # print(concat_embed.shape)
        concat_embed1 = torch.transpose(concat_embed1, 1, 2)    # (batch, 320, 80)

        concat_input0 = self.conv0(concat_embed1)
        concat_input1 = self.conv1(concat_embed1)

        # each one is (batch, 120, 80)
        # first concat, then batch_norm, permute, and drop
        concat_input = self.batch_norm1(torch.cat((concat_input0, concat_input1), dim=1))
        concat_input = concat_input.permute(2, 0, 1)
        # rnn_out1, _ = self.rnn(self.drop(concat_input))   # rnn_out: (80, batch, 600)
        # rnn_out1 = self.drop(rnn_out1)

        rnn_out2, _ = self.rnn2(self.drop(concat_input))  # rnn_out: (80, batch, 600)
        rnn_out2 = self.drop(rnn_out2)

        rnn_out = torch.cat((rnn_out1, rnn_out2), dim=2)
        padding_mask = torch.zeros(rnn_out.shape[1], rnn_out.shape[0], dtype=torch.int)  # (batch, 80)
        for i in range(rnn_out.shape[1]):
            if lengths[i] < rnn_out.shape[0]:
                padding_mask[i, lengths[i]:] = 1

        # Q, K, V: lstm_out
        attn_output1, _ = self.multi_attn1.forward(rnn_out, rnn_out, rnn_out, key_padding_mask=padding_mask)

        # the batch norm layer expects an input in [batch_size, features, temp.dim],
        # which is [batch_size, features, 80]

        attn_output2, _ = self.multi_attn2.forward(attn_output1, attn_output1, attn_output1, key_padding_mask=padding_mask)

        # output = self.hidden1_2(self.drop(attn_output))    # output: (80, batch, 120)
        # features = self.hidden2_tag(self.drop(output))    # features: (80, batch, tag_size)
        # feature1 = self.attn1_tag(attn_output1)
        # feature2 = self.attn2_tag(attn_output2)
        # rnn_feats = self.rnn_tag(rnn_out)

        features = torch.cat((attn_output1, attn_output2, rnn_out), dim=2)
        features = self.batch_norm_ffnn(features.permute(1, 2, 0))
        features = self.ffnn_tag3(features.permute(2, 0, 1))

        # features = torch.cat((feature1, feature2, rnn_feats), dim=2)
        # print(features.shape)    # (80, batch_size, 3 * tag_size)
        features = self.tag3_tag(features)
        return features

    def neg_log_likelihood(self, sentences, sent_em, tags, lengths, mask):
        feats = self._get_lstm_attn_features(sentences, sent_em, lengths)

        # feats: (80, batch_size, tag_size)
        # tags: (batch_size, 80)
        feats = torch.transpose(feats, 0, 1)

        ll_loss = self.crf.forward(emissions=feats, tags=tags, mask=mask)
        return -ll_loss

    def forward(self, sentence, sent_em, length, mask):  # don't confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        features = self._get_lstm_attn_features(sentence, sent_em, [length])    # (80, 1, tag_size)
        feats = torch.transpose(features, 0, 1)

        # Find the best path, given the features.
        tag_seq = self.crf.decode(emissions=feats, mask=mask)
        return tag_seq


# training function
def model_train(train_word, train_word_em, train_length, train_label, dev_word, dev_word_em, dev_length, dev_label,
                test_word, test_word_em, test_length, test_label, setting: preprocess_update.Setting, tag_id, id_tag,
                vector, vector_em, id_to_word) -> Bilstm_crf:
    pretrained_weight1 = torch.FloatTensor(vector)
    pretrained_weight2 = torch.FloatTensor(vector_em)
    batch_size = 8

    model = Bilstm_crf(vocab_size=len(vector), tag_to_id=tag_id, id_to_tag=id_tag,
                       embedding_dim=2*len(vector[0]), hidden_dim=400,
                       pretrained_weight1=pretrained_weight1, pretrained_weight2=pretrained_weight2,
                       batch_size=batch_size, drop_out=0.35, num_steps=setting.num_steps,
                       id2word=id_to_word)

    seg_length = setting.num_steps
    train_mask = torch.zeros(train_word.shape[0], seg_length, dtype=torch.uint8)    # (1350, 80)
    for i in range(train_word.shape[0]):
        train_mask[i, 0:train_length[i]] = 1

    dev_mask = torch.zeros(dev_word.shape[0], seg_length, dtype=torch.uint8)  # (270, 80)
    for i in range(test_word.shape[0]):
        dev_mask[i, 0:dev_length[i]] = 1

    test_mask = torch.zeros(test_word.shape[0], seg_length, dtype=torch.uint8)  # (270, 80)
    for i in range(test_word.shape[0]):
        test_mask[i, 0:test_length[i]] = 1

    # model.train()
    # torch.autograd.set_detect_anomaly(True)
    learning_rate = 0.0005
    # learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 45

    for epoch in range(epochs):
        model.train()
        torch.manual_seed(epoch)
        shuffle = torch.randperm(train_word.shape[0])
        total_loss = 0
        for i in range(train_word.shape[0] // batch_size):  # range(54)
            # print(f"pre-shape: {pretrained_weight.shape}")
            model.zero_grad()
            sentences = torch.LongTensor(train_word[shuffle[i * batch_size:(i + 1) * batch_size], :])  # (batch, 80)
            sent_em = torch.LongTensor(train_word_em[shuffle[i * batch_size:(i + 1) * batch_size], :])
            # print(f"sentences: {sentences}")
            tags = torch.LongTensor(train_label[shuffle[i * batch_size:(i + 1) * batch_size], :])  # (batch, 80)
            lengths = train_length[shuffle[i * batch_size:(i + 1) * batch_size]]  # (batch)
            mask = train_mask[shuffle[i * batch_size:(i + 1) * batch_size], :]

            loss = model.neg_log_likelihood(sentences, sent_em, tags, lengths, mask) / batch_size
            # gradient/loss clipping
            # if loss > 100: loss /= 200
            total_loss += loss * batch_size

            if i % 20 == 0:
                print(f"loss at epoch {epoch} batch {i} is {loss}")
            loss.backward()
            optimizer.step()

        # one more step if training size is not divisible by batch size
        if train_word.shape[0] % batch_size != 0:
            last_idx = (train_word.shape[0] // batch_size) * batch_size
            model.zero_grad()
            sentences = torch.LongTensor(train_word[shuffle[last_idx:], :])  # (remained, 80)
            sent_em = torch.LongTensor(train_word_em[shuffle[last_idx:], :])
            tags = torch.LongTensor(train_label[shuffle[last_idx:], :])  # (remained, 80)
            lengths = train_length[shuffle[last_idx:]]  # (remained)
            mask = train_mask[shuffle[last_idx:], :]

            loss = model.neg_log_likelihood(sentences, sent_em, tags, lengths, mask) / (train_word.shape[0] - last_idx)
            total_loss += loss * (train_word.shape[0] - last_idx)
            loss.backward()
            optimizer.step()

        print(f"total loss after epoch {epoch} is {total_loss}")

        model.eval()
        print(f"start dev evaluate after epoch {epoch}: ")
        pred_label = []
        for i in range(dev_label.shape[0]):
            sentence = torch.IntTensor(dev_word[i].reshape(1, -1))
            sent_em = torch.IntTensor(dev_word_em[i].reshape(1, -1))
            mask = dev_mask[i, :].reshape(1, -1)
            pred = model.forward(sentence, sent_em, dev_length[i], mask)
            pred_label.append(pred[0])

        sent_entity = [['O' for i in range(dev_label.shape[1])] for j in range(dev_label.shape[0])]
        pred_entity = [['O' for i in range(dev_label.shape[1])] for j in range(dev_label.shape[0])]

        for idx in range(dev_label.shape[0]):
            for i in range(dev_length[idx]):
                sent_entity[idx][i] = id_tag[dev_label[idx, i]]
                pred_entity[idx][i] = id_tag[pred_label[idx][i]]

        # print('start evaluation......')
        utils.entity_eval(sent_entity, pred_entity)

        print(f"start test evaluate after epoch {epoch}: ")
        model.eval()
        pred_label = []
        for i in range(test_label.shape[0]):
            sentence = torch.IntTensor(test_word[i].reshape(1, -1))
            sent_em = torch.IntTensor(test_word_em[i].reshape(1, -1))
            mask = test_mask[i, :].reshape(1, -1)
            pred = model.forward(sentence, sent_em, test_length[i], mask)
            pred_label.append(pred[0])

        sent_entity = [['O' for i in range(test_label.shape[1])] for j in range(test_label.shape[0])]
        pred_entity = [['O' for i in range(test_label.shape[1])] for j in range(test_label.shape[0])]

        for idx in range(test_label.shape[0]):
            for i in range(test_length[idx]):
                sent_entity[idx][i] = id_tag[test_label[idx, i]]
                pred_entity[idx][i] = id_tag[pred_label[idx][i]]

        # print('start evaluation......')
        utils.entity_eval(sent_entity, pred_entity)

    return model
