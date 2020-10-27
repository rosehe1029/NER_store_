import kashgari
from kashgari.tasks.labeling import BiGRU_CRF_Model
from kashgari.embeddings import BERTEmbedding

# 语聊格式
train_x = [['Hello', 'world'], ['Hello', 'Kashgari'], ['I', 'love', 'Beijing']]
train_y = [['O', 'O'], ['O', 'B-PER'], ['O', 'B-LOC']]

import json


def load_data(filename):
    features = [[]]
    labels = [[]]
    D = []
    f = open(filename, encoding='utf-8')
    medical = json.load(f)
    for medical in medical:
        medical_text = medical["text"]
        medical_labels = medical["annotations"]
        laster_label = 0
        d = []
        small_features = []
        small_labels = []
        for medical_label in medical_labels:
            begin_label = medical_label["start_offset"]

            d.append([medical_text[laster_label:begin_label], "O"])
            last_label = medical_label["end_offset"]
            d.append([medical_text[begin_label:last_label], medical_label["label"]])
            small_features.append(medical_text[begin_label:last_label])
            small_labels.append(medical_label["label"])
            laster_label = last_label
            # if medical_label["label"] not in labels:
            #     labels.append(medical_label["label"])
        D.append(d)
        features.append(small_features)
        labels.append(small_labels)
    return features, labels


clean_medical_ner_entities, medical_labels = load_data("clean_medical_ner_entities.json")
import numpy as np
import os

if not os.path.exists('random_order.json'):
    random_order = list(range(len(clean_medical_ner_entities)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('random_order.json', 'w'), indent=4)
else:
    random_order = json.load(open('random_order.json'))

# 划分valid
train_data, train_label, dev_data, dev_label, test_data, test_label = [], [], [], [], [], []
for index, message_index in enumerate(random_order):
    if index % 10 != 0 and index % 10 != 1:
        train_data.append(clean_medical_ner_entities[message_index])
        train_label.append(medical_labels[message_index])
    if index % 10 == 0:
        dev_data.append(clean_medical_ner_entities[message_index])
        dev_label.append(medical_labels[message_index])
    if index % 10 == 1:
        test_data.append(clean_medical_ner_entities[message_index])
        test_label.append(medical_labels[message_index])

#bert_embed = BERTEmbedding('wwm',
#                           trainable=True,
#                          task=kashgari.LABELING,
#                           sequence_length=10)
#model = BiGRU_CRF_Model(bert_embed)
model = BiGRU_CRF_Model()
model.fit(train_data, train_label, dev_data, dev_label,epochs=100,batch_size=32)
#model.evaluate(dev_data, dev_label)

