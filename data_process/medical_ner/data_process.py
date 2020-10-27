import json
import numpy as np
import os


def load_data(filename):
    features = []
    labels = []
    f = open(filename, encoding='utf-8')
    medical = json.load(f)
    for medical in medical:
        feature = []
        label = []
        for ner_message in medical[1:]:
            for index, message in enumerate(ner_message[0]):
                feature.append(message)
                label.append(ner_message[1])
        features.append(feature)
        labels.append(label)
    return features, labels


clean_medical_ner_entities, medical_labels = load_data("clean_medical_ner_exist.json")
# if __name__ == "__main__":
#     a = [10, 2, 5, 1, 3, 7, 3]
#     j_sort(a)
print(clean_medical_ner_entities[:10], medical_labels[:10])



if not os.path.exists('random_order'):
    random_order = list(range(len(clean_medical_ner_entities)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('random_order', 'w'), indent=4)
else:
    random_order = json.load(open('random_order'))

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

train_x, train_y = train_data,train_label
test_x, test_y = test_data,test_label
valid_x, valid_y = dev_data,dev_label







