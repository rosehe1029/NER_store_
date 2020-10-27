import kashgari
from kashgari.tasks.labeling import BiGRU_CRF_Model
from kashgari.embeddings import BERTEmbedding
from kashgari.corpus import ChineseDailyNerCorpus


train_x, train_y = ChineseDailyNerCorpus.load_data('train')
test_x, test_y = ChineseDailyNerCorpus.load_data('test')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')

model =BiGRU_CRF_Model()
model.fit(train_x, train_y, x_validate=valid_x, y_validate=valid_y, epochs=1, batch_size=100)
model.evaluate(test_x, test_y)
