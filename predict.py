# coding: utf-8

from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from lstm_model import TLSTMConfig, TextLSTM
from data.articles_loader import read_category, read_vocab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    bool(type(unicode))
except NameError:
    unicode = str
base_dir = 'data/articles'
vocab_dir = os.path.join(base_dir, 'articles.vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')
save_rnn_dir = 'checkpoints/textrnn'
save_rnn_path = os.path.join(save_rnn_dir, 'best_validation')
save_lstm_dir = 'checkpoints/textlstm'
save_lstm_path = os.path.join(save_lstm_dir, 'best_validation')

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)

    def predict(self, message):
        hqbc = percent = 0.0
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        y_softmax = self.session.run(tf.nn.softmax(self.model.logits), feed_dict=feed_dict)
        for index, item in enumerate(y_softmax[0]):
            #print("{}, {:.5%}".format(self.categories[index], item))
            if(item>0.85): 
                hqbc = 1
                percent = item
        if(hqbc==1): return self.categories[y_pred_cls[0]], "{:.5%}".format(percent)
        else: return "未分類", "BAD DATA"

class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=self.session, save_path=save_rnn_path)

    def predict(self, message):
        hqbc = 0
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 0.5
        }
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        y_softmax = self.session.run(tf.nn.softmax(self.model.logits), feed_dict=feed_dict)
        for index, item in enumerate(y_softmax[0]):
            #print("{}, {:.5%}".format(self.categories[index], item))
            if(item>0.85):
                hqbc = 1
                percent = item
        if(hqbc==1): return self.categories[y_pred_cls[0]], "{:.5%}".format(percent)
        else: return "未分類", "BAD DATA"

class LstmModel:
    def __init__(self):
        self.config = TLSTMConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextLSTM(self.config)
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=self.session, save_path=save_lstm_path)

    def predict(self, message):
        hqbc = 0
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 0.5
        }
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        y_softmax = self.session.run(tf.nn.softmax(self.model.logits), feed_dict=feed_dict)
        for index, item in enumerate(y_softmax[0]):
            #print("{}, {:.5%}".format(self.categories[index], item))
            if(item>0.85):
                hqbc = 1
                percent = item
        if(hqbc==1): return self.categories[y_pred_cls[0]], "{:.5%}".format(percent)
        else: return "未分類", "BAD DATA"

if __name__ == '__main__':
    cnn_model = CnnModel()
    texts = ['通告一則，關於為填補二等高級技術員（機電工程範疇）一缺的統一管理制度普通的專業或職務能力評估對外開考。', '公告一則，關於張貼為填補二等高級技術員（法律範疇）一缺的統一管理制度的專業或職務能力評估對外開考的投考人的知識考試（筆試）成績名單。', '為填補二等高級技術員（機械及材料範疇）一缺的統一管理制度之專業或職務能力評估對外開考的投考人最後成績名單。', '公告一則，關於張貼為填補一般行政技術輔助範疇二等技術輔導員三缺的統一管理制度的專業或職務能力評估對外開考的投考人臨時名單。', '公告一則，關於張貼為填補醫院職務範疇（病理解剖科）主治醫生兩缺對外開考的投考人確定名單。', '公告一則，關於張貼為填補一級護士一缺對外開考經更正的確定名單及知識考試（筆試）准考人名單。', '公告一則，關於張貼為填補醫院職務範疇（放射科及影像學科）主治醫生四缺對外開考的投考人臨時名單。', '公告一則，關於張貼為填補醫院職務範疇（病理解剖科）主治醫生兩缺對外開考的投考人臨時名單。', '通告一則，關於為填補二等地形測量員兩缺的統一管理制度的專業或職務能力評估對外開考的准考人知識考試（筆試）舉行日期、時間及地方的更改。', '通告一則，關於為填補二等高級技術員（水務技術範疇）一缺的統一管理制度的專業或職務能力評估對外開考准考人甄選面試的舉行日期、時間和地點。', '通告一則，關於為填補一級護士一缺對外開考的投考人知識考試（筆試）舉行日期、時間及地點。', '通告一則，關於為填補二等高級技術員（機電工程範疇）一缺的統一管理制度的普通的專業或職務能力評估對外開考。', '公告一則，關於張貼為填補醫院職務範疇（內科）主治醫生一缺對外開考的投考人臨時名單。', '公告一則，關於張貼為填補重型車輛司機八缺的統一管理制度之專業或職務能力評估對外開考的知識考試（駕駛實踐考試）成績名單。', '通告一則，關於為填補二等法證高級技術員（物證範疇）五缺對外開考的知識考試（筆試）的舉行日期、時間和地點。', '通告一則，關於為填補二等高級技術員（土木工程範疇）四缺的統一管理制度的專業或職務能力評估對外開考知識考試（筆試）的舉行日期、時間和地點。', '通告一則，關於為填補建築範疇二等高級技術員一缺的統一管理制度的專業或職務能力評估對外開考知識考試（筆試）的舉行日期、時間和地點。', '公告一則，關於張貼為填補醫院職務範疇（兒科）主治醫生四缺對外開考的投考人臨時名單。', '為填補二等高級技術員（機電工程範疇）兩缺的統一管理制度的專業或職務能力評估對外開考的投考人最後成績名單。', '通告一則，關於為填補二等高級技術員（機電工程範疇）三缺的統一管理制度的普通的專業或職務能力評估對外開考。', '公告一則，關於張貼為填補醫院職務範疇（整形外科）主治醫生一缺對外開考的投考人臨時名單。']
    for x in texts:
        #text = '公告一則，關於張貼為填補醫院職務範疇（病理解剖科）主治醫生兩缺對外開考的投考人確定名單。'
        result = cnn_model.predict(x)
        print("Text: " + x)
        print("Result: " + result[0])
