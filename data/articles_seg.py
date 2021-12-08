# coding: utf-8

import jieba
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def check_number(string):
    str1 = str(string)
    if str1.count('.') > 1:
        return False
    elif str1.isdigit():
        return True
    else:
        new_str = str1.split('.')
        frist_num = new_str[0]
        if frist_num.count('-') > 1:
            return False
        else:
            frist_num = frist_num.replace('-','')
        if frist_num.isdigit() and new_str[1].isdigit():
            return True
        else:
            return False

def load_stopwords(filename):
    stopwords = []
    with open(filename, 'r') as f:
        stopwords.extend([line.strip() for line in f])
    return stopwords

def read_file_seg(filename, save_file):
    contents, labels = [], []
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if count % 5000 == 0:
                print 'finished: ', count
            count += 1
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append([i for i in jieba.cut(content.strip()) if len(i) > 1 and not check_number(i)])
                    labels.append(label)
            except:
                pass
    savefile = open(save_file, 'w')
    for i in range(len(contents)):
        savefile.write(labels[i] + '\t' + ",".join(contents[i]) + '\n')
    return contents, labels


def build_vocab(contents, _vocab_dir, vocab_size=3000):
    stop_words = load_stopwords('articles/stopwords.txt')
    all_data = []
    count = 0
    for content in contents:
        if count % 100 == 0:
            print 'finished: ', count
        count += 1
        all_data.extend([i for i in content if i not in stop_words])
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open(_vocab_dir, mode='w').write('\n'.join(words) + '\n')

if __name__ == '__main__':
    train_contents, _ = read_file_seg('articles/articles.train.txt', 'articles/articles.train.seg.txt')
    print 'finished train data segment'
    test_contents, _ = read_file_seg('articles/articles.test.txt', 'articles/articles.test.seg.txt')
    print 'finished test data segment'
    val_contents, _ = read_file_seg('articles/articles.val.txt', 'articles/articles.val.seg.txt')
    print 'finished val data segment'
    build_vocab(train_contents, 'articles/articles.vocab.seg.txt')
