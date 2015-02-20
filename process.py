import email_process as ep
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def label_extract(class_label):
    if class_label.find("Not") > -1 or class_label.find("Non") > -1:
        return 0
    else:
        return 1

class Data_Process(object):
    def __init__(self,filename=None):
        if filename == None:
            self.filename = {}
        else:
            self.filename = filename

        self.filename_key = (["train_bag_words","train_class", "train_email","vocab","test_class",
            "test_email","test_bag_words"])
        self.filename_name = (["train_emails_bag_of_words_200.dat","train_emails_classes_200.txt",
            "train_emails_samples_class_200.txt","train_emails_vocab_200.txt",
            "test_emails_classes_0.txt","test_emails_samples_class_0.txt",
            "test_emails_bag_of_words_0.dat"])

        self._filename_construct()

    def _filename_construct(self):
        for i in range(len(self.filename_key)):
            try:
                self.filename[self.filename_key[i]]
            except:
                self.filename[self.filename_key[i]] = self.filename_name[i]


    def read_data(self, detail=False):
        self.train_email_name = np.loadtxt(self.filename["train_email"], dtype=str)
        self.train_email_class = np.loadtxt(self.filename["train_class"], dtype=str)
        self.vocab = np.loadtxt(self.filename["vocab"],dtype=str)

        self.label = {}
        self.label[self.train_email_class[0]] = label_extract(self.train_email_class[0])
        self.label[self.train_email_class[-1]] = label_extract(self.train_email_class[-1])

        self.train_email_class = [self.label[n] for n in self.train_email_class]

        self.test_email_name = np.loadtxt(self.filename["test_email"], dtype=str)
        self.test_email_class = np.loadtxt(self.filename["test_class"], dtype=str)
        self.test_bag_words = ep.read_bagofwords_dat(self.filename["test_bag_words"],
            len(self.test_email_name))

        self.test_email_class = [self.label[n] for n in self.test_email_class]

        if detail:
            print "Total number of emails: ", len(self.train_email_name)
            print "Total number of class labels: ", len(self.train_email_class)
            print "Total number of words: ", len(self.vocab)

        self.train_bag_words = ep.read_bagofwords_dat(self.filename["train_bag_words"],
            len(self.train_email_name))

    def data_frequency(self, idf = False, sublinear_tf = False):
        tfidf_transformer = TfidfTransformer(use_idf = idf, sublinear_tf = sublinear_tf)
        self.train_bag_words_transformed = tfidf_transformer.fit_transform(self.train_bag_words)
        self.test_bag_words_transformed = tfidf_transformer.transform(self.test_bag_words)

def test_result(predicted, test_email_class):
    total_error = 0
    spam_error = 0
    notspam_error = 0
    for n in range(len(predicted)):
        s = predicted[n] - test_email_class[n]
        total_error += abs(s)
        notspam_error += max(s,0)
        spam_error += abs(min(0,s))


    print "Total Number of Wrong Prediction: ", total_error
    print "Total Number of Spam Error: ", spam_error
    print "Total Number of Non-Spam Error: ", notspam_error
    print "Total Number of Test Emails: ", len(test_email_class)
    print "Probability of Error: ", float(total_error)/float(len(test_email_class))
    print "Matching Probability: ", 1 - float(total_error)/float(len(test_email_class))

def time_process(elapse_time):
    from math import floor
    hour = floor(elapse_time / 3600)
    minute = floor((elapse_time - 3600*hour) / 60)
    second = elapse_time - 3600*hour - 60 * minute
    return hour, minute, second

