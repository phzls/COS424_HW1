import email_process as ep
import numpy as np

def label_extract(class_label):
    if class_label.find("Not") > -1 or class_label.find("Non") > -1:
        return 0
    else:
        return 1

filename = {}
filename["train_bag_words"] = "train_emails_bag_of_words_200.dat"
filename["train_class"] = "train_emails_classes_200.txt"
filename["train_email"] = "train_emails_samples_class_200.txt"
filename["vocab"] = "train_emails_vocab_200.txt"
filename["test_class"] = "test_emails_classes_0.txt"
filename["test_email"] = "test_emails_samples_class_0.txt"
filename["test_bag_words"] = "test_emails_bag_of_words_0.dat"

train_email_name = np.loadtxt(filename["train_email"], dtype=str)
train_email_class = np.loadtxt(filename["train_class"], dtype=str)
vocab = np.loadtxt(filename["vocab"],dtype=str)

print "Total number of emails: ", len(train_email_name)
print "Total number of class labels: ", len(train_email_class)
print "Total number of words: ", len(vocab)

train_bag_words = ep.read_bagofwords_dat(filename["train_bag_words"], len(train_email_name))

print np.shape(train_bag_words)

label = {}
label[train_email_class[0]] = label_extract(train_email_class[0])
label[train_email_class[-1]] = label_extract(train_email_class[-1])

train_email_class = [label[n] for n in train_email_class]

test_email_name = np.loadtxt(filename["test_email"], dtype=str)
test_email_class = np.loadtxt(filename["test_class"], dtype=str)
test_bag_words = ep.read_bagofwords_dat(filename["test_bag_words"], len(test_email_name))

test_email_class = [label[n] for n in test_email_class]

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_bag_words_tfidf = tfidf_transformer.fit_transform(train_bag_words)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_bag_words_tfidf, train_email_class)

test_bag_words_tfidf = tfidf_transformer.transform(test_bag_words)
predicted = clf.predict(test_bag_words_tfidf)

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
print "Total Number of Test Emails: ", len(test_email_name)
print "Probability of Error: ", float(total_error)/float(len(test_email_name))

