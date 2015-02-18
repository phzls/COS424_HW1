import email_process as ep
import numpy as np

# Any string with "Not" or "Non" in it is assigned with 0; otherwise 1
# In this case, "NotSpam" is 0, "Spam" is 1
def label_extract(class_label):
    if class_label.find("Not") > -1 or class_label.find("Non") > -1:
        return 0
    else:
        return 1

# A dictionary for various filenames
filename = {}
filename["train_bag_words"] = "train_emails_bag_of_words_200.dat"
filename["train_class"] = "train_emails_classes_200.txt"
filename["train_email"] = "train_emails_samples_class_200.txt"
filename["vocab"] = "train_emails_vocab_200.txt"
filename["test_class"] = "test_emails_classes_0.txt"
filename["test_email"] = "test_emails_samples_class_0.txt"
filename["test_bag_words"] = "test_emails_bag_of_words_0.dat"

# Email names for training set
train_email_name = np.loadtxt(filename["train_email"], dtype=str)

# Classes of each email for training set
train_email_class = np.loadtxt(filename["train_class"], dtype=str)

# Vocabulary
vocab = np.loadtxt(filename["vocab"],dtype=str)

print "Total number of emails: ", len(train_email_name)
print "Total number of class labels: ", len(train_email_class)
print "Total number of words: ", len(vocab)

# Bag of words for training set
train_bag_words = ep.read_bagofwords_dat(filename["train_bag_words"], len(train_email_name))

print np.shape(train_bag_words)

# Change "NotSpam" and "Spam" to 0 and 1
label = {}
label[train_email_class[0]] = label_extract(train_email_class[0])
label[train_email_class[-1]] = label_extract(train_email_class[-1])

train_email_class = [label[n] for n in train_email_class]

# Read test data
test_email_name = np.loadtxt(filename["test_email"], dtype=str)
test_email_class = np.loadtxt(filename["test_class"], dtype=str)
test_bag_words = ep.read_bagofwords_dat(filename["test_bag_words"], len(test_email_name))

test_email_class = [label[n] for n in test_email_class]

# Joint prob of word and email being spam
joint_prob_spam = np.zeros(len(vocab))

# Joint prob of word and email being not spam
joint_prob_notspam = np.zeros(len(vocab))

# Check if spam and not spam are organized contiguously.
# If it is, the position where a new type starts is stored in diff.
# Otherwise an exception is raised
start = train_email_class[0]
diff = 0
for index in range(len(train_email_class)):
    if start != train_email_class[index]:
        diff = index
        start = train_email_class[index]

for index in range(diff, len(train_email_class)):
    if start != train_email_class[index]:
        raise Exception("train_email_class not contiguous")

# Record count of words in spam and not spam respectively
if start == 1:
    joint_prob_spam = np.sum(train_bag_words[:diff],axis=0)
    joint_prob_notspam = np.sum(train_bag_words[diff:],axis=0)
else:
    joint_prob_notspam = np.sum(train_bag_words[:diff],axis=0)
    joint_prob_spam = np.sum(train_bag_words[diff:],axis=0)

# Kill zeros
joint_prob_spam = [n+1 for n in joint_prob_spam]
joint_prob_notspam = [n+1 for n in joint_prob_notspam]

# Normalize the joint probability
total_count = np.sum(joint_prob_notspam) + np.sum(joint_prob_spam)

joint_prob_spam = [n/float(total_count) for n in joint_prob_spam]
joint_prob_notspam = [n/float(total_count) for n in joint_prob_notspam]


temp = np.array([joint_prob_notspam, joint_prob_spam])

# The marginal distribution of each word
word_prob = np.sum(temp, axis=0)

# The marginal distribution of each class
class_prob = np.sum(temp, axis=1)

print "Prob of each class: ", class_prob[0], class_prob[1]
print "Total prob of word: ", np.sum(word_prob)
print "Total prob of joint prob: ", np.sum(joint_prob_spam) + np.sum(joint_prob_notspam)

# Compute the ratio inside the logarithm of mutual information
ratio_spam = [joint_prob_spam[i]/(class_prob[1]*word_prob[i]) for i in range(len(word_prob))]
ratio_notspam = [joint_prob_notspam[i]/(class_prob[0]*word_prob[i]) for i in range(len(word_prob))]

# Compute mutual information
from math import log
I = np.zeros(len(word_prob))
for i in range(len(word_prob)):
    I[i] = joint_prob_spam[i]*log(ratio_spam[i]) + joint_prob_notspam[i]*log(ratio_notspam[i])

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

tfidf_transformer = TfidfTransformer()
train_bag_words_tfidf = tfidf_transformer.fit_transform(train_bag_words)

test_bag_words_tfidf = tfidf_transformer.transform(test_bag_words)

num = 20

# Sort mutual information in descending order and take first num terms
word_index_raw = np.argsort(I)[::-1]

list_range = [5,10,20,100,200,1000,5000,9000]

for num in list_range:
    word_index = word_index_raw[:num]

    clf = MultinomialNB().fit(train_bag_words_tfidf[:,word_index], train_email_class)
    predicted = clf.predict(test_bag_words_tfidf[:,word_index])

    total_error = 0
    for n in range(len(predicted)):
        total_error += abs(predicted[n] - test_email_class[n])

    print '\n', "Using " + str(num) + " words with highest MI:"
    print "Total Number of Wrong Prediction: ", total_error
    print "Total Number of Test Emails: ", len(test_email_name)
    print "Probability of Error: ", float(total_error)/float(len(test_email_name))

"""
print '\n',"First " + str(num) +" words of highest MI:"
for i in range(len(word_index)):
    print vocab[word_index[i]], I[word_index[i]]
"""
