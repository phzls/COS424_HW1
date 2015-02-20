import process as pr

DP = pr.Data_Process()

DP.read_data(detail=True)
DP.data_frequency(idf=True)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(DP.train_bag_words_transformed, DP.train_email_class)
predicted = clf.predict(DP.test_bag_words_transformed)

pr.test_result(predicted, DP.test_email_class)