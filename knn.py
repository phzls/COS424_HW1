import process as pr
import time

start_time = time.time()
DP = pr.Data_Process()

DP.read_data(detail=True)
DP.data_frequency(idf=True)

K = 2

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(K)
clf.fit(DP.train_bag_words_transformed, DP.train_email_class)
fit_time = time.time() - start_time


hour, minute, second = pr.time_process(fit_time)

print '\n'
print 'fit time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
predicted = clf.predict(DP.test_bag_words_transformed)

predict_time = time.time() - start_time

hour, minute, second = pr.time_process(predict_time)

print 'Prediction time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "
print '\n'

pr.test_result(predicted, DP.test_email_class)
