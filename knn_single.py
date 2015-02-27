import process as pr
import time

K = 3 # Number of Nearest Neighbor

start_time = time.time()
DP = pr.Data_Process()

DP.read_data(detail=True)
DP.data_frequency(idf=True)

data_time = time.time() - start_time
hour, minute, second = pr.time_process(data_time)
print 'data process time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

from sklearn import neighbors

total_email = float(len(DP.test_email_class))

start_time = time.time()
clf = neighbors.KNeighborsClassifier(K)
clf.fit(DP.train_bag_words_transformed, DP.train_email_class)
fit_time = time.time() - start_time

hour, minute, second = pr.time_process(fit_time)

print '\n'
print "K: ", K
print 'fit time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
predicted = clf.predict(DP.test_bag_words_transformed)

predict_time = time.time() - start_time

hour, minute, second = pr.time_process(predict_time)

print 'Prediction time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "
print '\n'

pr.test_result(predicted, DP.test_email_class)

proba = clf.predict_proba(DP.test_bag_words_transformed)

filename = "KNN_K_" + str(K) + "_"
DP.ROC_curve(proba[:,1], plot_show=False, filename=filename + "ROC")

DP.PRC_curve(proba[:,1], filename=filename + "PRC", xmax = 1.01)

