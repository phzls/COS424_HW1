import process as pr
import time

start_time = time.time()
DP = pr.Data_Process()

DP.read_data(detail=True)
DP.data_frequency(idf=True)

data_time = time.time() - start_time
hour, minute, second = pr.time_process(data_time)
print 'data process time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

K_range = range(1,11)

err_prob = []
false_pos = []
false_neg = []

from sklearn import neighbors

total_email = float(len(DP.test_email_class))

for K in K_range:
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

    total_err, spam_err, notspam_err = pr.test_result(predicted, DP.test_email_class, print_out = False)

    err_prob.append(float(total_err)/total_email)
    false_pos.append(notspam_err)
    false_neg.append(spam_err)

f = open("KNN_result.txt",'w')
print >> f, "K", "Error_Prob", "False_Pos", "False_Neg"

for i in range(len(K_range)):
    print >> f, K_range[i], err_prob[i], false_pos[i], false_neg[i]

import pylab

# Set fonts for plotting
font = {'weight' : 'normal',
        'size'   : 18}

pylab.rc('font', **font)

pylab.figure(1)
ax=pylab.subplot(111)
ax.set_xlim(xmin = K_range[0] - 1, xmax = K_range[-1]+1)
pylab.plot(K_range, err_prob, linewidth = 0, marker = 'o', markersize = 6)
pylab.xlabel("Number of Words")
pylab.ylabel("Error Prob")
pylab.savefig("Error_prob_knn.pdf", box_inches='tight')

pylab.figure(2)
ax=pylab.subplot(111)
ax.set_xlim(xmin = K_range[0] - 1, xmax = K_range[-1]+1)
pylab.plot(K_range, false_pos, label = "false positive", linewidth = 0, marker = 'o', markersize = 6)
pylab.plot(K_range, false_neg, label = "false negative", linewidth = 0, marker = 's', markersize = 6)
pylab.legend(loc='upper right', ncol=1, prop={'size':15})
pylab.xlabel("Number of Words")
pylab.ylabel("Number of Emails")
pylab.savefig("False_pos_neg_knn.pdf", box_inches='tight')

pylab.show()

