#!/usr/env/bin python
#coding=utf-8

import os
import sys
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt

y = []
scores = []
for line in sys.stdin:
    splits = line.strip('\r\n').split('\t')
    if len(splits) < 4:
        continue
    y.append(int(splits[2]))
    scores.append(float(splits[3]))

y = np.array(y)
scores = np.array(scores)
fprs, tprs, thresholds = metrics.roc_curve(y, scores, pos_label=1)
# print tprs
# print fprs
# print thresholds
auc = metrics.auc(fprs, tprs)
# print auc
# for fpr, tpr in zip(fprs, tprs):
#     print "%.4f\t%.4f" % (fpr, tpr)

plt.plot(fprs, tprs, 'k--', label='Mean ROC (area = %0.4f)' % auc, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()
