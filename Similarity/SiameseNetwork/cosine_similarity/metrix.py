#!/usr/env/bin python
#coding=utf-8

import os
import sys

threshold = float(sys.argv[1])

def metrix(tp, tn, fp, fn):
    accu = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    f1 = 2*tp / (2*tp + fp + fn)
    prec = tp / (tp + fp)
    print "tp=%d, tn=%d, fp=%d, fn=%d" % (tp, tn, fp, fn)
    print "accu=%.4f, recall=%.4f, prec=%.4f, f1=%.4f" % (accu, recall, prec, f1)

tp = 0.0
fp = 0.0
tn = 0.0
fn = 0.0
sum_neg = 0.0
sum_pos = 0.0
sum_neg_score = 0.0
sum_pos_score = 0.0
wp_tp = open("./analysis/tp.txt", 'w')
wp_fn = open("./analysis/fn.txt", 'w')
wp_fp = open("./analysis/fp.txt", 'w')
wp_tn = open("./analysis/tn.txt", 'w')
for line in sys.stdin:
    splits = line.strip('\r\n').split('\t')
    if len(splits) < 4:
        continue
    query1 = splits[0]
    query2 = splits[1]
    label = int(splits[2])
    score = float(splits[3])
    if label == 1 and score >= threshold:
        tp += 1
        sum_pos += 1
        sum_pos_score += score
        print >> wp_tp, line.strip('\r\n')
    elif label == 1 and score < threshold:
        fn += 1
        sum_pos += 1
        sum_pos_score += score
        print >> wp_fn, line.strip('\r\n')
    elif label == 0 and score >= threshold:
        fp += 1
        sum_neg += 1
        sum_neg_score += score
        print >> wp_fp, line.strip('\r\n')
    elif label == 0 and score < threshold:
        tn += 1
        sum_neg += 1
        sum_neg_score += score
        print >> wp_tn, line.strip('\r\n')
wp_tp.close()
wp_fn.close()
wp_fp.close()
wp_tn.close()
avr_pos_score = 0.0
if sum_pos > 0:
    avr_pos_score = sum_pos_score / sum_pos
avr_neg_score = 0.0
if sum_neg > 0:
    avr_neg_score = sum_neg_score / sum_neg
print "pos_score: %.4f, neg_score: %.4f" % (avr_pos_score, avr_neg_score)
metrix(tp, tn, fp, fn)