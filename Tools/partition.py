import sys
import numpy as np
import random

num = int(sys.argv[1])
pos_texts = {} 
neg_texts = {}
cnt = 0
for line in open('total_final.txt', 'r'):
    cnt += 1
    if cnt == 1:
        continue
    if float(line.split('\t')[2]) == 1.0:
        pos_texts[line] = 1
    else:
        neg_texts[line] = 1

pos = random.sample(pos_texts.keys(), num)
neg = random.sample(neg_texts.keys(), num)

r = open('data/test.txt', 'w')
for line in pos:
    r.write(line)
for line in neg:
    r.write(line)
r.close()

# samples in test should not exist in train and valid
testset = {}
for line in open('data/test.txt', 'r'):
    testset[line.split('\t')[0] + '\t' + line.split('\t')[1]] = 1

VALIDATION_SPLIT = 0.15
texts = []
cnt = 0
for line in open("total_final.txt"):
    cnt += 1
    if cnt == 1:
        continue
    if line.split('\t')[0] + '\t' + line.split('\t')[1] not in testset:
        texts.append(line)

perm = np.random.permutation(len(texts))
val_domain = int(len(texts)*VALIDATION_SPLIT)
idx_train = perm[val_domain:]
idx_val = perm[:val_domain]
r1 = open('data/train.txt', 'w')
r2 = open('data/valid.txt', 'w')
for idx in idx_train:
    r1.write(texts[idx])
for idx in idx_val:
    r2.write(texts[idx])
r1.close()
r2.close()
