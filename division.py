import random

import numpy as np
import pandas as pd

seed = 1
random.seed(seed)
np.random.seed(seed)



t2truth = {}
f_open = open('./datasets1500/total_data/KT1_truth.txt', 'r')
reader = f_open.readlines()
reader = [line.strip("\n") for line in reader]
for line in reader:
    task, truth = line.split('\t')
    t2truth[task] = truth
print(t2truth)
f_open.close()


w2t = {}
f_open = open('./datasets1500/total_data/KT1_crowd.txt', 'r')
reader = f_open.readlines()
reader = [line.strip("\n") for line in reader]
for line in reader:
    task, worker, label = line.split('\t')
    worker = int(worker)
    if worker not in w2t:
        w2t[worker] = {}
    w2t[worker][task] = label
print(len(w2t))
print(w2t)
print("-------------------")
for i in range(len(w2t)):
    item = list(w2t[i].items())
    random.shuffle(item)
    w2t[i] = item

print(w2t)

w2t_train = w2t.copy()
w2t_test = w2t.copy()



for i in range(len(w2t_train)):
    w2t_train[i] = w2t_train[i][0: int(len(w2t_train[i])/2)]

print(w2t_train)


f_save = open('./datasets1500/train_data/KT1_crowd_train.txt', 'w')
for worker, t2l in w2t_train.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        label = t2l[k][1]
        f_save.write(str(task_id) + '\t' + str(worker) + '\t' + str(label) + '\n')
f_save.close()


task_train = []
for worker, t2l in w2t_train.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        if task_id not in task_train:
            task_train.append(task_id)

f_save = open('./datasets1500/train_data/KT1_truth_train.txt', 'w')
for task in task_train:
    gt = t2truth[task]
    f_save.write(str(task) + '\t' + str(gt) + '\n')
f_save.close()


for i in range(len(w2t_test)):
    w2t_test[i] = w2t_test[i][int(len(w2t_test[i])/2):len(w2t_test[i])]

print(w2t_test)


f_save = open('./datasets1500/test_data/KT1_crowd_test.txt', 'w')
for worker, t2l in w2t_test.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        label = t2l[k][1]
        f_save.write(str(task_id) + '\t' + str(worker) + '\t' + str(label) + '\n')
f_save.close()

task_test = []
for worker, t2l in w2t_test.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        if task_id not in task_test:
            task_test.append(task_id)

f_save = open('./datasets1500/test_data/KT1_truth_test.txt', 'w')
for task in task_test:
    gt = t2truth[task]
    f_save.write(str(task) + '\t' + str(gt) + '\n')
f_save.close()




























