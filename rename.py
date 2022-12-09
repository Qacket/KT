# # import os
# # path = r"./datasets/original_data/ori_KT1"
# # # 遍历更改文件名
# # num = 0
# # for file in os.listdir(path):
# #     os.rename(os.path.join(path, file), os.path.join(path, str(num)) + ".csv")
# #     num = num + 1


# import pandas as pd
# import os
# path = r"./datasets1500/original_data/ori_KT1"
#
# annotator_num = len(os.listdir(path))
#
# data_total = pd.read_csv('./datasets1500/original_data/ori_KT1/' + str(0) + '.csv')
# data_total.insert(loc=0, column='annotator_id', value=0)
#
# data_total = data_total.drop(labels='timestamp', axis=1)
# data_total = data_total.drop(labels='solving_id', axis=1)
# data_total = data_total.drop(labels='elapsed_time', axis=1)
# print(data_total)
# for i in range(1, annotator_num):
#     data_item = pd.read_csv('./datasets1500/original_data/ori_KT1/' + str(i) + '.csv')
#     data_item.insert(loc=0, column='annotator_id', value=i)
#
#     data_item = data_item.drop(labels='timestamp', axis=1)
#     data_item = data_item.drop(labels='solving_id', axis=1)
#     data_item = data_item.drop(labels='elapsed_time', axis=1)
#     data_total = pd.concat([data_total, data_item])
# print(data_total)
# data_total.to_csv('./datasets1500/original_data/total.txt', header=None, index=None)
#
# # 过滤 answer 为nan的值
# import pandas as pd
# data_total = pd.read_csv('./datasets1500/original_data/total.txt', header=None)
# data_total.columns = ['annotator_id', 'task_id', 'answer']
# print(data_total)
# data_total = data_total.dropna(subset=["answer"],axis=0)
# data_total.to_csv('./datasets1500/original_data/total.txt', header=None, index=None)
#
# import pandas as pd
#
# data_total = pd.read_csv('./datasets1500/original_data/total.txt', header=None)
# print(data_total)
#
# task_total = pd.read_csv('./datasets1500/original_data/questions.csv')
# print(task_total)
#
# data_total[3] = ''
# data_total[4] = ''
#
# for i in range(len(data_total)):
#     print("--------------------------", i)
#     current_task = task_total[task_total.question_id == data_total.iloc[i, 1]]
#     data_total.iloc[i, 3] = current_task.iloc[0, 3]
#     data_total.iloc[i, 4] = current_task.iloc[0, 5]
#
#
# print(data_total)
# data_total.to_csv('./datasets1500/original_data/total2.txt', header=None, index=None)
#
#
# import pandas as pd
#
# data_total = pd.read_csv('./datasets1500/original_data/total2.txt', header=None)
# task_idx = {}
# task_tag = {}
# idx = 0
# tag = 0
# class2idx = {'a':'0', 'b':'1', 'c':'2', 'd':'3'}
# for i in range(len(data_total)):
#     print("--------------------------", i)
#     if data_total.iloc[i, 1] not in task_idx:
#         task_idx[data_total.iloc[i, 1]] = idx
#         idx += 1
#     if data_total.iloc[i, 4] not in task_tag:
#         task_tag[data_total.iloc[i, 4]] = tag
#         tag += 1
#     data_total.iloc[i, 1] = task_idx[data_total.iloc[i, 1]]
#     data_total.iloc[i, 4] = task_tag[data_total.iloc[i, 4]]
#     data_total.iloc[i, 2] = class2idx.get(data_total.iloc[i, 2])
#     data_total.iloc[i, 3] = class2idx.get(data_total.iloc[i, 3])
# print(data_total)
# data_total.to_csv('./datasets1500/original_data/total3.txt', header=None, index=None)
#
#
# import pandas as pd
# data_total = pd.read_csv('./datasets1500/original_data/total3.txt', header=None)
#
# data_total.columns = ['annotator_id', 'task_id', 'answer', 'ground_truth', 'tag']
# print(data_total)
#
# data_total.drop_duplicates(subset=['task_id'], keep='first', inplace=True)
# data_total.index = range(len(data_total))
# print(data_total)
#
# file = open('./datasets1500/total_data/KT1_truth.txt', 'w')
# for i in range(len(data_total)):
#     file.write(str(data_total.iloc[i, 1]) + '\t' + str(data_total.iloc[i, 3]) + '\n')
# file.close()
#
#
# import pandas as pd
# data_total = pd.read_csv('./datasets1500/original_data/total3.txt', header=None)
#
# data_total.columns = ['annotator_id', 'task_id', 'answer', 'ground_truth', 'tag']
# print(data_total)
#
# file = open('./datasets1500/total_data/KT1_crowd.txt', 'w')
# for i in range(len(data_total)):
#     file.write(str(data_total.iloc[i, 1]) + '\t' + str(data_total.iloc[i, 0]) + '\t' + str(data_total.iloc[i, 2]) +'\n')
# file.close()
#
#
#
#
#
# from collections import Counter
#
# import numpy as np
# import pandas as pd
#
#
# data_total = pd.read_csv('./datasets1500/original_data/total3.txt', header=None)
#
# data_total.columns = ['annotator_id', 'task_id', 'answer', 'ground_truth', 'tag']
#
# task_tag = []
# for i in range(len(data_total)):
#     if data_total.iloc[i, 4] not in task_tag:
#         task_tag.append(data_total.iloc[i, 4])
# tag_num = len(task_tag)
# print(tag_num)
#
# acc = []
# for k in range(1500):
#     current_annotator = data_total[data_total.annotator_id == k]
#     tag2acc = {}
#     for i in range(tag_num):
#         tag2acc[i] = []
#
#     for i in range(int(len(current_annotator)/2)):
#         if current_annotator.iloc[i, 2] == current_annotator.iloc[i, 3]:
#             tag2acc[current_annotator.iloc[i, 4]].append(1)
#         else:
#             tag2acc[current_annotator.iloc[i, 4]].append(0)
#
#
#     for tag, state in tag2acc.items():
#
#         count = Counter(state)
#
#         if len(state) == 0:
#             acc.append(0)
#         else:
#             acc.append(count[1]/len(state))
#
#
# acc = np.array(acc)
# acc = acc.reshape(1500, -1)
# print(acc)
# print(type(acc))
# np.save('./datasets1500/annotatorfeature.npy', acc)


# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from sklearn.decomposition import PCA
#
# crowd_data = pd.read_csv('./datasets2000/test_data/KT1_crowd_test.txt', sep="\t", header=None)
#
# task_dim = 11571
#
# for i in range(150000, 200000, 50000):
#     print(i)
#     if (i == 150000):
#         task_id = crowd_data.iloc[i:, 0]
#         annotator_id = crowd_data.iloc[i:, 1]
#         label = crowd_data.iloc[i:, 2]
#     else:
#         task_id = crowd_data.iloc[i:i + 50000, 0]
#         annotator_id = crowd_data.iloc[i:i + 50000, 1]
#         label = crowd_data.iloc[i:i + 50000, 2]
#
#     features = np.load('./datasets2000/annotatorfeature.npy')  # 工人特征 (2000 * 1696)
#     one_hot_dim = features.shape[0]
#     feature_dim = features.shape[1]
#
#     # featrue
#     annotator_id_feature_np = np.array(annotator_id).astype(dtype=float).reshape(-1, 1)
#     annotator_feature = np.repeat(annotator_id_feature_np, feature_dim).reshape(-1, feature_dim)
#     for k in range(len(annotator_feature)):
#         annotator_feature[k] = features[int(annotator_id_feature_np[k][0])]
#     annotator_feature_tensor = torch.from_numpy(annotator_feature).type(torch.float32)
#
#     # onehot
#     annotator_id_onehot_np = np.array(annotator_id).astype(dtype=int).reshape(-1, 1)
#     annotator_tensor = torch.from_numpy(annotator_id_onehot_np)
#     annotator_onehot = F.one_hot(annotator_tensor, one_hot_dim).resize_(len(annotator_tensor), one_hot_dim)
#     annotator_onehot = annotator_onehot.type(torch.float32)
#
#     # cat
#     annotator_inputs = torch.cat((annotator_onehot, annotator_feature_tensor), 1)
#     annotator_inputs = annotator_inputs.cpu().data.numpy()
#     # pca
#     pca = PCA(n_components=1000)
#     annotator_inputs = pca.fit_transform(annotator_inputs)
#     annotator_inputs = torch.from_numpy(annotator_inputs)
#
#     annotator_inputs = annotator_inputs
#
#     task_id_np = np.array(task_id).astype(dtype=int).reshape(-1, 1)
#     task_tensor = torch.from_numpy(task_id_np)
#     task_onehot = F.one_hot(task_tensor, task_dim).resize_(len(task_tensor), task_dim)
#     task_inputs = task_onehot.type(torch.float32)
#     task_inputs = task_inputs.cpu().data.numpy()
#     # pca
#     pca = PCA(n_components=1000)
#     task_inputs = pca.fit_transform(task_inputs)
#     task_inputs = torch.from_numpy(task_inputs)
#
#
#
#     crowd_data_total = np.zeros((len(task_id), 2003), dtype='float32')
#     crowd_data_total[:, 0:1] = np.array(task_id).reshape(-1, 1)
#     crowd_data_total[:, 1:1001] = task_inputs.cpu().data.numpy()
#     crowd_data_total[:, 1001:1002] = np.array(annotator_id).reshape(-1, 1)
#     crowd_data_total[:, 1002:2002] = annotator_inputs.cpu().data.numpy()
#     crowd_data_total[:, 2002:2003] = np.array(label).reshape(-1, 1)
#     print(crowd_data_total)
#     print(crowd_data_total.shape)
#
#     np.save('./datasets2000/test_data/CrowdNpy/KT1_crowd_'+str(i)+'.npy', crowd_data_total)

import os
#
# import numpy as np
#
# temp = []
# for i in [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]:
#     path = './datasets2000/total_data/CrowdNpy/KT1_crowd_' + str(i) + '.npy'
#     real_data = np.load(path, allow_pickle=True)  # 类型是numpy array
#     temp.append(real_data)
# np.save('./datasets2000/total_data/CrowdNpy/KT1_crowd_total.npy', temp)
#
# import numpy as np
#
# data = np.load('./datasets2000/total_data/CrowdNpy/KT1_crowd_total.npy', allow_pickle=True)
# print(data)
# print(type(data))
# print(data.shape)
#
# data_total = data[0]
# print(data_total)
# print(type(data_total))
# print(data_total.shape)
# for i in range(1, len(data)):
#     data_total = np.concatenate((data_total,data[i]),axis=0)
# print(data_total)
# print(type(data_total))
# print(data_total.shape)
# np.save('./datasets2000/total_data/CrowdNpy/KT1_crowd_total_finally.npy', data_total)

import numpy as np

data = np.load('./datasets2000/test_data/CrowdNpy/KT1_crowd_total_finally.npy', allow_pickle=True)
print(data)
print(type(data))
print(data.shape)