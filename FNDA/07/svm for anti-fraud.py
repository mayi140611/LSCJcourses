import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, metrics
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression

train_data = pd.read_csv('/Users/Code/lianshu/Data/train_data.csv',header = 0)
test_data = pd.read_csv('/Users/Code/lianshu/Data/test_data.csv',header = 0)

all_features = list(train_data.columns)
all_features.remove('ID')
all_features.remove('flag')

train_features = train_data[all_features]

sample_features = random.sample(all_features,20)

pd.scatter_matrix(train_features[sample_features], alpha = 0.3, figsize = (14,8), diagonal = 'kde')

########第一步：挑选变量########
max_age, min_age = max(train_data['age']), min(train_data['age'])
train_data['age'] = train_data['age'].map(lambda x: (x-min_age)/(max_age - min_age))

check_imbalance = np.mean(train_data['flag'])   #坏样本占比1.2%，非均衡样本
good_samples, bad_samples = train_data[train_data['flag'] == 0], train_data[train_data['flag'] == 1]
#从好样本中抽取一部分样本与全部坏样本合在一起组成新的训练集。新的集合里，好坏比为5：1
good_samples_2 = good_samples.sample(bad_samples.shape[0]*5)
train_data_2 = pd.concat([good_samples_2, bad_samples])
print(np.mean(train_data_2['flag']))  #16.7%

X, y = np.mat(train_data_2[all_features]), np.mat(train_data_2['flag']).T

#第1步:去掉取值变化小的特征

features_std = np.std(X, axis=0).getA()[0]
features_std_sorted = sorted(features_std, reverse=True)
plt.bar(x = range(X.shape[1]), height = features_std_sorted)
#选择标准差超过0.5的特征
large_std_features_index = [i for i in range(len(features_std)) if features_std[i]>0.5]

X2 = X[:,large_std_features_index]

#第2步：利用Lasso约束下的逻辑回归模型进行变量挑选
#先在验证集上找出最好的参数C
auc_list = []
for Ci in list(range(1,101)):
    X21, X22, y21,y22 = model_selection.train_test_split(X2,y,test_size=0.2)

    lr = RandomizedLogisticRegression(C=Ci)       # 可在此步对模型进行参数设置
    lr.fit(X21, y21)                                 # 训练模型，传入X、y, 数据中不能包含miss_value
    X_new = lr.inverse_transform(lr.fit_transform(X21,y21))
    #找出X_new中不全部为0的列
    zero_columns = np.sum(np.abs(X_new),axis=0)
    nonzero_columns_index = [i for i in range(len(zero_columns)) if zero_columns[i]>0.0001]
    X3 = X21[:,nonzero_columns_index]
    lr_best = LogisticRegression()
    lr_best.fit(X21,y21)
    prob_predict = lr_best._predict_proba_lr(X22)[:,1]
    auc = metrics.auc(y22,prob_predict,reorder=True)
    auc_list.append(auc)

best_C_position = auc_list.index(max(auc_list))
best_C = list(range(1,101))[best_C_position]


lr = RandomizedLogisticRegression(C=best_C)       # 可在此步对模型进行参数设置
lr.fit(X2, y)                                 # 训练模型，传入X、y, 数据中不能包含miss_value
X_new = lr.inverse_transform(lr.fit_transform(X2,y))
#找出X_new中不全部为0的列
zero_columns = np.sum(np.abs(X_new),axis=0)
nonzero_columns_index = [i for i in range(len(zero_columns)) if zero_columns[i]>0.0001]
X3 = X2[:,nonzero_columns_index]

#画出剩余变量的相关性的热力图
df = pd.DataFrame(X3)
dfData = df.corr()
sns.heatmap(dfData, center=0)
plt.show()


#使用GridSearch选择最优参数组合
###### 先从大范围内尝试参数 ######
parameters = {'kernel':('linear', 'rbf'), 'C':range(1,101,10)}
svc = SVC(class_weight = 'balanced')
clf = model_selection.GridSearchCV(svc, parameters, scoring='f1')
clf.fit(X3, y)
sorted(clf.cv_results_.keys())
best_C, best_kernel = clf.best_params_['C'],clf.best_params_['kernel']

#best_C = 1
parameters = {'kernel':('linear', 'rbf'), 'C':range(1,11)}
svc = SVC(class_weight = 'balanced')
clf = model_selection.GridSearchCV(svc, parameters, scoring='f1')
clf.fit(X3, y)
best_C, best_kernel = clf.best_params_['C'],clf.best_params_['kernel']

#仍然是1，那么我们要考虑较小的C
parameters = {'kernel':('linear', 'rbf'), 'C':np.arange(0.1,2,0.1)}
svc = SVC(class_weight = 'balanced')
clf = model_selection.GridSearchCV(svc, parameters, scoring='f1')
clf.fit(X3, y)
best_C, best_kernel = clf.best_params_['C'],clf.best_params_['kernel']



clf_best = SVC(C = best_C, kernel = best_kernel, class_weight = 'balanced')
clf_best.fit(X3, y)
y_pred_train = np.mat(clf_best.predict(X3))
f1_train = metrics.f1_score(y.getA(), y_pred_train.getA()[0])
conf_mat = metrics.confusion_matrix(y.reshape(y_pred_train.shape[1]).getA()[0], y_pred_train.getA()[0])
tn, fp, fn, tp = conf_mat.ravel()
recall = tp/(tp+fn)


####### 测试 #######
test_data['age'] = test_data['age'].map(lambda x: (x-min_age)/(max_age - min_age))
X_test, y_test = np.mat(test_data[all_features]), np.mat(test_data['flag'])
X_test = X_test[:,large_std_features_index]
X_test = X_test[:,nonzero_columns_index]

y_pred = np.mat(clf_best.predict(X_test))
f1 = metrics.f1_score(y_test.getA()[0], y_pred.getA()[0])

y_all = np.vstack((y_pred,y_test))

error = 1 - np.sum(np.abs(y_all[0,:] - y_all[1,:]))/y_all.shape[1]
conf_mat = metrics.confusion_matrix(y_test.getA()[0], y_pred.getA()[0])
tn, fp, fn, tp = conf_mat.ravel()
recall = tp/(tp+fn)




'''
方案二：利用Informed UnderSampling
'''
good_samples, bad_samples = train_data[train_data['flag'] == 0], train_data[train_data['flag'] == 1]

X_good = np.mat(good_samples[all_features])
np.random.shuffle(X_good)
X_good = X_good[:,large_std_features_index]
X_good = X_good[:,nonzero_columns_index]

X_bad = np.mat(bad_samples[all_features])
X_bad = X_bad[:,large_std_features_index]
X_bad = X_bad[:,nonzero_columns_index]


total_size = X_good.shape[0]
n_samples = 60
block_size = int((total_size - np.mod(total_size,n_samples))/n_samples)
block_indices = []
for i in range(n_samples-1):
    indices = [i*block_size + j for j in range(block_size)]
    block_indices.append(indices)
last_indices =list(range(block_indices[-1][-1]+1, total_size))
block_indices.append(last_indices)

y_test_pred = np.zeros((X_test.shape[0], n_samples))
for i in range(n_samples):
    X_good_small = X_good[block_indices[i],:]
    y_good_small = [0]*X_good_small.shape[0]
    y_bad = [1]*X_bad.shape[0]
    X_small = np.vstack((X_good_small,X_bad))
    y_small = y_good_small+y_bad

    clf_small = SVC(C=best_C, kernel=best_kernel, class_weight='balanced')
    clf_small.fit(X_small, y_small)

    y_pred = np.mat(clf_small.predict(X_test))
    y_pred = y_pred.reshape(X_test.shape[0]).getA()[0]
    y_test_pred[:,i] = y_pred

y_pred= y_test_pred.mean(axis=1)
y_pred = np.round(y_pred,0)

f1 = metrics.f1_score(y_test.getA()[0], y_pred)
conf_mat = metrics.confusion_matrix(y_test.getA()[0], y_pred)
tn, fp, fn, tp = conf_mat.ravel()
recall = tp/(tp+fn)

