import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import pickle
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import SKCompat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from importlib import reload
from matplotlib import pyplot as plt
import operator
reload(sys)
sys.setdefaultencoding( "utf-8")
# -*- coding: utf-8 -*-


def Missingrate_Column(df, col):
    '''
    :param df:
    :param col:
    :return:
    '''
    missing_records = df[col].map(lambda x: int(x!=x))
    return missing_records.mean()


def Makeup_Missing(df,col, makeup_value):
    '''
    :param df:
    :param col:
    :return:
    '''
    raw_values = list(df[col])
    missing_position = [i for i in range(len(raw_values)) if raw_values[i] != raw_values[i]]
    for i in missing_position:
        raw_values[i] = makeup_value
    return raw_values


def Outlier_Effect(df,col,target,percentiles=[1,99]):
    '''
    :param df:
    :param col:
    :param target:
    :return:
    '''
    p1, p3 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    d = p3 - p1
    upper, lower = p3 + 1.5 * d, p1 - 1.5 * d
    lower_sample, middle_sample, upper_sample = df[df[col]<lower], df[(df[col]>=lower)&(df[col]<=upper)], df[df[col]>upper]
    lower_fraud, middle_fraud, upper_fraud = lower_sample[target].mean(), middle_sample[target].mean(), upper_sample[target].mean()
    lower_logodds, upper_logodds = np.log(lower_fraud/middle_fraud),np.log(upper_fraud/middle_fraud)
    return [lower_logodds, upper_logodds]

def Zero_Score_Normalization(series):
    '''
    :param df:
    :param col:
    :param target:
    :return:
    '''
    p1, p3 = np.percentile(series, 25), np.percentile(series, 75)
    d = p3 - p1
    if d == 0:
        return -1
    upper, lower = p3 + 1.5 * d, p1 - 1.5 * d
    new_col = series.map(lambda x: min(max(x, lower),upper))
    mu,sigma = new_col.mean(), np.sqrt(new_col.var())
    new_var = new_col.map(lambda x: (x-mu)/sigma)
    return {'new_var':new_var,'lower':lower,'upper':upper,'mu':mu, 'sigma':sigma}



def Avg_Calc(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator


def Detect_Outlier(x):
    p1,p3 = np.percentile(x,25), np.percentile(x,75)
    d = p3-p1
    upper, lower = p3+1.5*d, p1 - 1.5*d
    x2 = x.map(lambda x: min(max(x, lower), upper))
    return x2

def ROC_AUC(df, score, target, plot=True):
    df2 = df.copy()
    s = list(set(df2[score]))
    s.sort()
    tpr_list = [0]
    fpr_list = [0]
    for k in s:
        df2['label_temp'] = df[score].map(lambda x: int(x >= k))
        TP = df2[(df2.label_temp==1) & (df2[target]==1)].shape[0]
        FN = df2[(df2.label_temp == 1) & (df2[target] == 0)].shape[0]
        FP = df2[(df2.label_temp == 0) & (df2[target] == 1)].shape[0]
        TN = df2[(df2.label_temp == 0) & (df2[target] == 0)].shape[0]
        try:
            TPR = TP / (TP + FN)
        except:
            TPR =0
        try:
            FPR = FP / (FP + TN)
        except:
            FPR = 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    tpr_list.append(1)
    fpr_list.append(1)
    ROC_df = pd.DataFrame({'tpr': tpr_list, 'fpr': fpr_list})
    ROC_df = ROC_df.sort_values(by='tpr')
    ROC_df = ROC_df.drop_duplicates()
    auc = 0
    ROC_mat = np.mat(ROC_df)
    for i in range(1, ROC_mat.shape[0]):
        auc = auc + (ROC_mat[i, 1] + ROC_mat[i - 1, 1]) * (ROC_mat[i, 0] - ROC_mat[i - 1, 0]) * 0.5
    if plot:
        plt.plot(ROC_df['fpr'], ROC_df['tpr'])
        plt.plot([0, 1], [0, 1])
        plt.title("AUC={}%".format(int(auc * 100)))
    return auc


def KS(df, score, target, plot = True):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score, ascending = False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS_list = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS = max(KS_list)
    if plot:
        plt.plot(list(all.index), all['badCumRate'])
        plt.plot(list(all.index), all['goodCumRate'])
        plt.title('KS ={}%'.format(int(KS*100)))
    return KS



#######################
####  1，读取数据  #####
#######################
folderOfData = '/Users/Code/Data Collections/AF/'
data = pd.read_csv(folderOfData + 'anti_fraud_data.csv', header = 0)
train_data, test_data = train_test_split(data, test_size=0.3)
y_train, y_test = train_data['flag'], test_data['flag']
del train_data['ID']
##############################
####  2，缺失值分析与补缺  #####
##############################
#先检查是否有常数型字段

all_columns = list(train_data.columns)
all_columns.remove('flag')

fix_value_check = {col: len(set(train_data[col])) for col in all_columns}
fix_value_vars = [var for var in list(fix_value_check.keys()) if fix_value_check[var] == 1]
for var in fix_value_vars:
    print("{} is a constant".format(var))
    all_columns.remove(var)
    del train_data[var]

#查看每个字段的缺失率
column_missingrate = {col: Missingrate_Column(train_data, col) for col in all_columns}
column_MR_df = pd.DataFrame.from_dict(column_missingrate, orient='index')
column_MR_df.columns = ['missing_rate']
column_MR_df_sorted = column_MR_df.sort_values(by='missing_rate', ascending=False)
plt.bar(x=range(column_MR_df_sorted.shape[0]), height=column_MR_df_sorted.missing_rate)
plt.title('Columns Missing Rate')
#由于变量ip_desc_danger在训练集中全部缺失，故将其删去。
all_columns.remove('ip_desc_danger')
del train_data['ip_desc_danger']
column_MR_df_sorted = column_MR_df_sorted.drop(index=['ip_desc_danger'])
columns_with_missing = column_MR_df_sorted[column_MR_df_sorted.missing_rate > 0].index

#查看缺失值与非缺失值对欺诈的影响
check_missingrate = {}
for col in columns_with_missing:
    temp_df = train_data[[col,'flag']]
    temp_df[col] = temp_df.apply(lambda x: int(x[col] != x[col]),axis=1)
    a = temp_df['flag'].groupby(temp_df[col]).mean()
    check_missingrate[col] = [a.ix[0], a.ix[1]]
check_missingrate_df = pd.DataFrame.from_dict(check_missingrate, orient='index')
check_missingrate_df['log_odds'] = check_missingrate_df.apply(lambda x: np.log(x[1] / x[0]), axis = 1)
check_missingrate_df_sorted = check_missingrate_df.sort_values(by='log_odds', ascending=False)
check_missingrate_df_sorted.columns = ['fraud_rate_nonmissing', 'fraud_rate_missing','log_odds']
plt.bar(x = range(check_missingrate_df_sorted.shape[0]), height = check_missingrate_df_sorted.log_odds)


categorical_cols = []
numerical_cols = []
#区分类别型变量和数值型变量
for col in all_columns:
    temp_df = train_data[train_data[col].notna()][col]
    temp_df = list(set(temp_df))
    if len(temp_df)<=10 or isinstance(temp_df[0],str):
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)

### 对类别型变量进行哑变量编码，并删除原始变量 ###
dummy_map = {}
dummy_columns = []
for raw_col in categorical_cols:
    dummies = pd.get_dummies(train_data.loc[:, raw_col], prefix=raw_col)
    col_onehot = pd.concat([train_data[raw_col], dummies], axis=1)
    col_onehot = col_onehot.drop_duplicates()
    train_data = pd.concat([train_data, dummies], axis=1)
    del train_data[raw_col]
    dummy_map[raw_col] = col_onehot
    dummy_columns = dummy_columns + list(dummies)


#对于数值型变量，可以将原始变量与表示缺失状态的示性变量交互地使用.此外，由于这些变量都是非负数，对于缺失，可以用0来填补
for col in numerical_cols:
    missing_values = train_data[train_data[col].isna()]
    if missing_values.shape[0]>0:
        train_data[col+'_ismissing']  = train_data[col].map(lambda x: int(x!=x))

train_data = train_data.fillna(0)

#注意到，原始数据中，年龄age没有缺失值，但是有0.需要将0看成缺失
train_data['age_ismissing']  = train_data['age'].map(lambda x: int(x==0))




#######################
####  3，特征衍生  #####
#######################
#（1）构造平均值型变量
periods = ['10m','30m','1h','12h','1d','7d','15d','30d','60d','90d']
for period in periods:
    amount = period+'_Sum_pay_amount'
    times = period+'_pay_times'
    avg_payment = period+'_Avg_pay_amount'
    train_data[avg_payment] = train_data[[amount,times]].apply(lambda x: Avg_Calc(x[amount],x[times]),axis=1)
    numerical_cols.append(avg_payment)

#（2）构造变量，检查平均每次支付金额上升量
for i in range(len(periods)-1):
    avg_payment_1 = periods[i]+'_Avg_pay_amount'
    avg_payment_2 = periods[i+1] + '_Avg_pay_amount'
    increase_payment = periods[i] + '_' + periods[i+1] + '_payment_increase'
    train_data[increase_payment] = train_data[[avg_payment_1,avg_payment_2]].apply(lambda x: x[avg_payment_1] - x[avg_payment_2],axis=1)
    numerical_cols.append(increase_payment)


#（3）在（1）的基础上求最大的平均支付金额值
avg_payments = [d+'_Avg_pay_amount' for d in periods]
train_data['max_Avg_pay_amount'] = train_data[avg_payments].apply(lambda x: max(x),axis=1)
numerical_cols.append('max_Avg_pay_amount')
######################################
####  4，特征极端值分析与归一化处理  #####
######################################
outlier_fraud = {}
for col in numerical_cols:
    temp_df = train_data[[col,'flag']]
    outlier_fraud[col] = Outlier_Effect(temp_df, col, 'flag')
outlier_fraud_df = pd.DataFrame.from_dict(outlier_fraud, orient='index')
outlier_fraud_df.columns = ['log_odds_lower','log_odds_upper']


#由于本案例中极端值对欺诈概率无显著影响，故可以直接对特征做归一化处理
#我们使用均值-标准差归一化方法，且需平滑异常值

lower, upper, mu, sigma = {}, {}, {}, {}
deleted_cols = []
for col in numerical_cols:
    temp_df = train_data[col]
    zero_score = Zero_Score_Normalization(temp_df)
    if zero_score == -1:
        deleted_cols.append(col)
        continue
    print(col)
    train_data[col] = zero_score['new_var']
    lower[col], upper[col], mu[col], sigma[col] =zero_score['lower'], zero_score['upper'], zero_score['mu'], zero_score['sigma']

for col in deleted_cols:
    del train_data[col]
    numerical_cols.remove(col)


########################################
# Step 3: 构建基于TensorFlow的神经网络模型 #
########################################
X_train = train_data.copy()
del X_train['flag']
x_train = np.matrix(X_train)
y_train = np.matrix(train_data['flag']).T



#进一步将训练集拆分成训练集和验证集。在新训练集上进行参数估计，在验证集上决定最优的参数

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,test_size=0.4,random_state=9)

#Example: select the best number of units in the 1-layer hidden layer
no_hidden_units_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for no_hidden_units in range(50,101,10):
    print("the current choise of hidden units number is {}".format(no_hidden_units))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units=[no_hidden_units, no_hidden_units-10,no_hidden_units-20],
                                          n_classes=2,
                                          dropout = 0.5
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 100000)
    #monitor the performance of the model using AUC score
    clf_pred_proba = clf._estimator.predict_proba(x_validation)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_validation.getA(),pred_proba)
    no_hidden_units_selection[no_hidden_units] = auc_score
best_hidden_units = max(no_hidden_units_selection.items(), key=operator.itemgetter(1))[0]   #60



#Example: check the dropout effect
dropout_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for dropout_prob in np.linspace(0,0.99,20):
    print("the current choise of drop out rate is {}".format(dropout_prob))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [best_hidden_units, best_hidden_units-10,best_hidden_units-20],
                                          n_classes=2,
                                          dropout = dropout_prob
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 100000)
    #monitor the performance of the model using AUC score
    clf_pred_proba = clf._estimator.predict_proba(x_validation)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_validation.getA(),pred_proba)
    dropout_selection[dropout_prob] = auc_score
best_dropout_prob = max(dropout_selection.items(), key=operator.itemgetter(1))[0]  #0.781


#the best model is
clf1 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [best_hidden_units, best_hidden_units-10,best_hidden_units-20],
                                          n_classes=2,
                                          dropout = best_dropout_prob)
clf1.fit(x_train, y_train, batch_size=256,steps = 100000)
clf_pred_proba = clf1.predict_proba(x_train)
pred_proba = [i[1] for i in clf_pred_proba]
auc_score = roc_auc_score(y_train.getA(),pred_proba)    #0.995




