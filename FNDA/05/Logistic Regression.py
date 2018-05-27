import pandas as pd
import numpy as np
from sklearn import model_selection, metrics,cross_val_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from scipy import interp


def Missingrate_Column(df, col):
    '''
    :param df:
    :param col:
    :return:
    '''
    missing_records = df[col].map(lambda x: int(x!=x))
    return missing_records.mean()

def MakeupM_Missing(df,col, makeup_value):
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

def Avg_Calc(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator/denominator

def Outlier_Effect(df,col,target,percentiles=[1,99]):
    '''
    :param df:
    :param col:
    :param target:
    :return:
    '''
    lower, upper = np.percentile(df[col],percentiles[0]),np.percentile(df[col],percentiles[1])
    lower_sample, middle_sample, upper_sample = df[df[col]<lower], df[(df[col]>=lower)&(df[col]<=upper)], df[df[col]>upper]
    lower_fraud, middle_fraud, upper_fraud = lower_sample[target].mean(), middle_sample[target].mean(), upper_sample[target].mean()
    lower_logodds, upper_logodds = np.log(lower_fraud/middle_fraud),np.log(upper_fraud/middle_fraud)
    return [lower_logodds, upper_logodds]

def Zero_Score_Normalization(df,col,percentiles=[1,99]):
    '''
    :param df:
    :param col:
    :param target:
    :return:
    '''
    lower, upper = np.percentile(df[col], percentiles[0]), np.percentile(df[col], percentiles[1])
    new_col = df[col].map(lambda x: min(max(x, lower),upper))
    mu,sigma = new_col.mean(), np.sqrt(new_col.var())
    new_var = new_col.map(lambda x: (x-mu)/sigma)
    return {'new_var':new_var,'lower':lower,'upper':upper,'mu':mu, 'sigma':sigma}



##########################################
####  1，读取数据，划分成训练集与测试集  #####
##########################################
raw_data = pd.read_csv('/Users/Code/Data Collections/AF/anti_fraud_data.csv', header = 0)
train_data, test_data = model_selection.train_test_split(raw_data, test_size=0.3)

all_columns = list(train_data.columns)
all_columns.remove('ID')
all_columns.remove('flag')

#查看整体的欺诈概率
print("the overall fraud rate is {0:2.4f}%".format(train_data['flag'].mean()*100))

##############################
####  2，缺失值分析与补缺  #####
##############################
#先检查是否有常数型字段
fix_value_check = {col: len(set(train_data[col])) for col in all_columns}
fix_value_vars = [var for var in list(fix_value_check.keys()) if fix_value_check[var] == 1]
for var in fix_value_vars:
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


### 对于类别型变量，需要将缺失看做单独的状态,然后进行哑变量编码，并删除原始变量 ###
categorical_cols_withmissing = ['area1_danger','registered_channels','sex','is_email_acct','area8_danger','area6_danger','area11_danger']
dummy_columns = []
for raw_col in categorical_cols_withmissing:
    dummies = pd.get_dummies(train_data.loc[:, raw_col], prefix=raw_col)
    train_data = pd.concat([train_data, dummies], axis=1)
    del train_data[raw_col]
    dummy_columns.append(list(dummies.columns))

#对于数值型变量，可以将原始变量与表示缺失状态的示性变量交互地使用.此外，由于这些变量都是非负数，对于缺失，可以用0来填补
continuous_cols_withmissing = [i for i in columns_with_missing if i not in categorical_cols_withmissing]
for col in continuous_cols_withmissing:
    train_data[col+'_ismissing']  = train_data[col].map(lambda x: int(x!=x))
    train_data[col] = MakeupM_Missing(train_data, col, 0)

#注意到，原始数据中，年龄age没有缺失值，但是有0.需要将0看成缺失
train_data['age'+'_ismissing']  = train_data['age'].map(lambda x: int(x==0))



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

#（2）构造变量，检查平均每次支付金额上升量
for i in range(len(periods)-1):
    avg_payment_1 = periods[i]+'_Avg_pay_amount'
    avg_payment_2 = periods[i+1] + '_Avg_pay_amount'
    increase_payment = periods[i] + '_' + periods[i+1] + '_payment_increase'
    train_data[increase_payment] = train_data[[avg_payment_1,avg_payment_2]].apply(lambda x: x[avg_payment_1] - x[avg_payment_2],axis=1)


#（3）在（1）的基础上求最大的平均支付金额值
avg_payments = [d+'_Avg_pay_amount' for d in periods]
train_data['max_Avg_pay_amount'] = train_data[avg_payments].apply(lambda x: max(x),axis=1)

######################################
####  4，特征极端值分析与归一化处理  #####
######################################
all_columns = sorted(list(train_data.columns))
outlier_columns = ['10m_30m_payment_increase','10m_Avg_pay_amount','10m_pay_times','10m_Sum_pay_amount','12h_1d_payment_increase','12h_Avg_pay_amount',
                   '12h_pay_times','12h_release_bank_cnt','12h_sign_bank_cnt','12h_Sum_pay_amount','15d_30d_payment_increase','15d_Avg_pay_amount','15d_Max_pay_amount',
                   '15d_Min_pay_amount','15d_pay_times','15d_Sum_pay_amount','1d_7d_payment_increase','1d_Avg_pay_amount','1d_pay_times',
                   '1d_Sum_pay_amount','1h_12h_payment_increase','1h_Avg_pay_amount','1h_pay_times','1h_Sum_pay_amount','30d_60d_payment_increase','30d_Avg_pay_amount',
                   '30d_Max_pay_amount','30d_Min_pay_amount','30d_pay_times','30d_Sum_pay_amount','30days_sign_bank_cnt','30m_1h_payment_increase','30m_Avg_pay_amount',
                   '30m_pay_times','30m_Sum_pay_amount','60d_90d_payment_increase','60d_Avg_pay_amount','60d_Max_pay_amount','60d_Min_pay_amount','60d_pay_times','60d_Sum_pay_amount',
                   '7d_15d_payment_increase','7d_Avg_pay_amount','7d_Max_pay_amount','7d_Min_pay_amount','7d_pay_times','7d_pay_times_ismissing','7d_Sum_pay_amount',
                   '90d_Avg_pay_amount','90d_Max_pay_amount','90d_Min_pay_amount','90d_pay_times','90d_Sum_pay_amount','90days_sign_bank_cnt','max_Avg_pay_amount','pay_amount',
                   'registered_days']

outlier_fraud = {}
for col in outlier_columns:
    temp_df = train_data[[col,'flag']]
    outlier_fraud[col] = Outlier_Effect(temp_df, col, 'flag')
outlier_fraud_df = pd.DataFrame.from_dict(outlier_fraud, orient='index')
outlier_fraud_df.columns = ['log_odds_lower','log_odds_upper']


#由于本案例中极端值对欺诈概率无显著影响，故可以直接对特征做归一化处理
#我们使用均值-标准差归一化方法，且需平滑异常值

lower, upper, mu, sigma = {}, {}, {}, {}
for col in outlier_columns:
    temp_df = train_data[[col,'flag']]
    zero_score = Zero_Score_Normalization(temp_df,col)
    train_data[col] = zero_score['new_var']
    lower[col], upper[col], mu[col], sigma[col] =zero_score['lower'], zero_score['upper'], zero_score['mu'], zero_score['sigma']



########################################
####  5，构建基于Ridge的逻辑回归模型  #####
########################################
all_features = list(train_data.columns)
all_features.remove('ID')
all_features.remove('flag')


#对于参数C的选择，我们用交叉验证法选择最优的C
C_list = np.arange(0.01,1,0.01)
auc = []
for c in C_list:
    train2, validation = model_selection.train_test_split(train_data, test_size=0.2)
    LR = LogisticRegression(C=c).fit(train2[all_features], train2['flag'])
    pred = LR.predict_proba(validation[all_features])[:,1]
    test_auc = metrics.roc_auc_score(validation['flag'], pred)
    auc.append(test_auc)

position = auc.index(max(auc))
C_best = C_list[position]
print(max(auc))


LR = LogisticRegression(C=C_best).fit(train_data[all_features], train_data['flag'])
pred = LR.predict_proba(train_data[all_features])

####画ROC曲线
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

fpr, tpr, thresholds = roc_curve(train_data['flag'], pred[:, 1])
mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0                               #初始处为0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.title('AUC=%0.2f'%(roc_auc))