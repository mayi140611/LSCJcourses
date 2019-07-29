# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:32:47 2018

@author:ldd
"""
import numpy as np
import matplotlib.pyplot as plt

#Q1.CAPM模型中的alpha值，经常被投资者用来作为投资决策的依据，一般来讲投资者会选择持有alpha较大的股票，现在，假设市场投资组合的收益率为10%,无风险利率为4%，A公司的β值为1.2，预期收益率为9%，而B公司的β值为1.3，预期收益率为12%，那么投资者应该选择那只股票？
#==============================================================================
#Python根据alpha分析股票情况
#α=Ri−Rf−βi∗(E(Rm)−Rf)
αA=0.09-0.04-1.2*(0.1-0.04)
αB=0.12-0.04-1.3*(0.1-0.04)

print(αA)
print(αB)

#%%
#Q2.获取中国农业银行2014年的股票数据，并建立CAPM模型，市场组合收益率用本次课程的数据，无风险利率为3.6%
#==============================================================================
#
import os
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
#从雅虎财经读取中国农业银行2014年的股票数据（股票代码：601288.SS ）
nyyh = web.DataReader('601288.SS','yahoo', dt.datetime(2014,1,1),dt.datetime(2014,12,31)) 
nyyh.tail()

nyyh['return'] = (nyyh['Close'] - nyyh['Close'].shift(1))/nyyh['Close'].shift(1)
#保留收益率变量
nyyh = nyyh['return']
nyyh.dropna(inplace=True)

#%%
os.chdir('H:\LDD\炼数成金\金融\第四章\第四章\作业')
indexcd = pd.read_csv("TRD_Index.csv",index_col = 'Trddt')
#获取中证流通指数的收益率
mktcd = indexcd[indexcd.Indexcd ==902]
mktret = pd.Series(mktcd.Retindex.values,index = pd.to_datetime(mktcd.index))
mktret.name= 'market'
mktret = mktret['2014-01-01':'2014']


#%%
#将中国农业银行2014年的股票数据股份收益率和市场收益率数据进行合并，计算风险溢价
Ret = pd.merge(pd.DataFrame(mktret),pd.DataFrame(nyyh),left_index=True,right_index=True,
               how ='inner')
#计算无风险收益率
rf = 0.036
Ret['risk_premium'] = Ret['market'] - rf

#%%
#绘制中国农业银行2014年的股票数据和中证指数的散点图
import matplotlib.pyplot as plt
plt.scatter(Ret['return'],Ret['market'])
plt.xlabel('return'); plt.ylabel('market')
plt.title('return VS market return')
#%%
#拟合曲线，找到beta
#提出X和Y
import  statsmodels.api as sm 
Ret['constant'] = 1 #增加截距项
X  = Ret[['constant','risk_premium']]
Y = Ret['return']

model= sm.OLS(Y,X)
result =model.fit()
print(result.summary())
#result.predict([1,1])



#%%
#Q3. Ri的值
Ri=0.01+1.2*(0.02-0.005)+0.5*0.024+0.1*0.018+0.005
print(Ri)


#%%
#Q4.读取problem21.txt文件中中远航运2014年股价数据以及ThreeFactors.txt文件中的2014年三因子数据，按照相关步骤建立三因子模型
import os
import pandas as pd

os.chdir('H:\LDD\炼数成金\金融\第四章\第四章\作业')

#读取problem21.txt文件中中远航运2014年股价数据
stock=pd.read_table('problem21.txt',sep='\t',index_col='Date')

stock.index=pd.to_datetime(stock.index)
HXRet=stock.zyhy
HXRet.name='HXRet'
HXRet.plot()
#%%
#读取三因子数据
ThreeFactors=pd.read_table('ThreeFactors.txt',sep='\t',
                           index_col='TradingDate')
#将索引转化为时间格式
ThreeFactors.index=pd.to_datetime(ThreeFactors.index)
ThrFac=ThreeFactors['2014-01-01':] #截取2014年1月1号以后的数据
ThrFac=ThrFac.iloc[:,[2,4,6]] #提取对应的3个因子
#合并股票收益率和3因子的相关数据
HXThrFac=pd.merge(pd.DataFrame(HXRet),pd.DataFrame(ThrFac),
                  left_index=True,right_index=True)

#%%
#作图
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(HXThrFac.HXRet,HXThrFac.RiskPremium2)
plt.subplot(2,2,2)
plt.scatter(HXThrFac.HXRet,HXThrFac.SMB2)
plt.subplot(2,2,3)
plt.scatter(HXThrFac.HXRet,HXThrFac.HML2)
plt.show()
#%%
#回归
import statsmodels.api as sm
regThrFac=sm.OLS(HXThrFac.HXRet,sm.add_constant(HXThrFac.iloc[:,1:4]))
result=regThrFac.fit()
result.summary()

result.params
