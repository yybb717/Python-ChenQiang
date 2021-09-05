# 导入需要的包（首选项 本项目 项目解释器 双击pip 选择想要的包安装 在导入）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块



import xlrd    # 为了excel文件导入成功必须的包

# 导入数据
df = pd.read_excel(r'/Users/yy/Documents/chenqiang-data/icecream.xls')
df.head()

# 时间-消费量、温度折线图
plt.plot(df.time,df.consumption,label='consumption')
plt.plot(df.time,df.temp100,label='temp100')
plt.xlabel('time')
plt.legend()


# 小样本ols
smallols = smf.ols('consumption~ temp+price+income',data=df).fit()
print(smallols.summary())


# 检验扰动项是否存在自相关

# 1、残差和他的一阶滞后散点图
df1 = pd.DataFrame()
df1['x']=smallols.resid
df1['y']=smallols.resid.shift(-1)
sns.lmplot(x='x',y='y',data=df1)
plt.xlabel('resid')
plt.ylabel('resid(-1)')

# 2、残差的自相关图
sm.graphics.tsa.plot_acf(smallols.resid, lags=10)

# 3、BG检验（不会）

# 4、白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(smallols.resid, lags=1, boxpierce=True)

# 5、DW检验

# OLS➕HAC=Newey-West估计法
# 截断参数为观测值个数的四分之一次方。约为3
NW = smf.ols(formula='consumption~ temp+price+income', data=df).fit(cov_type='HAC',cov_kwds={'maxlags': 3},use_t=True)
print(NW.summary())

# 把截断参数改为6，看看HAC标准误是否对截断参数敏感
NW1 = smf.ols(formula='consumption~ temp+price+income', data=df).fit(cov_type='HAC',cov_kwds={'maxlags': 6},use_t=True)
print(NW1.summary())
# HAC标准误没啥大变化

# CO估计法（不会）
# PW估计法（不会）


# 加入气温的一阶滞后值来OLS回归
df['Ltemp']=df['temp'].shift(1)

smallols1 = smf.ols('consumption~ temp+Ltemp+price+income',data=df).fit()
print(smallols1.summary())