# 导入需要的包（首选项 本项目 项目解释器 双击pip 选择想要的包安装 在导入）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
import xlrd    # 为了excel文件导入成功必须的包

# 获取并查看数据
df = pd.read_excel('/Users/yy/Documents/chenqiang-data/icecream.xls')
print(df)

# 冰淇淋的消费量和气温随着时间变化的折线图
# plt.plot(df.time, df.consumption)
# plt.plot(df.time, df.temp100)
# plt.legend(['consumption', 'temp100'])
# plt.show()


# 小样本OLS回归
smallols = smf.ols('df.consumption ~ df.temp + df.price + df.income', data=df).fit()
print(smallols.summary())  # 回归拟合的全部结果



# 异方差的检验

# 残差与滞后残差的散点图
# e1 = smallols.resid
# e2 = e1.shift(-1)
# sns.regplot(e1, e2)
# plt.show()

# 残差的自相关图
# sm.graphics.tsa.plot_acf(smallols.resid)
# plot_acf(smallols.resid)   # 二选一
# plt.show()
# 残差的偏自相关图（报错！咋办！！）
# sm.graphics.tsa.plot_pacf(smallols.resid)
# plot_pacf(smallols.resid)  # 二选一
# plt.show()

# LB检验和BG检验（报错！！！！咋办！！！！）
# lb_test(smallols.resid, lags=1, boxpierce=True)
# DW检验的结果可在回归结果里面查看



# 处理自相关
# 1.用OLS+HAC标准误
# hacols1 = smf.ols('df.consumption ~ df.temp + df.price + df.income', data=df).fit(cov_type='HAC', cov_kwds={'maxlags':3})
# print(hacols1.summary())
#
# hacols2 = smf.ols('df.consumption ~ df.temp + df.price + df.income', data=df).fit(cov_type='HAC', cov_kwds={'maxlags':6})
# print(hacols2.summary())


# 2.用FGLS的CO估计法和PW估计法（找不到！！！！）


# 3.在解释变量中加入temp的滞后值，在进行OLS回归
Ltemp = df.temp.shift(-1)
smallols2 = smf.ols('df.consumption ~ df.temp + Ltemp + df.price + df.income', data=df).fit()
print(smallols2.summary())



# 用信息准则选择解释变量的个数
# 在解释变量中加入temp的一阶和二阶滞后值，在进行OLS回归.发现AIC和BIC上升，因此只有一阶滞后项是最好的
Ltemp2 = Ltemp.shift(-1)
smallols3 = smf.ols('df.consumption ~ df.temp + Ltemp + Ltemp2 + df.price + df.income', data=df).fit()
print(smallols3.summary())