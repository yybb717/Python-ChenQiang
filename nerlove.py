# 导入需要的包（首选项 本项目 项目解释器 双击pip 选择想要的包安装 在导入）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块
import xlrd    # 为了excel文件导入成功必须的包



# 获取并查看数据
from patsy.highlevel import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_excel('/Users/yy/Documents/chenqiang-data/nerlove.xls')
print(df)




# 审视数据
# print(df[['TC', 'Q']])  # 看变量TC和Q的具体数据
# print(df[['TC', 'Q']][:5])  # 看变量TC和Q的前五个数据
# print(df[['TC', 'Q']][df['Q'] < 10])  # 看变量TC和Q的当Q小于10的具体数据
# print(df.describe())  # 看所有数值型变量的统计特征
# print(df.corr())  # 计算所有变量之间的相关系数




# 画图
# 变量Q的直方图
# plt.hist(x=df['Q'], color='pink', edgecolor='black', density=False)
# plt.title('Q直方图')
# plt.xlabel('Q')
# plt.ylabel('Frequency')

# 变量Q和TC的散点图并给每个散点加上序号
# plt.scatter(df['Q'], df['TC'])
# plt.xlabel('Q')
# plt.ylabel('TC')
# for i in range(len(df['TC'])):
#     plt.annotate(i, xy=(df['Q'][i], df['TC'][i]), xytext=(df['Q'][i]+0.1, df['TC'][i]+0.1))
# plt.savefig('/Users/yy/Documents/picture1')   # 保存图片
# plt.show()

# 用seaborn模块画一个解释变量和被解释变量的带回归线和置信区间的散点图
# sns.regplot(x=df.Q, y=df.TC)
# plt.show()
# 用seaborn模块分别画所有解释变量和被解释变量的带回归线和置信区间的散点图
# sns.pairplot(df, x_vars=['Q', 'PL', 'PF', 'PK'], y_vars='TC', kind='reg')
# plt.show()





# 开始做线性回归！

# 先把变量都取对数
df = df.apply(np.log)
df.columns = ['lntc', 'lnq', 'lnpl', 'lnpf', 'lnpk']
print(df.head())
# 小样本ols回归

# 方法一
smallols = smf.ols('lntc ~ lnq + lnpl + lnpk + lnpf', data=df).fit()

# 方法二
X=df[['lnq', 'lnpl', 'lnpk', 'lnpf']]
y=df['lntc']
# X=sm.add_constant(X) # 必须加上常数项
# smallols=sm.OLS(y,X).fit()


# print(smallols.summary())  # 回归拟合的全部结果
# print(smallols.params)    # 部分结果：回归系数的ols估计值
# lntchat = smallols.fittedvalues   # 被解释变量lntc的拟合值
# e1 = smallols.resid   # 拟合的残差lntchat-lntc

# 将拟合结果画出来。首先设定图轴，图片大小为 8 × 6。画出原数据，图像为圆点，默认颜色为蓝。画出拟合数据，图像为红色带点间断线。放置注解。
# x1 = np.arange(0, 145)
# x = pd.Series(x1)
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x, df.lntc, 'o', label='data')
# ax.plot(x, lntchat, 'r--.', label='OLS')
# ax.legend(loc='best')
# plt.show()


# 异方差
# 残差图
# plt.scatter(lntchat, e1)
# plt.xlabel('lntchat')
# plt.ylabel('e1')
# for i in range(len(lntchat)):
#     plt.annotate(i, xy=(lntchat[i], e1[i]), xytext=(lntchat[i]+0.1, e1[i]+0.1))
# plt.show()

# plt.scatter(df.lnq, e1)
# plt.xlabel('lnq')
# plt.ylabel('e1')
# plt.show()

# 怀特检验
# print(sm.stats.diagnostic.het_white(smallols.resid, exog=smallols.model.exog))
# BP检验
# print(sm.stats.diagnostic.het_breuschpagan(smallols.resid, smallols.model.exog))

# FWLS
# 先算扰动项方差
# e1 = smallols.resid   # 拟合的残差lntchat-lntc
# e2 = np.square(e1)
# lne2 = np.log(e2)
# # 辅助回归（无截距项）
# fuzhuols = sm.OLS(lne2, df.lnq).fit()
# print(fuzhuols.summary())
# lne2f = fuzhuols.fittedvalues
# e2f = np.exp(lne2f)  # WLS要使用的权重  也是 扰动项方差的估计
# # 开始WLS回归
# gls = sm.GLS(y, X, sigma=e2f).fit()
# print(gls.summary())



# 计算VIF
# 将因变量lntc，自变量lnq + lnpl + lnpk + lnp和截距项（值为1的1维数组）以数据框的形式组合起来
# y, X = dmatrices('lntc ~ lnq + lnpl + lnpk + lnpf', data=df, return_type='dataframe')
# 构造空的数据框
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# vif["features"] = X.columns
# print(vif)



# 失败的线性插值的实验
# series1 = pd.Series([2,3,4,5,6,7,8])
# series2 = pd.Series([3,4,5,None,7,8,9])

# from scipy import interpolate

# print(interpolate.interp1d(series1, series2, kind='linear'))
