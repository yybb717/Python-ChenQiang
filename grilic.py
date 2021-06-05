# 导入需要的包（首选项 本项目 项目解释器 双击pip 选择想要的包安装 在导入）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块
import xlrd    # 为了excel文件导入成功必须的包

from patsy.highlevel import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 获取并查看数据
df = pd.read_excel('/Users/yy/Documents/chenqiang-data/grilic.xls')
print(df.head())

# 1、所有变量的统计特征
print(df.describe())

# 2、智商与受教育年限的相关关系
print(df['iq'].corr(df['s']))


# 3、参照系：用稳健标准误的OLS回归
X1 = df[['s', 'expr', 'tenure', 'rns', 'smsa']]
X1 = sm.add_constant(X1)  # 必须加上常数项
bigols1 = sm.RLM(df['lnw'], X1).fit()
print(bigols1.summary())

# 4、引入智商iq作为能力的代理变量，在进行OLS回归
X2 = df[['s', 'iq', 'expr', 'tenure', 'rns', 'smsa']]
X2 = sm.add_constant(X2)  # 必须加上常数项
bigols2 = sm.RLM(df['lnw'], X2).fit()
print(bigols2.summary())


# sm.sandbox.regression.gmm.IV2SLS()

from statsmodels.sandbox.regression.gmm import IV2SLS
# help(IV2SLS)
# a = IV2SLS(df.iq, [df.med, df.kww, df.mrt, df.age], [df.s, df.expr, df.tenure, df.rns, df.smsa]).fit()
# print(a.summary())