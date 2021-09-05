import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块
import xlrd    # 为了excel文件导入成功必须的包

df = pd.read_excel(r'/Users/yy/Desktop/Predict to Profit.xlsx')

# OLS1 = sm.formula.ols(formula='Profit~RD_Spend+Administration+Marketing_Spend', data=df).fit()
# print(OLS1.summary())


X=df[['RD_Spend', 'Administration', 'Marketing_Spend']]
y=df['Profit']
X=sm.add_constant(X) # 必须加上常数项
bigols=sm.RLM(y, X).fit()
print(bigols.summary())
