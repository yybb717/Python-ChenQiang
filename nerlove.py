# 导入需要的包（首选项 本项目 项目解释器 双击pip 选择想要的包安装 在导入）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm
import statsmodels.formula.api as smf   #导入模块
import statsmodels.sandbox


import xlrd    # 为了excel文件导入成功必须的包


# 导入数据
df = pd.read_excel(r'/Users/yy/Documents/chenqiang-data/nerlove.xls')

# 审视数据
print('数据框的前五行:\n',df.head())
print('数据框的列名称:',df.columns)
print('每一列的数据类型:\n',df.dtypes)
print('数据框的维度:',df.shape)
print('变量TC和Q的前五行:\n',df.loc[:4,['TC','Q']])
print('满足Q>=10000条件的变量TC和Q:\n',df.loc[df.Q>=10000,['TC','Q']])


# 考察变量的统计特征
df.describe()  # 所有变量统计特征
# df.describe()['Q'] # 变量Q的统计特征
df.drop(['TC','Q'],axis=1).corr(method='pearson', min_periods=1) # 三个价格变量PL、PF、PK的相关系数

# 画图
# 变量Q的直方图和核密度图
sns.distplot(df['Q'],hist=True,kde=True,kde_kws={'color':'red'})
# 变量TC和Q的散点图
sns.lmplot(x='Q',y='TC',data=df,
           scatter_kws={'color':'blue','label':'TC'},
           line_kws={'color':'red','label':'Fitted values'})
plt.xlabel('Q')
plt.ylabel('TC')
plt.legend()

# 生成新变量(新序列)
df['lntc']=np.log(df.TC)
df['lnq']=np.log(df.Q)
df['lnpl']=np.log(df.PL)
df['lnpf']=np.log(df.PF)
df['lnpk']=np.log(df.PK)
df['large']=(df.Q>=10000)

df.head()


# 小样本ols
smallols = smf.ols('lntc ~ lnq + lnpl + lnpk + lnpf', data=df).fit()
print(smallols.summary())


# 大样本OLS（异方差稳健的标准误）
bigols = smf.ols('lntc ~ lnq + lnpl + lnpk + lnpf', data=df).fit(cov_type='HC1',use_t=True)
print(bigols.summary())




# 扰动项的正态性检验
# 1、残差的直方图和核密度图
import scipy.stats as stats
sns.distplot(smallols.resid,fit=stats.norm,
            hist_kws={'color':'steelblue','edgecolor':'black'},
            kde_kws={'color':'black','linestyle':'--','label':'resid_kde'},
            fit_kws={'color':'red','linestyle':':','label':'norm'})
plt.xlabel('resid')
plt.legend()


# 可见，残差的核密度与正态密度还是有点差距的！

# 2、画残差的QQ图
sm.ProbPlot(smallols.resid).qqplot(line='q')

# 3、JB检验

# 4、shapiro检验
print(stats.shapiro(smallols.resid))
# 拒绝原假设，认为不服从正态分布




# 扰动项异方差的检验

# 1、画残差和拟合值、解释变量lnq的散点图

# 画残差和拟合值的散点图
ax1=plt.subplot2grid(shape=(2,1),loc=(0,0))
ax1.scatter(bigols.fittedvalues,smallols.resid)
ax1.set_xlabel('fittedvalues')
ax1.set_ylabel('resid')
ax1.axhline(y=0,c='red',ls='--',lw='3')
ax1.set_title('fittedvalues-resid')

# 画残差和解释变量lnq的散点图
ax2=plt.subplot2grid(shape=(2,1),loc=(1,0))
ax2.scatter(df.lnq,bigols.resid)
ax2.set_xlabel('lnq')
ax2.set_ylabel('resid')
ax2.axhline(y=0,c='red',ls='--',lw='3')
ax2.set_title('lnq-resid')

# 调整两个子图图的水平间距核高度间距
plt.subplots_adjust(hspace=0.6,wspace=0.3)

# 可见，拟合值较小时，残差波动大，也就是扰动项方差较大，因此可能扰动项很可能存在异方差

# 2、怀特检验
sm.stats.diagnostic.spec_white(bigols.resid,exog=smallols.model.exog)


# 3、BP检验
sm.stats.diagnostic.het_breuschpagan(bigols.resid,exog_het=smallols.model.exog)
# 第二个元素是LM统计量的p值，可见拒绝同方差的原假设，认为扰动项存在异方差



# 加权最小二乘WLS
df['lne2'] = np.log(smallols.resid**2)  # ln西格玛方
fuzhuols = sm.OLS(df.lne2,df.lnq).fit()  # 辅助回归
e2f = np.exp(fuzhuols.fittedvalues)      # 得到ln西格玛方的预测值后，再计算以e为底数的指数，得到西格玛方的估计

X=df[['lnq','lnpl','lnpk','lnpf']]
X=sm.add_constant(X)
WLS = sm.WLS(df['lntc'], X, weights=1/e2f).fit()
print(WLS.summary())


# 对函数形式的检验（不会）


# 多重共线性的检验
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 自变量X
X=df[['lnq','lnpl','lnpk','lnpf']]
X=sm.add_constant(X) # 必须加上常数项
vif=pd.DataFrame() # 先建立一个空数据框
vif['features']=X.columns
vif['VIF Factor']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
print(vif)



# 极端数据
outliers=smallols.get_influence()

# 高杠杆值
leverage=outliers.hat_matrix_diag

# diffits值
dffits=outliers.dffits[0]

# 学生化残差
resid_stu=outliers.resid_studentized_external

# cook距离
cook=outliers.cooks_distance[0]


# 把这四个统计量合并成一个数据框
contat1=pd.concat([pd.Series(leverage,name='leverage'),
                       pd.Series(dffits,name='dffits'),
                       pd.Series(resid_stu,name='resid_stu'),
                       pd.Series(cook,name='cook')],
                       axis=1)


# 再把这个数据框和原数据框合并，更方便观看每个样本点的值
outliers = pd.concat([df,contat1],axis=1)
# 按照杠杆值降序排序，找到杠杆值最大的三个数据
outliers.sort_values(by='leverage',axis=0,ascending=False).head(3)



