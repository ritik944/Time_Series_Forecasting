#%%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt
from sklearn.linear_model import LinearRegression

import warnings 
warnings.filterwarnings('ignore')

#%%
df =pd.read_csv(r'C:\Users\ritik\OneDrive\Desktop\data science project\time series forcasting\gold_monthly_csv.csv')
print(df.head())
# %%
df.shape
# %%
print(f"date range of gold prices available from - {df.loc[:,'Date'][0]} to {df.loc[:,'Date'][len(df)-1]}")
# %%
date = pd.date_range (start= '1/1/1950',end='8/1/2020',freq='M')
date
# %%
df['month']=date
df.drop('Date',axis=1,inplace= True)
df=df.set_index('month')
df.head()
# %%
df.plot(figsize=(20,8))
plt.title("gold price monthy since 1950 and onwards")
plt.xlabel("month")
plt.ylabel("price")
plt.grid()
# %%
round(df.describe(),3)
# %%
_, ax =plt.subplots(figsize=(25,8))
sns.boxplot(x=  df.index.year, y=df.values[:,0],ax=ax)
plt.title("gold price monthy since 1950 onwards")
plt.xlabel("year")
plt.ylabel("price")
plt.xticks(rotation=90)
plt.grid()

# %%
from statsmodels.graphics.tsaplots import month_plot

fig,ax =plt.subplots(figsize=(22,8))
month_plot(df,ylabel ='gold price',ax=ax)
plt.title("gold price monthy since 1950 onwards")
plt.xlabel("month")
plt.ylabel("price")
plt.grid()
# %%
_,ax=plt.subplots(figsize=(22,8))
sns.boxplot(x= df.index.month_name(),y=df.values[:,0],ax=ax)
plt.title("gold price monthy since 1950 onwards")
plt.xlabel("month")
plt.ylabel("price")
plt.show()
# %%
df_yearly_sum =df.resample('A').mean()
df_yearly_sum.plot()
plt.title("avg gold price yearly since 1950")
plt.xlabel('year')
plt.ylabel('price')
plt.grid()
# %%
df_quartely_sum= df.resample('Q').mean()
df_quartely_sum.plot()
plt.title('avg gold price quaterly')
plt.xlabel('quarter')
plt.ylabel('price')
plt.grid()
# %%
df_decade_sum=df.resample('10Y').mean()
df_decade_sum.plot()
plt.title('avg gold price per decade')
plt.xlabel('decade')
plt.ylabel('price')
plt.grid()
# %% not working
# df_1 =df.groupby(df.index.year).mean().rename(columns={'price':'Mean'})
# df_1 =df_1.merge(df.groupby(df.index.year).std().rename(columns={'price':'Std'}),left_index= True,right_index=True)
# df_1['Cov_pct']= ((df_1['Std']/df_1['Mean'])*100).round(2)
# df_1.head()
# %%    
# fig,ax=plt.subplots(figsize=(15,10))
# df_1['cov_pct'].plot()
# plt.title("avg gold price yearly since 1950")
# plt.xlabel("year")
# plt.ylabel("cv in %")
# plt.grid()
# %% time series forcasting 
train =df[df.index.year<=2015]
test=df[df.index.year>2015]
print(train.shape)
print(test.shape)

# %%
train["Price"].plot(figsize=(13,5),fontsize=15)
test["Price"].plot(figsize=(13,5),fontsize=15)
plt.grid()
plt.legend(['training data ','test data'])
plt.show()
# %% linear regression
train_time= [i+1 for i in range(len(train))]
test_time= [i+len(train)+1 for i in range(len(test))]
print(len(train_time),len(test_time))
# %%
LR_train =train.copy()
LR_test =test.copy()
LR_train['time']=train_time
LR_test['time']=test_time
# %%
lr=LinearRegression()
lr.fit(LR_train[['time']],LR_train["Price"].values)
# %%
test_pred_model=lr.predict(LR_test[['time']])
LR_test['forecast']=test_pred_model

plt.figure(figsize=(14,6))
plt.plot(train['Price'],label='train')
plt.plot(test['Price'],label='test')
plt.plot(LR_test['forecast'],label='ref on time_test data')
plt.legend(loc='best')
plt.grid()
# %%
def mape(actual,pred):
    return round((np.mean(abs(actual-pred)/actual))*100,2)
# %%
mape_model_test =mape(test['Price'].values,test_pred_model)
print((mape_model_test),"%")
# %%
results =pd.DataFrame({'testmape(%)':[mape_model_test]},index=["regressionontime"])
results
# %%
navie_train=train.copy()
navie_test=test.copy()

# %%
navie_test['naive']=np.array(train['Price'])[len(np.asarray(train['Price']))-1]
navie_test['naive'].head()
# %%
plt.figure(figsize=(12,8))
plt.plot(navie_train['Price'],label='Train')
plt.plot(test['Price'],label='Test')
plt.plot(navie_test['naive'],label='naive forecast on test data')
plt.legend(loc='best')
plt.title("naive forecasting")
plt.grid()
# %%
mape_model_test2 =mape(test['Price'].values,navie_test['naive'].values)
print((mape_model_test2),"%")
# %%
results_2 =pd.DataFrame({'testmape(%)':[mape_model_test2]},index=["naivemodel"])
results=pd.concat([results,results_2])
results
# %% final model
final_model = ExponentialSmoothing(df,trend='additive',seasonal='additive').fit(smoothing_level=0.4,smoothing_trend=0.3,smoothing_seasonal=0.6)

# %%
mape_final= mape(df['Price'].values,final_model.fittedvalues)
print("mape",mape_final)
# %%
prediction = final_model.forecast(steps=len(test))
pred_df =pd.DataFrame({'lower_CI': prediction - 1.96*np.std(final_model.resid,ddof=1),
                       'prediction':prediction,
                       'upper_CI':prediction+1.96*np.std(final_model.resid,ddof=1)})
pred_df.head()
# %%
axis=df.plot(label='Actual',figsize=(16,9))
pred_df['prediction'].plot(ax=axis,label='Forecast',alpha=0.5)
axis.fill_between(pred_df.index,pred_df['lower_CI'],pred_df['upper_CI'],color='m',alpha=.15)
axis.set_xlabel('year-month')
axis.set_ylabel('price')
plt.legend(loc='best')
plt.grid()
plt.show()
# %%
# pred_df