# %%

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %% [markdown]
# I have created a dataframe called df and i am storing s&p 500 data from jan 1 2010 and onwards

# %%
import yfinance as yf
df=yf.download('JNJ', start='2018-01-01') # end='2024-11-05
df

# %% [markdown]
# Below I am calculating log returns of the closing prices:
# 
# notes: percent change of the closing prices: df[i−1]
# df[i]−df[i−1]
# 
# i am using log returns to make sure im not j calculating from the base investment but also the returns reinvestment. Compounding effect
# 
# .log because of log returns -> additivity, symmetry, stability
# 
# pct_change caluclates percenatge of current and priior element. if 0.05 the +1 make into percent = 1.05%
# 
# 
# ​
# 

# %%
df['returns']= np.log(df.Close.pct_change()+1)  #I am converting to a percentage to make it more normlainzed by doing log
df

# %% [markdown]
# Below create lags of return couln and returns a list. this offest return frm 1, 2, 3 or 4 days ago depending on specified amnt

# %% [markdown]
# 

# %%
def lagit(df, amntOfLags): 
    names=[] #list
    for i in range(1, amntOfLags+1):
        df['Lag_'+str(i)]= df['returns'].shift(i)
        names.append('Lag_'+str(i))
    return names

# %% [markdown]
# Below I create lagnames column

# %%
lagnames=lagit(df,5)

# %%
# Drop rows with NaN values created by lagging
df.dropna(inplace=True)
df

# %%
model=LinearRegression()

# %%
model.fit(df[lagnames], df['returns'])

# %%
df['prediction_LR']= model.predict(df[lagnames])

# %% [markdown]
# 

# %% [markdown]
# If we want the value the return are -, short sell. If returns are + that you want to buy

# %%
df['direction_LR'] = [1 if i > 0 else -1 for i in df.prediction_LR]
df['strat_LR'] = df['direction_LR'] * df['returns']

result = np.exp(df[['returns', 'strat_LR']].sum())
print(result)

# %%
df

# %%
df['direction_LR']=[1 if i>0 else -1 for i in df.prediction_LR]

# %%
df['strat_LR']=df['direction_LR'] * df['returns'] 

# %%
print(df.columns)


# %%
# Calculate the cumulative returns and the strategy returns
result = np.exp(df[['returns', 'strat_LR']].sum())
print(result)

# %%
cumulative_returns = np.exp(df[['returns', 'strat_LR']].cumsum())
cumulative_returns.plot()
plt.title('S&P 500 and Strategy Returns')
plt.xlabel('Timeline')
plt.ylabel('Returns')
plt.legend(['S&P 500', 'Model Strategy'])
# plt.show()

# %%
from sklearn.model_selection import train_test_split 
train,test = train_test_split(df, shuffle= False, test_size=0.35) #shuffle before 2018 as train data-> better to not have time series actually 
#i also do test_size=0.3 as data is small, randoms state-> as train_test_split is repliciable-> random, 
# 

# %%
model=LinearRegression() 

# %%
train.copy()
test.copy() 

# %%
model.fit(train[lagnames], train['returns' ]) #train model on traising data

# %%
test['prediction_LR']=model.predict(test[lagnames]) #predicting testing model

# %%
test['direction_LR'] = [1 if i>0 else -1 for i in test.prediction_LR ] #

# %%
test['strat_LR']= test['direction_LR'] * test['returns']

# %%
np.exp(test[['returns','strat_LR']].sum())

# %% [markdown]
# 

# %%
#we have to find how many trades to achiabed this performance of above

(test['direction_LR'].diff() !=0).value_counts() #look at direction -1 to 1, -1 to 1, is transaction

# %%
np.exp(test[['returns','strat_LR']].cumsum()).plot()
plt.title('Johnson & Johnson (JNJ) and LR Strategy Returns')
plt.xlabel('Timeline')
plt.ylabel('Returns')
plt.legend(['Johnson & Johnson (JNJ)', 'Linear Reg. Strategy'])
plt.show()

# %%
from sklearn.metrics import r2_score

r2 = r2_score(test['returns'], test['prediction_LR'])
print(f'R-squared: {r2}')



