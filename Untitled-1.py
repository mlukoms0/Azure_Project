# %%
import pandas as pd
import json
import requests
import datetime


# %%
## stock info dataset
api_key = '112fd4c9fef462d4ff99217055be9015'

url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols=VOO&date_from=2015-01-01&date_to=2025-12-12&limit=1000"


response = requests.get(url)
if response.status_code != 200:
    print(f"Error: Received status code {response.status_code}, error: {response.json().get('error', 'Unknown error')}")
else:

    print("Request successful.")
    


# %% [markdown]
# ### bronze level data

# %%
raw = response.json()
#dfRaw = pd.json_normalize(raw['data'])
dfRaw = pd.DataFrame(raw['data'])
dfRaw.to_csv('stock_info.csv', index=False)
raw['pagination']

# %%
dfRaw.info()

# %%
dfRaw.head()

# %%
dfRaw.to_csv('bronze_stock_info.csv', index=False)

# %% [markdown]
# #### Silver Level
#     column transformations and imputation

# %%
### date column transformations for interpretability
dfRaw['month'] = pd.to_datetime(dfRaw['date']).dt.month
dfRaw['year'] = pd.to_datetime(dfRaw['date']).dt.year
dfRaw['day'] = pd.to_datetime(dfRaw['date']).dt.day
dfRaw['day_of_week'] = pd.to_datetime(dfRaw['date']).dt.dayofweek
dfRaw['quarter'] = pd.to_datetime(dfRaw['date']).dt.quarter
dfRaw['dateClean'] = pd.to_datetime(dfRaw['date']).dt.date

# %%
### create model DF
df = dfRaw.copy()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop(columns='date',inplace=True)
df.rename(columns={'dateClean': 'date'}, inplace=True)


# %%
### feature engineering for machine learning model
df['per_change'] = ((df['close']-df['open'])/df['open'])*100

df['movement'] = df['per_change'].apply(lambda x: 1 if x > 0.25 else 0)
df.head()

# %%
### silver level data final row counts
df.info()

# %%
df.to_csv('silver_stock_info.csv', index=False)

# %% [markdown]
# #### Gold Level

# %%
### business case is building a model that will predict if a stocks price will move .25% or higher in the day. 
### first step is featuer selection, then scaling/standardization

# %%

dfGold = df.copy()

# %%
target = dfGold['movement']
features = dfGold[['open', 'high', 'low', 'close', 'volume', 'month', 'year', 'day', 'day_of_week', 'quarter', 'per_change']]


