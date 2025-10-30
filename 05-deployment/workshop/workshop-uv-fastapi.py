#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[1]:


import pandas as pd
import numpy as np
import sklearn


# In[2]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[4]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[5]:


y_train = df.churn


# In[6]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# One Hot Encoding
# 

# In[7]:


dv = DictVectorizer()

train_dict = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[8]:


datapoint = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# Turn the customer into matrix - by using dictionary vectorizer, in the model you put numbers, not texts

# In[9]:


X = dv.transform(datapoint)


# to access probability of churning

# In[10]:


model.predict_proba(X)[0, 1]


# save the model to pickle  - save it to file called model.bin

# In[11]:


import pickle


# save the model to file output

# In[12]:


with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# load the model from file input

# In[13]:


with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# pipeline makes it easier by combining several steps into one - dictvectoriser and logistic regression , fit pipeline into our dictionaries and target variable
# 
# 

# In[14]:


from sklearn.pipeline import make_pipeline


# In[15]:


pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)


# In[16]:


pipeline.fit(train_dict, y_train)


# In[17]:


pipeline.predict_proba(datapoint)[0, 1]


# In[18]:


import requests


# In[20]:


url = 'http://localhost:9696/predict'

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

response = requests.post(url, json=customer)


# In[21]:


predictions = response.json()


# In[22]:


if predictions['churn']:
    print('accept loan application')
else:
    print('reject loan application')


# In[23]:


for n in numerical:
    print(df[n].describe())
    print()

for c in categorical:
    print(df[c].value_counts())
    print()


# In[ ]:


# Safe helper: check if the FastAPI server is reachable and start it if not (idempotent)
import os, time, subprocess
import requests

url = 'http://localhost:9696/predict'
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85,
}

def is_up():
    try:
        r = requests.post(url, json=customer, timeout=1)
        return r.status_code == 200
    except Exception:
        return False

if is_up():
    print('Server is already running and reachable')
else:
    pidfile = '/tmp/uvicorn_workshop.pid'
    def pid_running(pid):
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    start_server = True
    if os.path.exists(pidfile):
        try:
            with open(pidfile) as f:
                pid = int(f.read().strip())
            if pid_running(pid):
                print(f'Found running uvicorn pid {pid}; waiting for server to respond...')
                for _ in range(10):
                    if is_up():
                        print('Server came up')
                        start_server = False
                        break
                    time.sleep(1)
            else:
                print('Stale pidfile present; will start a new server')
        except Exception as e:
            print('Could not read pidfile; starting server', e)

    if start_server:
        workdir = '/workspaces/machine-learning-zoomcamp/05-deployment/workshop'
        log = '/tmp/uvicorn_workshop.log'
        print('Starting uvicorn (logs ->', log, ')')
        p = subprocess.Popen([
            'uvicorn', 'predict:app', '--host', '0.0.0.0', '--port', '9696', '--app-dir', workdir
        ], stdout=open(log, 'ab'), stderr=open(log, 'ab'))
        try:
            with open(pidfile, 'w') as f:
                f.write(str(p.pid))
        except Exception:
            pass

        for _ in range(15):
            if is_up():
                print('Server started and is reachable')
                break
            time.sleep(1)
        else:
            print('Server did not start within timeout; check', log)

