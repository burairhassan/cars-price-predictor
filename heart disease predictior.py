#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,PowerTransformer,FunctionTransformer,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,classification_report,precision_score,recall_score,f1_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier,BaggingClassifier,StackingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
df = pd.read_csv('hd.csv')


# In[2]:


df.head()


# In[3]:


pd.unique(df['chest_pain_type'])


# In[4]:


pd.unique(df['slope']) 


# In[5]:


pd.unique(df['vessels_colored_by_flourosopy'])


# In[6]:


df['thalassemia']=df['thalassemia'].replace('No','Normal')
pd.unique(df['thalassemia'])


# In[7]:


pd.unique(df['rest_ecg'])


# In[8]:


df.columns.tolist()


# In[9]:


cat=['resting_blood_pressure','cholestoral','Max_heart_rate','oldpeak']


# In[10]:


for i in cat:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[11]:


uppl=['resting_blood_pressure','cholestoral','oldpeak']
for i in uppl:
      q3=df[i].quantile(0.75)
      q1=df[i].quantile(0.25)
      iqr=q3-q1
      upl=q3+1.5*iqr
      limit=df[i]<upl
      df=df[limit]


# In[12]:


for i in cat:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[13]:


q3=df['Max_heart_rate'].quantile(0.75)
q1=df['Max_heart_rate'].quantile(0.25)
iqr=q3-q1
low=q1- 1.5*iqr
limit=df['Max_heart_rate']>=low
df=df[limit]


# In[14]:


sns.boxplot(data=df,x='Max_heart_rate')
plt.show()


# In[15]:


x=df.drop(['target'],axis=1)


# In[16]:


y=df['target']


# In[17]:


xtr,xts,ytr,yts=train_test_split(x,y,test_size=0.2,random_state=42)
ordinalcat=[['Asymptomatic','Non-anginal pain','Atypical angina','Typical angina'],['Normal','ST-T wave abnormality','Left ventricular hypertrophy'],['Upsloping','Flat','Downsloping'],['Zero','One','Two','Three','Four'],['Normal','Reversable Defect','Fixed Defect']]


# In[18]:


ct=ColumnTransformer([
    
  ('oe',OrdinalEncoder(categories=ordinalcat),['chest_pain_type','rest_ecg','slope','vessels_colored_by_flourosopy','thalassemia']),
    ('ohe',OneHotEncoder(sparse_output=False,drop='first'),['sex','fasting_blood_sugar','exercise_induced_angina'])

],remainder='passthrough')


# In[19]:


xtr=ct.fit_transform(xtr)
xts=ct.transform(xts)
sc=StandardScaler()
xtr=sc.fit_transform(xtr)
xts=sc.transform(xts)


# In[29]:


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(xtr,ytr)
xtrpred=rf.predict(xtr)
acc=accuracy_score(ytr,xtrpred)*100


# In[30]:


acc


# In[31]:


cf=confusion_matrix(ytr,xtrpred)
cf


# In[32]:


recall=recall_score(ytr,xtrpred)
recall


# In[33]:


xtspred=rf.predict(xts)
acc=accuracy_score(yts,xtspred)*100
acc


# In[34]:


cf=confusion_matrix(yts,xtspred)
cf


# In[35]:


recall=recall_score(yts,xtspred)
recall


# In[42]:


score=cross_val_score(rf,xtr,ytr,cv=20,scoring='accuracy')
print(score)


# In[43]:


scoree=cross_val_score(rf,xts,yts,cv=20,scoring='accuracy')
print(score)


# In[ ]:




