#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


iris_dataset = load_iris()
X,y = iris_dataset.data, iris_dataset.target


# In[4]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[6]:


rf=RandomForestClassifier(n_estimators=3,random_state=42)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))


# In[15]:


get_ipython().system('unzip heart+disease.zip -d /content/')


# In[9]:


get_ipython().system('find /content/my_data/ -name "*.csv"')


# In[ ]:





# In[22]:


cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df1 = pd.read_csv("/content/processed.cleveland.data",names=cols)
df1.head()


# In[24]:


df2 = pd.read_csv("/content/processed.hungarian.data",names=cols)
df2.head()


# In[29]:


len(df1)


# In[30]:


len(df1.replace("?",pd.NA).dropna())


# In[31]:


len(df2)


# In[32]:


len(df2.replace("?",pd.NA).dropna())


# In[34]:


df3 = pd.read_csv("/content/processed.switzerland.data",names=cols)
df3


# In[35]:


len(df3)


# In[36]:


len(df3.replace("?",pd.NA).dropna())


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[39]:


cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df1 = pd.read_csv("/content/processed.cleveland.data",names=cols)
df1.head()


# In[ ]:





# In[38]:


# call the model

model_knn = KNeighborsClassifier()
model_LR = LogisticRegression()
model_svm = SVC()
model_nb = GaussianNB()


# In[43]:


# train test splits
df1_cleaned = df1.replace("?", pd.NA).dropna()
X = df1_cleaned.drop('target', axis=1)
y = df1_cleaned['target']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12)


# In[44]:


# train the models

model_knn.fit(x_train,y_train)
model_LR.fit(x_train,y_train)
model_svm.fit(x_train,y_train)
model_nb.fit(x_train,y_train)


# In[45]:


# testing the model

y_pred_knn = model_knn.predict(x_test)
y_pred_LR = model_LR.predict(x_test)
y_pred_svm = model_svm.predict(x_test)
y_pred_nb = model_nb.predict(x_test)


# In[46]:


#evalution

print(classification_report(y_test,y_pred_knn))


# In[47]:


#evalution

print(classification_report(y_test,y_pred_LR))


# In[48]:


#evalution

print(classification_report(y_test,y_pred_svm))


# In[49]:


#evalution

print(classification_report(y_test,y_pred_nb))


# In[50]:


x_train.columns


# In[51]:


df1.columns


# In[52]:


df1.info()


# In[54]:


import pickle
with open('best_model.pkl','wb') as f:
  pickle.dump(model_LR,f)


# In[ ]:




