#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle


# In[5]:


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


# In[6]:


df


# In[7]:


ProfileReport(df)


# In[8]:


df['BMI'] = df['BMI'].replace(0 , df['BMI'].mean())


# In[9]:


df.columns


# In[10]:


df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())


# In[11]:


df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())


# In[12]:


df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())


# In[13]:


df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[14]:


ProfileReport(df)


# In[15]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df , ax = ax)


# In[16]:


q = df['Insulin'].quantile(.70)
df_new = df[df['Insulin'] < q]


# In[17]:


df_new


# In[18]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new , ax = ax)


# In[19]:


q = df['Pregnancies'].quantile(.98)
df_new = df[df['Pregnancies'] < q]

q = df_new['BMI'].quantile(.99)
df_new = df_new[df_new['BMI']< q]

q = df_new['SkinThickness'].quantile(.99)
df_new = df_new[df_new['SkinThickness']< q]

q = df_new['Insulin'].quantile(.95)
df_new = df_new[df_new['Insulin']< q]

q = df_new['DiabetesPedigreeFunction'].quantile(.99)
df_new = df_new[df_new['DiabetesPedigreeFunction']< q]


q = df_new['Age'].quantile(.99)
df_new = df_new[df_new['Age']< q]


# In[20]:


def outlier_removal(self,data):
        def outlier_limits(col):
            Q3, Q1 = np.nanpercentile(col, [75,25])
            IQR= Q3-Q1
            UL= Q3+1.5*IQR
            LL= Q1-1.5*IQR
            return UL, LL

        for column in data.columns:
            if data[column].dtype != 'int64':
                UL, LL= outlier_limits(data[column])
                data[column]= np.where((data[column] > UL) | (data[column] < LL), np.nan, data[column])

        return data


# In[21]:


#df_new


# In[22]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new , ax = ax)


# In[23]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df , ax = ax)


# In[24]:


ProfileReport(df_new)


# In[25]:


df_new


# In[26]:


y = df_new['Outcome']
y


# In[27]:


X = df_new.drop(columns=['Outcome'])


# In[28]:


X


# In[29]:


scalar = StandardScaler()
ProfileReport(pd.DataFrame(scalar.fit_transform(X)))
X_scaled = scalar.fit_transform(X)


# In[30]:


df_new_scalar = pd.DataFrame(scalar.fit_transform(df_new))
fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new_scalar , ax = ax)


# In[31]:


X_scaled


# In[32]:


y


# In[33]:


def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=["FEATURE", "VIF_SCORE"])


# In[34]:


vif_score(X)


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(X_scaled , y , test_size = .20 , random_state = 144)


# In[36]:


x_train


# In[37]:


x_test


# In[38]:


x_test[0]


# In[39]:


logr_liblinear = LogisticRegression(verbose=1,solver='liblinear')


# In[40]:


logr_liblinear.fit(x_train,y_train )


# In[41]:


logr.predict_proba([x_test[1]])


# In[ ]:


logr.predict([x_test[1]])


# In[ ]:


logr.predict_log_proba([x_test[1]])


# In[ ]:


type(y_test)


# In[ ]:


y_test.iloc[1]


# In[ ]:


y_test


# In[ ]:


logr = LogisticRegression(verbose=1)


# In[ ]:


logr.fit(x_train,y_train)


# In[ ]:


logr_liblinear


# In[ ]:


logr


# In[ ]:


y_pred_liblinear = logr_liblinear.predict(x_test)
y_pred_liblinear


# In[ ]:


y_pred_default = logr.predict(x_test)


# In[ ]:


y_pred_default


# In[ ]:


confusion_matrix(y_test,y_pred_liblinear)


# In[ ]:


confusion_matrix(y_test,y_pred_default)


# In[ ]:


def model_eval(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    specificity=tn/(fp+tn)
    F1_Score = 2*(recall * precision) / (recall + precision)
    result={"Accuracy":accuracy,"Precision":precision,"Recall":recall,'Specficity':specificity,'F1':F1_Score}
    return result
model_eval(y_test,y_pred_liblinear)


# In[ ]:


model_eval(y_test,y_pred_default)


# In[ ]:


auc = roc_auc_score(y_test,y_pred_liblinear)


# In[ ]:


roc_auc_score(y_test,y_pred_default)


# In[ ]:


fpr, tpr, thresholds  = roc_curve(y_test,y_pred_liblinear)


# In[ ]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:


#logist regression task 

https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29#
    
Task Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. WAP to read folder name and make a label in the csv with folder name
2. Remove unneccesarry info in Automated way
3. No other algorithm must be used other than Logistic Regression
4. Try to utilize multiple solvers and make multiple models
5. Provide the best models
6. EDA and all must be done accordingly
Note: No manual approaches will be appreciated


# In[ ]:




