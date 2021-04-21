#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# # "Machine Learning Classification Algorithm"

# # "Primary Tumor Domain"

# 
# |Group Members|Student ID|
# |-----|-------|
# |Diana Davila|C0750286|
# |Mansi Akhani|C0769480|
# |Janki Patel|C0773592|
# |Mital Goriya|C0771593|

# <b>Assignment Details:</b>
# 
# 1. Data Preparation <br>
# 
# 2. Preprocessing of data <br>
# 
# 3. Building and Validating Classification Model <br>
#  

# # 1. Data Preparation

# ## Loading data

# #### The primary tumor domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. <br>
# #### M. Zwitter and M. Soklic obtained the data.

# #### http://archive.ics.uci.edu/ml/datasets/Primary+Tumor

# ### Attributes: 
#  
#    1. class: lung, head & neck, esophasus, thyroid, stomach, duoden & sm.int, colon, rectum, anus, salivary glands, pancreas, gallblader, liver, kidney, bladder, testis, prostate, ovary, corpus uteri,cervix uteri, vagina, breast -----> [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
#    2. age:   <30, 30-59, >=60 -----> [0,1,2]
#    3. sex:   male, female  -----> [0,1]
#    4. histologic-type: epidermoid, adeno, anaplastic  -----> [0,1,2]
#    5. degree-of-diffence: well, fairly, poorly -----> [0,1,2]
#    6. bone: yes, no  -----> [0,1]
#    7. bone-marrow: yes, no -----> [0,1]
#    8. lung: yes, no -----> [0,1]
#    9. pleura: yes, no -----> [0,1]
#    10. peritoneum: yes, no -----> [0,1]
#    11. liver: yes, no -----> [0,1]
#    12. brain: yes, no -----> [0,1]
#    13. skin: yes, no -----> [0,1]
#    14. neck: yes, no -----> [0,1]
#    15. supraclavicular: yes, no -----> [0,1]
#    16. axillar: yes, no -----> [0,1]
#    17. mediastinum: yes, no -----> [0,1]
#    18. abdominal: yes, no -----> [0,1]

# #### Validating dataset content

# In[5]:


rawdata = open('primary-tumor.data').read()
rawdata[0:500]


# #### Dividing dataset by ","

# In[6]:


import pandas as pd
dataset = pd.read_csv("primary-tumor.data", sep=',', header=None)


# In[7]:


dataset.head()


# #### Adding column names

# In[8]:


dataset.columns=['class','age','sex','histologic-type','degree-of-diffence','bone','bone-marrow','lung','pleura','peritoneum','liver','brain','skin','neck','supraclavicular','axillar','mediastinum','abdominal']


# In[9]:


dataset.head()


# In[10]:


dataset.shape


# In[11]:


dataset.info()


# # 2. Preprocessing of data

# ## Missing values

# In[12]:


dataset.isnull().sum()


# #### Validating unique values

# In[13]:


def unique_values(column):
    values=dataset[column].unique()
    return values


# In[14]:


for i in range(len(dataset.columns)):
    column_name=dataset.columns[i]
    values = unique_values(column_name)
    print('Unique values in column number: {}'.format(i))
    print(values)


# In[15]:


dataset['sex'].value_counts()


# In[16]:


dataset['histologic-type'].value_counts()


# In[17]:


dataset['degree-of-diffence'].value_counts()


# In[18]:


dataset['skin'].value_counts()


# In[19]:


dataset['axillar'].value_counts()


# ### Replacing "?" with NaN values

# In[20]:


import numpy as np
df = dataset.replace('?',np.nan)


# In[21]:


df.head()


# ### Validating null values

# In[22]:


df.isnull().sum()


# In[23]:


df.info()


# ### Replacing missing values with "mode"

# In[24]:


def replace_null_mode(column):
    mode_value=(pd.to_numeric(column[column.notnull()])).mode()
    column.fillna(mode_value[0], inplace=True)
    return mode_value


# In[25]:


mode_sex = replace_null_mode(df['sex'])
print("The mode value for the column Sex is: {}".format(mode_sex[0]))


# In[26]:


mode_skin=replace_null_mode(df['skin'])
print("The mode value for the column Skin is: {}".format(mode_skin[0]))


# In[27]:


mode_axillar = replace_null_mode(df['axillar'])
print("The mode value for the column Axillar is: {}".format(mode_axillar[0]))


# In[28]:


mode_histologic_type= replace_null_mode(df['histologic-type'])
print("The mode value for the column Histologic Type is: {}".format(mode_histologic_type[0]))


# In[29]:


df.isnull().sum()


# ### "degree-of-diffence" Column has 155 missing values out of 339. We implemented two different options:

# ## Option 1 

# ### Removing "degree-of-diffence" column

# In[30]:


df_1 = df.drop(['degree-of-diffence'], axis=1)


# In[31]:


df_1.isnull().sum()


# In[32]:


def unique_values(column):
    values=df_1[column].unique()
    return values

for i in range(len(df_1.columns)):
    column_name=df_1.columns[i]
    values = unique_values(column_name)
    print('Unique values in column number: {}'.format(i))
    print(values)


# In[33]:


df_1.info()


# ### Columns with Nan values were replaced with the "mode" value (integer). Nevertheless,  the non-Nan values were string type. Therefore, we used pd.to_numeric() to convert all values to integers.

# In[34]:


df_1['sex']= pd.to_numeric(df_1["sex"])
df_1['histologic-type']= pd.to_numeric(df_1["histologic-type"])
df_1['skin']= pd.to_numeric(df_1["skin"])
df_1['axillar']= pd.to_numeric(df_1["axillar"])


# In[35]:


df_1.info()


# In[36]:


for i in range(len(df_1.columns)):
    column_name=df_1.columns[i]
    values = unique_values(column_name)
    print('Unique values in column number: {}'.format(i))
    print(values)


# ## Option 2

# ### Replacing "degree-of-diffence" column with mode value

# In[37]:


mode_degree_of_diffence= replace_null_mode(df['degree-of-diffence'])
print("The mode value for the column Degree of Difference is: {}".format(mode_degree_of_diffence[0]))


# In[38]:


df.isnull().sum()


# In[39]:


def unique_values(column):
    values=df[column].unique()
    return values

for i in range(len(df.columns)):
    column_name=df.columns[i]
    values = unique_values(column_name)
    print('Unique values in column number: {}'.format(i))
    print(values)


# In[40]:


df.info()


# ### Columns with Nan values were replaced with the "mode" value (integer). Nevertheless,  the non-Nan values were string type. Therefore, we used pd.to_numeric() to convert all values to integers.

# In[41]:


df['sex']= pd.to_numeric(df["sex"])
df['degree-of-diffence']= pd.to_numeric(df["degree-of-diffence"])
df['histologic-type']= pd.to_numeric(df["histologic-type"])
df['skin']= pd.to_numeric(df["skin"])
df['axillar']= pd.to_numeric(df["axillar"])


# In[42]:


df.info()


# In[43]:


def unique_values(column):
    values=df[column].unique()
    return values

for i in range(len(df.columns)):
    column_name=df.columns[i]
    values = unique_values(column_name)
    print('Unique values in column number: {}'.format(i))
    print(values)


# ##  Validating outliers

# In[44]:


import matplotlib.pyplot as plt
plt.boxplot(df_1['class'])
plt.show()


# In[45]:


plt.boxplot(df_1['age'])
plt.show()


# In[46]:


plt.boxplot(df_1['sex'])
plt.show()


# In[47]:


plt.boxplot(df_1['histologic-type'])
plt.show()


# In[48]:


plt.boxplot(df_1['bone'])
plt.show()


# In[49]:


plt.boxplot(df_1['bone-marrow'])
plt.show()


# In[50]:


plt.boxplot(df_1['lung'])
plt.show()


# In[51]:


plt.boxplot(df_1['pleura'])
plt.show()


# In[52]:


plt.boxplot(df_1['peritoneum'])
plt.show()


# In[53]:


plt.boxplot(df_1['liver'])
plt.show()


# In[54]:


plt.boxplot(df_1['brain'])
plt.show()


# In[55]:


plt.boxplot(df_1['skin'])
plt.show()


# In[56]:


plt.boxplot(df_1['neck'])
plt.show()


# In[57]:


plt.boxplot(df_1['supraclavicular'])
plt.show()


# In[58]:


plt.boxplot(df_1['axillar'])
plt.show()


# In[59]:


plt.boxplot(df_1['mediastinum'])
plt.show()


# In[60]:


plt.boxplot(df_1['abdominal'])
plt.show()


# In[61]:


plt.boxplot(df['degree-of-diffence'])
plt.show()


# In[62]:


df.describe()


# ### We validated that there are not outliers in out dataset. All values are within the correct range

# ## Validating  Correlation

# ### Correlation for Option 1 (column degree-of-difference was removed because of the number of missing values)

# In[63]:


df_1.corr()


# In[64]:


import seaborn as sb
sb.heatmap(df_1.corr())


# ### Correlation for Option 2 (missing values for column degree-of-difference were replaced by the "mode")

# In[65]:


df.corr()


# In[66]:


sb.heatmap(df.corr())


# ### We could validate that there are not correlation between the columns in our dataset for both approaches

# In[67]:


print(df.groupby('class').size())


# In[68]:


sb.countplot(df['class'],label="Count")
plt.show()


# In[69]:


df.head()


# # 3. Building and Validating Classification Model

# ## Multi-class classification problem

# # Splitting data into training and test 

# In[70]:


df.describe()


# In[71]:


X = df.drop('class', axis=1)
Y= df['class']


# ### Scaling data

# In[72]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X_scaled = pd.DataFrame(scale.fit_transform(X), columns=X.columns)


# In[73]:


X_scaled


# In[74]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                   Y, test_size=0.20, random_state=11)


# In[75]:


X_train[0:5]


# In[76]:


y_train[0:5]


# In[77]:


y_train.value_counts()


# In[78]:


y_test.value_counts()


# ## Classification algorithms

# ### RandomForest

# In[75]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
rf = RandomForestClassifier(n_estimators=150, random_state=0, max_features=10)
rf.fit(X_train, y_train)
Y_pred = rf.predict(X_test)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(rf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
print("\nAccuracy for Testing : ", accuracy_score(y_test, Y_pred))
train_pred = rf.predict(X_train)
print("\nAccuracy for Training: ", accuracy_score(y_train, train_pred))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train, train_pred))


# ### Decision Tree

# In[76]:


from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier(random_state=5)
d_tree.fit(X_train, y_train)
Y_pred = d_tree.predict(X_test)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(d_tree.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(d_tree.score(X_test, y_test)))
print("\nAccuracy for Testing : ", accuracy_score(y_test, Y_pred))
train_pred = d_tree.predict(X_train)
print("\nAccuracy for Training: ", accuracy_score(y_train, train_pred))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train, train_pred))


# ### Implementing Prunning using cost_complexity_prunnin_path

# In[77]:


d_tree_pruning = d_tree.cost_complexity_pruning_path(X_train, y_train)
alpha_value= d_tree_pruning['ccp_alphas']
alpha_value


# In[78]:


import seaborn as sb
accuracy_test=[]
accuracy_training=[]

for i in alpha_value:
    d_tree_prun=DecisionTreeClassifier(ccp_alpha=i)
    d_tree_prun.fit(X_train, y_train)
    pred_y_train=d_tree_prun.predict(X_train)
    pred_y_test=d_tree_prun.predict(X_test)
    accuracy_training.append(accuracy_score(y_train,pred_y_train))
    accuracy_test.append(accuracy_score(y_test,pred_y_test))
    
sb.set()
plt.figure(figsize=(13,6))
sb.lineplot(y=accuracy_test, x=alpha_value, label='Accuracy for testing set')
sb.lineplot(y=accuracy_training, x=alpha_value, label='Accuracy for training set')
plt.xticks(ticks=np.arange(0.00,0.045,0.005))
plt.show()


# ### Decision tree using ccp_alpha of 0.019

# In[79]:


from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier(random_state=5, ccp_alpha=0.019)
d_tree.fit(X_train, y_train)
Y_pred = d_tree.predict(X_test)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(d_tree.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(d_tree.score(X_test, y_test)))
print("\nAccuracy for Testing : ", accuracy_score(y_test, Y_pred))
train_pred = d_tree.predict(X_train)
print("\nAccuracy for Training: ", accuracy_score(y_train, train_pred))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train, train_pred))


# ### KNN

# In[79]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
print("\nAccuracy for Testing : ", accuracy_score(y_test, Y_pred))
train_pred = knn.predict(X_train)
print("\nAccuracy for Training: ", accuracy_score(y_train, train_pred))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train, train_pred))


# ### Implementing Grid searching of the KNeighborsClassifier hyperparameters

# In[85]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
number_neighbors = range(1, 23, 2)
weights_value = ['uniform', 'distance']
distance = ['euclidean', 'manhattan', 'minkowski']

grid_search = dict(n_neighbors=number_neighbors,weights=weights_value,metric=distance)
grid_search_knn = GridSearchCV(estimator=knn, param_grid=grid_search, n_jobs=-1, scoring='accuracy')
grid_result_knn = grid_search_knn.fit(X_train, y_train)

print("The best acurracy of: %f was obtained using the following hyperparameters %s" % (grid_result.best_score_, grid_result_knn.best_params_))


# ### Visualization of Classification Error

# In[3]:


import matplotlib.pyplot as plt

classification_algorithm = ['Random Forest', 'Decision Tree', 'KNN']
color= ['green','blue','red']
error = [0.12, 0.13, 0.47]
plt.bar(classification_algorithm,error, color=color)
plt.title('Error for each Classification algorithm')
plt.xlabel('Classification algorithm')
plt.ylabel('Classification Error')
plt.show()


# # Fixing overfitting

# ## F-Score Selection for overfitting

# In[86]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(f_classif, k=10)
selected_features = selector.fit_transform(X_scaled, Y)


# In[87]:


selector.scores_


# In[157]:


f_score_indexes = (-selector.scores_).argsort()[:8]
f_score_indexes


# In[158]:


X_fr=X_scaled.iloc[:, [14,12,1,2,8,4,9,15]]


# In[159]:


X_fr.head()


# In[160]:


X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fr,
                                                   Y, test_size=0.20, random_state=12)


# ### RandomForest with F-score selection

# In[162]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_f, y_train_f)
Y_pred_f = rf.predict(X_test_f)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(rf.score(X_train_f, y_train_f)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test_f, y_test_f)))
print("\nAccuracy for Testing : ", accuracy_score(y_test_f, Y_pred_f))
train_pred_f = rf.predict(X_train_f)
print("\nAccuracy for Training: ", accuracy_score(y_train_f, train_pred_f))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train_f, train_pred_f))


# ### Decision Tree with F-score selection

# In[163]:


from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train_f, y_train_f)
Y_pred_f = d_tree.predict(X_test_f)
print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(d_tree.score(X_train_f, y_train_f)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(d_tree.score(X_test_f, y_test_f)))
print("\nAccuracy for Testing : ", accuracy_score(y_test_f, Y_pred_f))
train_pred_f = d_tree.predict(X_train_f)
print("\nAccuracy for Training: ", accuracy_score(y_train_f, train_pred_f))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train_f, train_pred_f))


# ### KNN with F-score selection

# In[164]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_f, y_train_f)
Y_pred_f = knn.predict(X_test_f)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_f, y_train_f)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_f, y_test_f)))
print("\nAccuracy for Testing : ", accuracy_score(y_test_f, Y_pred_f))
train_pred_f = knn.predict(X_train_f)
print("\nAccuracy for Training: ", accuracy_score(y_train_f, train_pred_f))
print("\nClassification error: ", 1-metrics.accuracy_score(y_train_f, train_pred_f))


# ## K-fold Cross Validation

# In[79]:


df['class'].value_counts()


# ### K-fold Cross Validation could not be implemented because some classes contain only one value.

# ### Visualization of Accuracy

# In[4]:


classification_algorithm = ['Decision Tree with F-score', 'KNN', 'Random Forest with F-score selection']
color= ['green','blue','red']
accuracy = [0.56, 0.52, 0.56]
plt.bar(classification_algorithm,accuracy, color=color)
plt.title('Accuracy for each Classification algorithm')
plt.xlabel('Classification algorithm')
plt.ylabel('Accuracy')
plt.show()


# ### We could obtain the best result using Random Forest with F-score selection with accuracy of: 56%
# 

# In[ ]:




