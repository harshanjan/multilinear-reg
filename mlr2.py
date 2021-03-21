
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
mydata = pd.read_csv("C:/Users/user/Desktop/datasets/Computer_Data.csv")
mydata.head()

mydata.drop(['Unnamed: 0'], axis=1, inplace=True)
mydata.head()

mydata.describe()
mydata.dtypes
p = mydata.hist(figsize = (20,20))
mydata.shape

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()


mydata['cd']= labelencoder.fit_transform(mydata['cd'])
mydata['multi'] = labelencoder.fit_transform(mydata['multi'])
mydata['premium'] = labelencoder.fit_transform(mydata['premium'])


import seaborn as sns


print(mydata.cd.value_counts())
p=mydata.cd.value_counts().plot(kind="bar")
print(mydata.multi.value_counts())
p=mydata.multi.value_counts().plot(kind="bar")
p=sns.pairplot(mydata, hue = 'cd')
plt.figure(figsize=(12,10))  
p=sns.heatmap(mydata.corr(), annot=True,cmap ='RdYlGn')
sns.pairplot(mydata.iloc[:,:])

X=mydata[["price"]]
y=mydata.iloc[:,1:]

########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)  
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
print('Train Score: ', regressor.score(X_train, y_train))  
print('Test Score: ', regressor.score(X_test, y_test))  
#########

import statsmodels.formula.api as smf
model = smf.ols('price ~ (speed+hd+ram+screen+cd+multi+premium+ads+trend)', data= mydata).fit()
model.summary()
model1 = smf.ols('price ~ (speed+hd+ram+screen+cd+multi+premium+ads+trend)', data= mydata).fit()
model1.summary()
model2 = smf.ols('price ~ speed+np.log(hd)+ram+screen+cd+multi+premium+ads+trend', data= mydata).fit()
model2.summary()



from sklearn.model_selection import train_test_split
mydata_train,mydata_test  = train_test_split(mydata,test_size = 0.3) 

# preparing the model on train data 
import statsmodels.formula.api as smf
model_train = smf.ols('price ~ (speed+hd+ram+screen+cd+multi+premium+ads+trend)', data= mydata_train).fit()
model_train.summary()
# train_data prediction
train_pred = model_train.predict(mydata_train)

# train residual values 
train_resid  = train_pred - mydata_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) # 4.04 

# prediction on test data set 
test_pred = model_train.predict(mydata_test)

# test residual values 
test_resid  = test_pred - mydata_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
