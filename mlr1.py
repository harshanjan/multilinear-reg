import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("C:/Users/user/Desktop/datasets/50_Startups.csv") #load the dataset
dataset.head() #displays first 6 observations
plt.figure(figsize=(10,8)) 
g=sns.distplot(dataset['R&D Spend'],label='R&D Spend')
sns.pairplot(dataset.iloc[:,:])
sns.heatmap(dataset.corr())
g=sns.FacetGrid(dataset, col='State')
g=g.map(sns.kdeplot,'Profit')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4:5].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)  
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
print('Train Score: ', regressor.score(X_train, y_train))   #training accuracy
print('Test Score: ', regressor.score(X_test, y_test))  #testing accuracy
