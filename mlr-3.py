# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
df = pd.read_csv("C:/Users/user/Desktop/datasets/ToyotaCorolla.csv",encoding= 'unicode_escape')

df = df[['Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 


plt.hist(df.HP) #histogram
plt.boxplot(df.cc) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=df['cc'], y=df['Price'])
sns.jointplot(x=df['HP'], y=df['Price'])
sns.jointplot(x=df['KM'], y=df['Price'])
sns.jointplot(x=df['Doors'], y=df['Price'])
# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(df['cc'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(df.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(df.iloc[:, :])
input = df.iloc[:,1:]    
for i in input:
    sns.barplot(x=i,y='Price',data=df)
    plt.show()                 
# Correlation matrix 
df.corr()



# preparing model considering only required the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04+ KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df).fit() # regression model

# Summary
ml1.summary()
# p-values for cc and doors are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 80 is showing high influence so we can exclude that entire row

df_new = df.drop(df.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04+ KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight' , data = df_new).fit()    

# Summary
ml_new.summary()

#doors variable is having p-value more than .05 hence we will remove them.
df_new = df_new.drop(columns= 'Doors')
m2 = smf.ols('Price ~ Age_08_04+ KM+HP+cc+Gears+Quarterly_Tax+Weight' , data = df_new).fit()
m2.summary()
 
# Final model
final_ml = smf.ols('Price ~ Age_08_04+ KM+HP+np.log(cc)+Gears+Quarterly_Tax+Weight' , data = df_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(df_new)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = df_new.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()



### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_new, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04+ KM+HP+np.log(cc)+Gears+Quarterly_Tax+Weight' , data = df_train).fit()
model_train.summary()
# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid = test_pred - df_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
