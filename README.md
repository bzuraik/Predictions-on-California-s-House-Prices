# Predictions-on-California-s-House-Prices
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt


import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
```


```python
# 1. Load Data
```


```python
xlsx_path = '1553768847_housing.xlsx'
df=pd.read_excel(xlsx_path)
```


```python
# Print first few rows of this data.
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>NEAR BAY</td>
      <td>452600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>NEAR BAY</td>
      <td>358500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>NEAR BAY</td>
      <td>352100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>NEAR BAY</td>
      <td>341300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>NEAR BAY</td>
      <td>342200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'ocean_proximity', 'median_house_value'],
          dtype='object')




```python
# 2. Missing D ata
# Now once the below block is run, we can see that the column total_bedrooms has 207 empty values!
```


```python
df.isnull().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        207
    population              0
    households              0
    median_income           0
    ocean_proximity         0
    median_house_value      0
    dtype: int64




```python
mean = df.total_bedrooms.mean()
df.total_bedrooms=df.total_bedrooms.fillna(mean)
df.isnull().sum()

```




    longitude             0
    latitude              0
    housing_median_age    0
    total_rooms           0
    total_bedrooms        0
    population            0
    households            0
    median_income         0
    ocean_proximity       0
    median_house_value    0
    dtype: int64




```python
# 3. Encode categorical data :

# Convert categorical column in the dataset to numerical data.
```


```python
le = LabelEncoder()
df['ocean_proximity']=le.fit_transform(df['ocean_proximity'])
```


```python
# Standardize data :

# Standardize training and test datasets.
```


```python
names = df.columns
# Create the Scaler object
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)
scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.327835</td>
      <td>1.052548</td>
      <td>0.982143</td>
      <td>-0.804819</td>
      <td>-0.975228</td>
      <td>-0.974429</td>
      <td>-0.977033</td>
      <td>2.344766</td>
      <td>1.291089</td>
      <td>2.129631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.322844</td>
      <td>1.043185</td>
      <td>-0.607019</td>
      <td>2.045890</td>
      <td>1.355088</td>
      <td>0.861439</td>
      <td>1.669961</td>
      <td>2.332238</td>
      <td>1.291089</td>
      <td>1.314156</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.332827</td>
      <td>1.038503</td>
      <td>1.856182</td>
      <td>-0.535746</td>
      <td>-0.829732</td>
      <td>-0.820777</td>
      <td>-0.843637</td>
      <td>1.782699</td>
      <td>1.291089</td>
      <td>1.258693</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.337818</td>
      <td>1.038503</td>
      <td>1.856182</td>
      <td>-0.624215</td>
      <td>-0.722399</td>
      <td>-0.766028</td>
      <td>-0.733781</td>
      <td>0.932968</td>
      <td>1.291089</td>
      <td>1.165100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.337818</td>
      <td>1.038503</td>
      <td>1.856182</td>
      <td>-0.462404</td>
      <td>-0.615066</td>
      <td>-0.759847</td>
      <td>-0.629157</td>
      <td>-0.012881</td>
      <td>1.291089</td>
      <td>1.172900</td>
    </tr>
  </tbody>
</table>
</div>




```python
# extract X and Y
```


```python
X_Features=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'ocean_proximity']
X=scaled_df[X_Features]
Y=scaled_df['median_house_value']

print(type(X))
print(type(Y))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.series.Series'>



```python
print(df.shape)
print(X.shape)
print(Y.shape)
```

    (20640, 10)
    (20640, 9)
    (20640,)



```python
# 4. Split the dataset : 

# Split the data into 80% training dataset and 20% test dataset.
```


```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
```

    (16512, 9) (16512,)
    (4128, 9) (4128,)



```python
ig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='longitude',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='latitude',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='housing_median_age',y='median_house_value',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='total_rooms',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='total_bedrooms',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='population',y='median_house_value',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='households',y='median_house_value',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='median_income',y='median_house_value',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='ocean_proximity',y='median_house_value',ax=axs[2],figsize=(16,8))
```




    <AxesSubplot:xlabel='ocean_proximity', ylabel='median_house_value'>






![output_18_1](https://user-images.githubusercontent.com/95400232/163911764-ca17c92e-daf0-4d6c-865d-846c78f15eca.png)


![output_18_2](https://user-images.githubusercontent.com/95400232/163911827-10f5a776-fca6-4679-a2d4-f9719b6e1b0b.png)



![output_18_3](https://user-images.githubusercontent.com/95400232/163911836-b96bb62d-3a6d-4c03-8c0e-96ab7928bfc3.png)





```python
# Perform Linear Regression on training data.

```


```python
linreg=LinearRegression()
linreg.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
y_predict = linreg.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.6056598120301221
    0.6276223517950295



```python
# Extract just the median_income column from the independent variables (from X_train and X_test).
```


```python
x_train_Income=x_train[['median_income']]
x_test_Income=x_test[['median_income']]
```


```python
print(x_train_Income.shape)
print(y_train.shape)
```

    (16512, 1)
    (16512,)



```python
linreg=LinearRegression()
linreg.fit(x_train_Income,y_train)
y_predict = linreg.predict(x_test_Income)
```


```python
print(linreg.intercept_, linreg.coef_)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.005623019866893164 [0.69238221]
    0.7212595914243148
    0.47190835934467734



```python
scaled_df.plot(kind='scatter',x='median_income',y='median_house_value')
plt.plot(x_test_Income,y_predict,c='red',linewidth=2)
```




    [<matplotlib.lines.Line2D at 0x7f3486408a10>]




![output_26_1](https://user-images.githubusercontent.com/95400232/163911856-9948f47e-e4fd-4b0c-b1ba-7b324696cbe0.png)


