---
layout: post
title: "Price Prediction of Used Cars in Germany (2011-2021) — Part I: Exploratory Data Analysis"
date: 2021-11-05
description: ""
img_url: #/assets/img/2021-11-05-car-price-germany-eda/output_55_0.png__
tags: ["EDA"]
# github_url:
# colab_url:
# kaggle_url:
---

[AutoScout24](https://www.autoscout24.com/), based in Munich, is one of the largest European online marketplace for buying and selling new and used cars. With its products and services, the car portal is aimed at private consumers, dealers, manufacturers and advertisers. In this post, we use a high-quality car data set from Germany that was automatically scraped from the AutoScout24 and is available on the [Kaggle](https://www.kaggle.com/ander289386/cars-germany) website. This interesting data set comprises more than 45,000 records of cars for sale in Germany registered between 2011 and 2021.

## Outline

- [Handling Missing Data](#handling-missing-data)
  - [Overview](#overview-1)
  - [Median Imputation of Missing `hp` Values](#median-imputation-of-missing-hp-values)
  - [Mode Imputation of Missing `gear` Values](#mode-imputation-of-missing-gear-values)
  - [Revisiting Our Data Set](#revisiting-our-data-set)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Visualizing Categorical Data Using Treemaps](#visualizing-categorical-data-using-treemaps)
    - [Car Makes (Brands)](#car-makes-brands)
    - [Fuel Types](#fuel-types)
    - [Gear Types](#gear-types)
  - [Visualizing Categorical Data Using Box Plots](#visualizing-categorical-data-using-box-plots)
  - [Scatter Plots: Visualizing the Relationship Between Numerical Data](#scatter-plots-visualizing-the-relationship-between-numerical-data)
  - [Distribution of the Numerical Features](#distribution-of-the-numerical-features)
  - [Checking Correlations Between Price and Numerical Features](#checking-correlations-between-price-and-numerical-features)
- [Distribution of Target Variable](#distribution-of-target-variable)

---

Let's first install and import packages we'll use and set up the working environment:

```python
!conda install -c conda-forge -q -y squarify
```

```python
from datetime import datetime
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import probplot

import seaborn as sns
import squarify
```


```python
# Settings
%config InlineBackend.figure_format = 'retina'
sns.set_theme(font_scale=1.25)
random_seed = 87
```

---
## Data Loading

Let's load the data from the dataset file in CSV format. Pandas comes with a handy function `read_csv` to create a DataFrame from a file path or buffer.


```python
cars_rawdf = pd.read_csv('../data/autoscout24_germany_dataset.csv')
cars_rawdf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 46390 entries, 0 to 46389
    Data columns (total 9 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   mileage    46390 non-null  int64
     1   make       46390 non-null  object
     2   model      46247 non-null  object
     3   fuel       46390 non-null  object
     4   gear       46208 non-null  object
     5   offerType  46390 non-null  object
     6   price      46390 non-null  int64
     7   hp         46361 non-null  float64
     8   year       46390 non-null  int64
    dtypes: float64(1), int64(3), object(5)
    memory usage: 3.2+ MB



```python
cars_rawdf.sample(frac=1, random_state=random_seed).head(n=10)
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
      <th>mileage</th>
      <th>make</th>
      <th>model</th>
      <th>fuel</th>
      <th>gear</th>
      <th>offerType</th>
      <th>price</th>
      <th>hp</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10316</th>
      <td>151000</td>
      <td>BMW</td>
      <td>325</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>13999</td>
      <td>204.0</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>25361</th>
      <td>88413</td>
      <td>Fiat</td>
      <td>Punto Evo</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>4740</td>
      <td>105.0</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>45829</th>
      <td>50</td>
      <td>Skoda</td>
      <td>Fabia</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Pre-registered</td>
      <td>14177</td>
      <td>95.0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>34066</th>
      <td>25000</td>
      <td>Citroen</td>
      <td>C3</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>5000</td>
      <td>73.0</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>40035</th>
      <td>48882</td>
      <td>Opel</td>
      <td>Mokka</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>13200</td>
      <td>140.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>39682</th>
      <td>59500</td>
      <td>SEAT</td>
      <td>Mii</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>4450</td>
      <td>60.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>5457</th>
      <td>25602</td>
      <td>Ford</td>
      <td>Fiesta</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>7840</td>
      <td>71.0</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>15175</th>
      <td>2500</td>
      <td>Volkswagen</td>
      <td>Caddy</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>Demonstration</td>
      <td>38550</td>
      <td>122.0</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>18569</th>
      <td>26828</td>
      <td>Renault</td>
      <td>Clio</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>10990</td>
      <td>90.0</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>40890</th>
      <td>167482</td>
      <td>Volkswagen</td>
      <td>Golf</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>Used</td>
      <td>9799</td>
      <td>110.0</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>



The data set contains information about cars in Germany (registered between 2011 and 2021), for which we'll be predicting the sale price. It shows various fields such as `make` (car brand), `hp` (horse power), `mileage` (the aggregate number of miles traveled) etc. A quick look at the `offerType` field shows that the data set contains five distinct types of offer: Used, Pre-registered, Demonstration, Employer's car, and New.


```python
print(
    'Counts of unique offer types:',
    cars_rawdf['offerType'].value_counts(),
    sep='\n'
)
```

    Counts of unique offer types:
    Used              40110
    Pre-registered     2780
    Demonstration      2366
    Employee's car     1121
    New                  13
    Name: offerType, dtype: int64


Here, we select data for 'Used' cars only to build a model for predicting car sale prices in used-car markets.
After selecting all records for 'Used' cars in the original data set, we remove the column `offerType` since it is same for all entries in the extracted new data set.


```python
cars = cars_rawdf[cars_rawdf['offerType'] == 'Used'].copy()
cars.sample(frac=1).head(n=10)
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
      <th>mileage</th>
      <th>make</th>
      <th>model</th>
      <th>fuel</th>
      <th>gear</th>
      <th>offerType</th>
      <th>price</th>
      <th>hp</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4136</th>
      <td>47500</td>
      <td>Ford</td>
      <td>Fiesta</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>8150</td>
      <td>71.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>5965</th>
      <td>98600</td>
      <td>Ford</td>
      <td>Transit Custom</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>Used</td>
      <td>11781</td>
      <td>101.0</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>3054</th>
      <td>61150</td>
      <td>Volkswagen</td>
      <td>up!</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>7450</td>
      <td>60.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>23141</th>
      <td>22000</td>
      <td>Mazda</td>
      <td>CX-3</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>20000</td>
      <td>120.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>23861</th>
      <td>18100</td>
      <td>Nissan</td>
      <td>X-Trail</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>27375</td>
      <td>159.0</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>8375</th>
      <td>31008</td>
      <td>Volkswagen</td>
      <td>up!</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>6990</td>
      <td>60.0</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>40104</th>
      <td>64000</td>
      <td>Volkswagen</td>
      <td>Golf Variant</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>13800</td>
      <td>125.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>11711</th>
      <td>28900</td>
      <td>Renault</td>
      <td>Clio</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>8700</td>
      <td>90.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>10652</th>
      <td>75367</td>
      <td>Mercedes-Benz</td>
      <td>CLA 220</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>Used</td>
      <td>27970</td>
      <td>177.0</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>43350</th>
      <td>14721</td>
      <td>Peugeot</td>
      <td>108</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>Used</td>
      <td>9990</td>
      <td>72.0</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars.drop(columns='offerType', inplace=True)
cars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 40110 entries, 0 to 46375
    Data columns (total 8 columns):
     #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
     0   mileage  40110 non-null  int64
     1   make     40110 non-null  object
     2   model    39985 non-null  object
     3   fuel     40110 non-null  object
     4   gear     39937 non-null  object
     5   price    40110 non-null  int64
     6   hp       40091 non-null  float64
     7   year     40110 non-null  int64
    dtypes: float64(1), int64(3), object(4)
    memory usage: 2.8+ MB

---

## Data Manipulation and Cleaning

### Numerical Features

#### Add a New Column: Car Age

The columns `year`, that is the car registration year, does not solely provide direct information, but rather in combination with current date. A feature that is considered by many car buyers on the used-car markets is the age of the car. In combination with the average anual mileage (that can be different from country to country, and for different years in a given country!), it determines whether a car has reasonable mileage or not. This property is missing in the data set, but we can substract column `year` from current year and create a meaningful feature named car `age`:


```python
cars['age'] = datetime.now().year - cars['year']
cars.drop(columns='year', inplace=True)

cars.sample(frac=1).head(n=10)
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
      <th>mileage</th>
      <th>make</th>
      <th>model</th>
      <th>fuel</th>
      <th>gear</th>
      <th>price</th>
      <th>hp</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10837</th>
      <td>180000</td>
      <td>Ford</td>
      <td>Focus</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>6900</td>
      <td>95.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>31773</th>
      <td>82500</td>
      <td>Citroen</td>
      <td>Berlingo</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>7790</td>
      <td>92.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>35784</th>
      <td>210000</td>
      <td>Renault</td>
      <td>Megane</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>3700</td>
      <td>110.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>30646</th>
      <td>12461</td>
      <td>Audi</td>
      <td>Q3</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>46480</td>
      <td>190.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25864</th>
      <td>141500</td>
      <td>Volkswagen</td>
      <td>Caddy</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>9999</td>
      <td>102.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>31055</th>
      <td>11316</td>
      <td>Ford</td>
      <td>Focus</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>19589</td>
      <td>125.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13981</th>
      <td>138994</td>
      <td>Fiat</td>
      <td>500L</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>6995</td>
      <td>105.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17854</th>
      <td>63290</td>
      <td>Nissan</td>
      <td>Pulsar</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>9750</td>
      <td>110.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18375</th>
      <td>16075</td>
      <td>Ford</td>
      <td>Focus</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>19996</td>
      <td>150.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21141</th>
      <td>162200</td>
      <td>Citroen</td>
      <td>C3</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>3890</td>
      <td>68.0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



#### Mask Outliers

We can identify outliers (that are also considered as missing observations) for numerical features `mileage`, `hp`, `age`, as well as the target variable `price`. It is obvious that thses values cannot be assigned to zero or negative for any samples in the data set. Therefore, we mask negative values (outliers) for these columns:


```python
numerical_features = ['mileage', 'hp', 'age']

for feature in numerical_features + ['price']:
    cars[feature].mask(cars[feature] < 0, inplace=True)
```

### Categorical Features

#### Overview


```python
categorical_features = ['make', 'fuel', 'gear']

for feature in categorical_features:
    print(
        f"Categorical feature: '{feature}'\n"
        f"No. of unique elemnts: {cars[feature].nunique()}\n"
        f"Unique values:\n{cars[feature].unique()}"
    )
    print('=' * 40)
```

    Categorical feature: 'make'
    No. of unique elemnts: 76
    Unique values:
    ['BMW' 'Volkswagen' 'SEAT' 'Renault' 'Peugeot' 'Toyota' 'Opel' 'Mazda'
     'Ford' 'Mercedes-Benz' 'Chevrolet' 'Audi' 'Fiat' 'Kia' 'Dacia' 'MINI'
     'Hyundai' 'Skoda' 'Citroen' 'Infiniti' 'Suzuki' 'SsangYong' 'smart'
     'Volvo' 'Jaguar' 'Porsche' 'Nissan' 'Honda' 'Lada' 'Mitsubishi' 'Others'
     'Lexus' 'Cupra' 'Maserati' 'Bentley' 'Land' 'Alfa' 'Jeep' 'Subaru'
     'Dodge' 'Microcar' 'Baic' 'Tesla' 'Chrysler' '9ff' 'McLaren' 'Aston'
     'Rolls-Royce' 'Alpine' 'Lancia' 'Abarth' 'DS' 'Daihatsu' 'Ligier'
     'Caravans-Wohnm' 'Aixam' 'Piaggio' 'Morgan' 'Tazzari' 'Trucks-Lkw' 'RAM'
     'Ferrari' 'Iveco' 'DAF' 'Alpina' 'Polestar' 'Maybach' 'Brilliance'
     'FISKER' 'Lamborghini' 'Cadillac' 'Trailer-Anhänger' 'Isuzu' 'Corvette'
     'DFSK' 'Estrima']
    ========================================
    Categorical feature: 'fuel'
    No. of unique elemnts: 11
    Unique values:
    ['Diesel' 'Gasoline' 'Electric/Gasoline' '-/- (Fuel)' 'Electric' 'CNG'
     'LPG' 'Electric/Diesel' 'Others' 'Hydrogen' 'Ethanol']
    ========================================
    Categorical feature: 'gear'
    No. of unique elemnts: 3
    Unique values:
    ['Manual' 'Automatic' nan 'Semi-automatic']
    ========================================


Car `make`s (brands) in our data set possesses a **high cardinality**, meaning that there too many of unique values of this category. One-Hot Encoding becomes a big problem in this case since we will have a separate column for each unique `make` value indicating its presence or absence in a given record. This leads to a big problem called **the curse of dimensionality**: as the number of features increases, the amount of data required to be able to distinguish between these features and generalize learned model grows exponentially. We will come back to this later and use a different encoding algorithm for this feature.

#### Fuel Type

We group the fuel types of the cars into four major categories: 'Gasoline', 'Diesel', 'Electric' and 'Others'.


```python
replace_mapping = {'Electric': ['Electric/Gasoline', 'Electric/Diesel'],
                   'Others': ['-/- (Fuel)', 'CNG', 'LPG', 'Hydrogen', 'Ethanol']}

for value, to_replace in replace_mapping.items():
    cars['fuel'].replace(to_replace, value, inplace=True)
```

---

### Handling Missing Data

#### Overview

First step here is to get a better picture of the *frequency* of missing observations in our data set. This is important since it leads us to come to a 'decision' about how to treat missing values (i.e. what strategy to use for missing records). Moreover, in order to handle missing observations, also data preparation for training, it is necessary to split the features into numerical and categorical features.


```python
def countplot_missing_data(
    df: pd.DataFrame,
    ax: mpl.axes.Axes,
    **kwargs
) -> None:
    """
    Utility function to plot the counts of missing values in each
    column of the given dataframe `df` using bars.
    """
    missings = df.isna().sum(axis=0).rename('count')
    missings = missings.rename_axis(index='feature').reset_index()
    missings.sort_values('count', ascending=False, inplace=True)

    x = range(missings.shape[0])
    bars = ax.bar(x, missings['count'], **kwargs)

    ax.bar_label(bars, fmt='%d', label_type='edge', padding=3)
    ax.set(xticks=x, xticklabels=missings['feature'])
```


```python
all_features = {
    'numerical': ['mileage', 'hp', 'age'],
    'categorical': ['make', 'fuel', 'gear']
}

colors = ['teal', 'skyblue']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for (item, ax, color) in zip(all_features.items(), axes, colors):
    features_type, features_list = item
    countplot_missing_data(cars[features_list], ax, color=color, width=0.35)
    ax.set(
        title=f'{features_type.capitalize()} Features',
        ylabel='Missing Values, Count [#]'
    )
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_24_0.png)



The is really a high-quality data set; no missing values for most of the features available. The number of missing observations for features `hp` and `gear` are very small compared to the total number of data (less than 0.05% and 0.5%, respectively). Instead of eliminating (dropping) records for these samples, we try to impute the missing values using information from other records. For the numerical feature `hp` we use the *median* and for the categorical feature `gear` we use *most frequrnt* imputation strategy to replace missing values. For both cases, we compuate the values of interest for each car `make` and `model`, meaning that we first group the data based on `make` and `model`:


```python
agg_data = cars.groupby(['make', 'model']).aggregate(
    {'hp': np.nanmedian, 'gear': pd.Series.mode}
)

agg_data.sample(frac=1, random_state=912).head(n=20)
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
      <th>make</th>
      <th>model</th>
      <th>hp</th>
      <th>gear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Peugeot</td>
      <td>807</td>
      <td>136.0</td>
      <td>Manual</td>
    </tr>
    <tr>
      <td>Lexus</td>
      <td>GS 450h</td>
      <td>345.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Lada</td>
      <td>Niva</td>
      <td>83.0</td>
      <td>Manual</td>
    </tr>
    <tr>
      <td>Ford</td>
      <td>Flex</td>
      <td>314.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Mitsubishi</td>
      <td>Eclipse Cross</td>
      <td>163.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Suzuki</td>
      <td>Vitara</td>
      <td>120.0</td>
      <td>Manual</td>
    </tr>
    <tr>
      <td>smart</td>
      <td>forTwo</td>
      <td>71.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Citroen</td>
      <td>C-Zero</td>
      <td>57.5</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>BMW</td>
      <td>X5 M</td>
      <td>400.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Peugeot</td>
      <td>Rifter</td>
      <td>131.0</td>
      <td>Manual</td>
    </tr>
    <!-- <tr>
      <td>Volkswagen</td>
      <td>T5 Shuttle</td>
      <td>140.0</td>
      <td>Manual</td>
    </tr>
    <tr>
      <td>Ford</td>
      <td>F 150</td>
      <td>403.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Mercedes-Benz</td>
      <td>GLE 400</td>
      <td>333.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Tesla</td>
      <td>Model S</td>
      <td>428.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Mercedes-Benz</td>
      <td>250</td>
      <td>204.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>BMW</td>
      <td>230</td>
      <td>252.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Dodge</td>
      <td>Dart</td>
      <td>188.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Lancia</td>
      <td>Thema</td>
      <td>286.0</td>
      <td>Automatic</td>
    </tr>
    <tr>
      <td>Opel</td>
      <td>Combo</td>
      <td>95.0</td>
      <td>Manual</td>
    </tr>
    <tr>
      <td>BMW</td>
      <td>X6</td>
      <td>313.0</td>
      <td>Automatic</td>
    </tr> -->
  </tbody>
</table>
</div>



Function below can be used as imputation transformer. This function groups the data based on specific features, then aggregate the grouped data to extract values of interest according to our imputation strategy. Finally, it returns the index and corresponding filling value for each missing observation, so it can be used to update the original data frame and replace the missing values. In the next section, we'll see how this function comes in handy.


```python
def imputation_transform(
    df: pd.DataFrame,
    label: str,
    group_by: Union[str, List[str]],
    strategy: str
) -> pd.Series:
    """
    Imputation transformer for replacing missing values.

    Parameters
    ----------
    df:
        Main data set.
    label:
        Column for which the missing values are imputed. After groupby,
        the data is aggregated over this column.
    group_by:
        Column, or list of columns used to group data `df`.
    strategy : str, {'mean', 'median', 'most_frequent'}
        The imputation strategy.
        If 'mean' or 'median', then uses equivalent NumPy functions
        ignoring NaNs (`np.nanmean` and `np.nanmedian`, respectively).
        If 'most_frequent', it uses `pd.Series.mode` method.

    Returns
    -------
    Values that can be used to update `df` to replace its missing
    values for the column `label`.
    """
    strategy_maps = {
        'mean': np.nanmean,
        'median': np.nanmedian,
        'most_frequent': pd.Series.mode
    }

    func = strategy_maps[strategy]
    agg_data = df.groupby(group_by).aggregate({label: func})

    def imputer(row):
        """
        Imputation function to apply to each row of the `df` dataframe
        to replace missing values for the feature `agg_over` using the
        imputation strategy `strategy`.
        """
        try:
            x = agg_data.loc[tuple(row[group_by]), label]
            if isinstance(x, np.ndarray):
                fill_value = x[0] if x.size != 0 else np.nan
            else:
                fill_value = x
        except KeyError:
            fill_value = np.nan

        return fill_value

    nan_idx = df[label].isna()
    return df.loc[nan_idx].apply(lambda row: imputer(row), axis=1)
```

## Median Imputation of Missing `hp` Values


```python
hp_imp = imputation_transform(
    df=cars,
    label='hp',
    strategy='median',
    group_by=['make', 'model']
)

cars['hp'].update(hp_imp)
```

## Mode Imputation of Missing `gear` Values


```python
gear_imp = imputation_transform(
    df=cars,
    label='gear',
    strategy='most_frequent',
    group_by=['make', 'model']
)

cars['gear'].update(gear_imp)
```

## Revisiting Our Data Set


```python
all_features = {
    'numerical': ['mileage', 'hp', 'age'],
    'categorical': ['make', 'fuel', 'gear']
}

colors = ['teal', 'skyblue']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for (item, ax, color) in zip(all_features.items(), axes, colors):
    features_type, features_list = item
    countplot_missing_data(cars[features_list], ax, color=color, width=0.35)
    ax.set(
        title=f'{features_type.capitalize()} Features',
        ylabel='Missing Values, Count [#]'
    )
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_34_0.png)



So far so good. We successfully handelled the missing values for most of the records in the data set. But there are still some missing observations. At this point, we could simply eliminate (drop) records for these samples:


```python
cars.dropna(axis=0, subset=['hp', 'gear'], inplace=True)
cars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 40088 entries, 0 to 46375
    Data columns (total 8 columns):
     #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
     0   mileage  40088 non-null  int64
     1   make     40088 non-null  object
     2   model    39977 non-null  object
     3   fuel     40088 non-null  object
     4   gear     40088 non-null  object
     5   price    40088 non-null  int64
     6   hp       40088 non-null  float64
     7   age      40088 non-null  int64
    dtypes: float64(1), int64(3), object(4)
    memory usage: 2.8+ MB

---

## Exploratory Data Analysis

### Visualizing Categorical Data Using Treemaps

Treemap is a popular visualization technique used to visualize 'Part of a Whole' relationship. Treemaps are easy to follow and interpret. Treemaps are often used for sales data, as they capture relative sizes of data categories, allowing for quick perception of the items that are large contributors to each category. Apart from the sizes, categories can be colour-coded to show a separate dimension of data. Here, we use treemap visualization to explore the relative sizes (counts) of different categories and their average price in the data sat. The size of nested grids in our treemap visualization represents the counts of each category, and grid-cell colours indicate the mean price for that category.


```python
def plot_treemap(
    df: pd.DataFrame,
    size_key: str,
    color_key: str,
    ax: mpl.axes.Axes,
    cmap: mpl.colors.Colormap = mpl.cm.Blues,
    font_size: int = 8,
    title_prefix: Optional[str] = None,
    **kwargs
) -> None:
    """
    Utility function to plot treemaps for categorical features using squarify.

    Parameters
    ----------
    size_key:
        Column name whose numeric values specify sizes of rectangles.
    color_key:
        Column name whose numeric values specify colors of rectangles.
    ax:
        Matplotlib Axes instance.
    cmap:
        Matplotlib colormap.
    font_size:
        Set font size of the label text.
    title_prefix:
        Text to prepend to the figure title.
    **kwargs:
        Additional keyword arguments passed to matplotlib.Axes.bar by
        squarify (e.g. `edgecolor`, `linewidth` etc).
    """
    df_sorted = df.sort_values(by=size_key, ascending=False)
    sizes = df_sorted[size_key]

    norm = mpl.colors.Normalize(vmin=df[color_key].min(), vmax=df[color_key].max())

    colors = [cmap(norm(x)) for x in df_sorted[color_key]]

    labels = [
        f"{entry[0]}\nCount: {entry[1]}\nAvg. Pr.: {entry[2]:.0f}"
        for entry in df_sorted.values
    ]

    squarify.plot(sizes=sizes, color=colors, label=labels, ax=ax, **kwargs)

    ax.axis('off')
    for text_obj in ax.texts:
        text_obj.set_fontsize(font_size)

    fig = ax.get_figure()
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=ax)
    cbar.set_label(color_key)

    title = f"Size: {size_key}, Color: {color_key}"
    if title_prefix:
        title = title_prefix + '\n' + title
    ax.set_title(title)
```


```python
def get_category_meanprice(
    df_in: pd.DataFrame,
    categ_key: str
) -> pd.DataFrame:
    """
    Utility function to get mean price for a given categorical feature.

    Parameters
    ----------
    df_in:
        Input dataframe.
    categ_key:
        Categorical feature name (column name).

    Returns
    -------
    df_out:
        Output dataframe with column names `['count', 'mean_price']`.
    """
    df_out = pd.concat(
        objs=[
            df_in[categ_key].value_counts().rename('count'),
            df_in.groupby(categ_key)['price'].mean().rename('mean_price')
          ],
        axis=1
    )

    df_out.reset_index(drop=False, inplace=True)
    df_out.rename(columns={'index': categ_key}, inplace=True)

    return df_out
```

### Car Makes (Brands)


```python
print(
    f"Number of unique car-makes: {cars['make'].nunique()}",
    f"Unique car makes listed in the data set:",
    sep='\n'
)

cars['make'].unique()
```

    Number of unique car-makes: 72
    Unique car makes listed in the data set:

    array(['BMW', 'Volkswagen', 'SEAT', 'Renault', 'Peugeot', 'Toyota',
           'Opel', 'Mazda', 'Ford', 'Mercedes-Benz', 'Chevrolet', 'Audi',
           'Fiat', 'Kia', 'Dacia', 'MINI', 'Hyundai', 'Skoda', 'Citroen',
           'Infiniti', 'Suzuki', 'SsangYong', 'smart', 'Volvo', 'Jaguar',
           'Porsche', 'Nissan', 'Honda', 'Lada', 'Mitsubishi', 'Others',
           'Lexus', 'Cupra', 'Maserati', 'Bentley', 'Land', 'Alfa', 'Jeep',
           'Subaru', 'Dodge', 'Microcar', 'Baic', 'Tesla', 'Chrysler', '9ff',
           'McLaren', 'Aston', 'Rolls-Royce', 'Lancia', 'Abarth', 'DS',
           'Daihatsu', 'Ligier', 'Caravans-Wohnm', 'Aixam', 'Piaggio',
           'Alpine', 'Morgan', 'RAM', 'Ferrari', 'Iveco', 'Alpina',
           'Polestar', 'Maybach', 'Brilliance', 'FISKER', 'Lamborghini',
           'Cadillac', 'Isuzu', 'Corvette', 'DFSK', 'Estrima'], dtype=object)




```python
# Count up cars by `make` feature & average `price`s for each `make` type
makes = get_category_meanprice(cars, 'make')

# Select data w/ a minimum number of items
min_nitems = 200
indexer = makes['count'] >= min_nitems

treemap_kwargs = dict(edgecolor='gray', linewidth=1, font_size=10, alpha=0.8)

fig, ax = plt.subplots(figsize=(16, 9))
plot_treemap(
    makes.loc[indexer, :],
    'count',
    'mean_price',
    ax,
    cmap=mpl.cm.PuBu,
    title_prefix=f'Car Makes (# >= {min_nitems})',
    **treemap_kwargs
)
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_42_0.png)



### Fuel Types


```python
# Count up cars by `fuel` feature & average `price`s for each `fuel` type
fuels = get_category_meanprice(cars, 'fuel')

fig, ax = plt.subplots(figsize=(12, 7))
plot_treemap(
    fuels,
    'count',
    'mean_price',
    ax,
    cmap=mpl.cm.BuPu,
    title_prefix='Fuel Types',
    **treemap_kwargs
)
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_44_0.png)



### Gear Types


```python
# Count up cars by `gear` feature & average `price`s for each `gear` type
gears = get_category_meanprice(cars, 'gear')

fig, ax = plt.subplots(figsize=(12, 7))
plot_treemap(
    gears,
    'count',
    'mean_price',
    ax,
    cmap=mpl.cm.Reds,
    title_prefix='Gear Types',
    **treemap_kwargs
)
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_46_0.png)



## Visualizing Categorical Data Using Box Plots


```python
min_nitems = 200
s = cars['make'].value_counts() > min_nitems
indexer = cars['make'].isin(s[s].index)
order = cars[indexer].groupby('make')['price'].median().sort_values(ascending=False).index

fig = plt.figure(figsize=(12, 7), tight_layout=True)
gs = mpl.gridspec.GridSpec(2, 2)
boxplot_kwargs = dict(orient='h', showfliers=False, linewidth=1)

sns.boxplot(
    y=cars.loc[indexer, 'make'],
    x=cars.loc[indexer, 'price'] / 1e+3,
    ax=fig.add_subplot(gs[:, 0]),
    order=order,
    palette='pink',
    **boxplot_kwargs
)

for ifeature, feature in enumerate(['fuel', 'gear']):
    order = cars.groupby(feature)['price'].median().sort_values(ascending=False).index
    sns.boxplot(
        y=cars[feature],
        x=cars['price'] / 1e+3,
        ax=fig.add_subplot(gs[ifeature, 1]),
        order=order,
        **boxplot_kwargs
    )

for ax in fig.get_axes():
    ax.set_xlabel(r'price ($\times$ 1000)')
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_48_0.png)



## Scatter Plots: Visualizing the Relationship Between Numerical Data


```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.tight_layout(w_pad=2.5)

for (feature, ax) in zip(all_features['numerical'], axes):
    ax.scatter(cars[feature], cars['price'] / 1e+3, edgecolor='w', alpha=0.8)
    ax.set(xlabel=feature, ylabel=r'price ($\times$ 1000)')

axes[0].set(ylabel=r'price ($\times$ 1000)');
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_50_0.png)



Scatter plots above shows that the car sale prices have high correlation with `mileage` and `hp`. Moreover, these plots reveal another important characteristic of the data: **heteroscedasticity**. We can see that the predictive variable (price) monitored over different independent variables show *unequal* variability (particularly, over `hp` feature. Look at the cone-shape of the scatter plot). This means that a linear regression model is not a proper model to predict the sale price for this data set, sice the heteroscedasticity of the data will ruin the results.

## Distribution of the Numerical Features


```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.tight_layout(w_pad=2.5)

for (feature, ax) in zip(all_features['numerical'], axes):
    ax.hist(cars[feature], bins=25)
    ax.set(xlabel=feature, ylabel='Count [#]')
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_53_0.png)



## Checking Correlations Between Price and Numerical Features


```python
corr = cars.corr()
new_order = ['price'] + numerical_features
corr = corr.reindex(index=new_order, columns=new_order)

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_facecolor('none')
sns.heatmap(corr, ax=ax, mask=mask, annot=True, cmap='BrBG', vmin=-1);
```



<img
  src="/assets/img/2021-11-05-car-price-germany-eda/output_55_0.png"
  class="center_img"
  style="width: 50%;"
>




The highest correlation shows for `price` with horsepower `hp` and `age` (no big news). Looking at negative correlation between `price` with `age` also seems natural.

---
# Distribution of Target Variable


```python
prices = cars['price'] / 1e+3

bin_width = 5   # in €1000
nbins = int(round((prices.max() - prices.min()) / bin_width)) + 1
bins = np.linspace(prices.min(), prices.max(), nbins)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.hist(prices, bins=bins)
ax1.set(xlabel=r'Price ($\times$ 1000)', ylabel='Count [#]')
_ = probplot(prices, plot=ax2)
```



![png](/assets/img/2021-11-05-car-price-germany-eda/output_58_0.png)



The target distribution (of sale prices) is so skewed. Most of regression algorithms perform best when the target variable is normally distributed (or close to normal) and has a standard deviation of one.

---
**Save Processed Data to a New CSV File**


```python
cars.to_csv(
    '../data/autoscout24_germany_dataset_cleaned.csv',
    header=True,
    index=False
)
```
