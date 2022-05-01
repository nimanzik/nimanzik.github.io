---
layout: post
title: "Price Prediction of Used Cars in Germany (2011-2021) — II. Prediction Using Quantile Loss"
date: 2021-11-17
description: ""
img_url: /assets/img/2021-11-17-car-price-germany-pred/output_38_0.png
tags: ["Prediction", "Uncertainty", "LightGBM"]
# github_url:
# colab_url:
# kaggle_url:
---

[AutoScout24](https://www.autoscout24.com/), based in Munich, is one of the largest European online marketplace for buying and selling new and used cars. With its products and services, the car portal is aimed at private consumers, dealers, manufacturers and advertisers. In this post, we use a high-quality car data set from Germany that was automatically scraped from the AutoScout24 and is available on the [Kaggle](https://www.kaggle.com/ander289386/cars-germany) website. This interesting data set comprises more than 45,000 records of cars for sale in Germany registered between 2011 and 2021.

**Outline**
- [Quantile Loss Function](#quantile-regression-loss-function-aka-pinball-loss-function)
- [Preparatory Feature Selection](#preparatory-feature-selection)
- [Leave-One-Out Encoding](#leave-one-out-encoding)
- [Box-Cox Transformation of Targets](#box-cox-transformation-of-targets)
- [Creating Model Pipeline Step by Step](#creating-model-pipeline-step-by-step)
- [Hyperparameter Tuning with scikit-optimize](#hyperparameter-tuning-with-scikit-optimize)
- [Final Estimators: Computing Predictive Intervals](#final-estimators-computing-predictive-intervals)

---
Let’s first install and import packages we’ll use and set up the working environment:


```python
!conda install -q -y -c conda-forge lightgbm
!conda install -q -y -c conda-forge scikit-optimize
!conda install -q -y -c conda-forge category_encoders
```


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from category_encoders import LeaveOneOutEncoder
from lightgbm import LGBMRegressor
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss, median_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn import set_config
from skopt import gp_minimize, space
from skopt.utils import use_named_args
```


```python
# Settings
%config InlineBackend.figure_format = 'retina'
sns.set_theme(font_scale=1.5, rc={'lines.linewidth': 2.5})
```

---
# Quantile Loss Function (aka Pinball Loss Function)

Instead of only reporting a single best estimate, we want to predict an estimate and uncertainty as to our estimate. This is especially challenging when that uncertainty varies with the independent variables. A quantile regression is one method for estimating uncertainty which can be used with our model (gradient boosted decision trees).

A quantile is the value below which a fraction of observations in a group falls.

By minimizing the quantile loss, $$\alpha.100\,\%$$ of the observations should be **smaller than** predictions ($$y^{(i)} \lt \hat{y}^{(i)}$$), and $$(1-\alpha).100\, \%$$ should be **bigger** ($$y^{(i)} \ge \hat{y}^{(i)}$$).

In other words, the model should **over-predict** $$\alpha.100\,\%$$ of the times, and **under-predict** $$(1-\alpha).100\, \%$$ of the times. Note that, here the error is defined as $$y - \hat{y}$$. As an example, a prediction for quantile $$\alpha=0.9$$ should over-predict $$90\%$$ of the times.

The quantile loss function is defined as below:

$$
\begin{equation}
  \mathcal{L}_{\alpha}(y^{(i)}, \hat y^{(i)}) = \begin{cases}
  \begin{aligned}
    \alpha\,&(y^{(i)} - \hat y^{(i)}) && \text{if} \, y^{(i)} \ge \hat y^{(i)}\quad \text{(i.e., under-predictions)}, \\
    (\alpha - 1)\,&(y^{(i)} - \hat y^{(i)}) && \text{if} \, y^{(i)} \lt \hat y^{(i)}\quad \text{(i.e., over-predictions)},
  \end{aligned}
  \end{cases}
\end{equation}
$$

For a set of predictions, the total cost is its average. Therefore, the cost function will be:

$$
\begin{equation}
  \mathcal{J} = \frac{1}{m} \Sigma_{i=1}^m \mathcal{L}_{\alpha}(y^{(i)}, \hat y^{(i)}),
\end{equation}
$$

We can see that the quantile loss differs depending on the evaluated quantile value, $\alpha$. The closer the $\alpha$ is to 0, the more this loss function penalizes over-predictions (negative errors). As the $\alpha$ value gets closer to 1, the more this loss function penalizes under-predictions (positive errors). Figure below illustrates this.


```python
def quantile_loss(y_true, y_pred, alpha):
    """
    Quantile loss function.
    
    Parameters
    ----------
    y_true : ndarray
        True observations (i.e., labels).
    y_pred : ndarray
        Model predictions (same shape as `y_true`).
    alpha : float
        Evaluated quantile values. Should be between 0 and 1.
    """
    errors = y_true - y_pred
    pos = errors >= 0.0
    neg = np.logical_not(pos)
    loss_vals = alpha * (pos * errors) + (alpha - 1) * (neg * errors)
    return loss_vals
```

For the sake of illustration, we consider one single true value and investigate how the quantile loss function behaves when model predictions are larger than the true value (negative erros) or smaller that that (positive errors) for different quantile values $$\alpha$$.


```python
y_true = 5.0
y_pred = np.linspace(0.0, 10, 51)
errors = y_true - y_pred

# Quantile values
alphas = [0.1, 0.5, 0.9]
colors = ['sandybrown', 'chocolate', 'saddlebrown']

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
fig.tight_layout(w_pad=1)
axline_kw = dict(ls='--', lw=1.5, c='gray')

for alpha, color, ax in zip(alphas, colors, axes):
    ax.axvline(0.0, **axline_kw)
    ax.axhline(0.0, **axline_kw)

    loss_vals = quantile_loss(y_true, y_pred, alpha)
    ax.plot(errors, loss_vals, color)
    ax.set(title=fr'$\alpha={alpha}$', xlabel=r'Error, $y - \hat{y}$')

axes[0].set_ylabel(r'Quantile loss, $\mathcal{L}$');
```


    
![png](/assets/img/2021-11-17-car-price-germany-pred/output_7_0.png)
    


---
# Data Loading

Let's load our data set that we have analysed and cleaned previously.


```python
cars = pd.read_csv('../data/autoscout24_germany_dataset_cleaned.csv')
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
      <th>3206</th>
      <td>26566</td>
      <td>Fiat</td>
      <td>500</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>8240</td>
      <td>69.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2076</th>
      <td>92000</td>
      <td>Mitsubishi</td>
      <td>Space Star</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>5399</td>
      <td>71.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8537</th>
      <td>101709</td>
      <td>smart</td>
      <td>forTwo</td>
      <td>Gasoline</td>
      <td>Automatic</td>
      <td>8490</td>
      <td>102.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>502</th>
      <td>20302</td>
      <td>Ford</td>
      <td>Galaxy</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>28711</td>
      <td>190.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37020</th>
      <td>79500</td>
      <td>Ford</td>
      <td>Edge</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>28740</td>
      <td>209.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9958</th>
      <td>38335</td>
      <td>Hyundai</td>
      <td>i20</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>12470</td>
      <td>84.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3856</th>
      <td>74860</td>
      <td>Volkswagen</td>
      <td>up!</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>5500</td>
      <td>75.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10262</th>
      <td>106000</td>
      <td>Ford</td>
      <td>Galaxy</td>
      <td>Diesel</td>
      <td>Automatic</td>
      <td>14000</td>
      <td>163.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>23216</th>
      <td>29450</td>
      <td>Ford</td>
      <td>Focus</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>16450</td>
      <td>140.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22424</th>
      <td>50000</td>
      <td>Mercedes-Benz</td>
      <td>Sprinter</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>17990</td>
      <td>114.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



---
# Preparatory Feature Selection

We do not utilize `'model'` column since it is a subset of `'make'` (car brand) feature and each car brand has its own set of model categories, that is not the same for other car brands. It is better to use this column when predicting the sale price of a specific car make, e.g. BMW, is aimed.


```python
cars.drop(columns=['model'], inplace=True)
cars.head()
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
      <th>fuel</th>
      <th>gear</th>
      <th>price</th>
      <th>hp</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>235000</td>
      <td>BMW</td>
      <td>Diesel</td>
      <td>Manual</td>
      <td>6800</td>
      <td>116.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>92800</td>
      <td>Volkswagen</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>6877</td>
      <td>122.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149300</td>
      <td>SEAT</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>6900</td>
      <td>160.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96200</td>
      <td>Renault</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>6950</td>
      <td>110.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>156000</td>
      <td>Peugeot</td>
      <td>Gasoline</td>
      <td>Manual</td>
      <td>6950</td>
      <td>156.0</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



---
# Data Splitting

We use $$80\%$$ of samples in the training set, while the remaining samples ($$20\%$$) will be available in the testing set.


```python
data = cars.drop(columns='price', inplace=False)
target = cars['price']

X_train, X_test, y_train, y_test = \
    train_test_split(data, target, test_size=0.2)
```

---
# Leave-One-Out Encoding

Our data consists a mixture of numerical and categorical features. More specifically, `('mileage', 'hp', 'age')` are the numerical features, and `('make', 'fuel', 'gear')` are the categorical features. In orther to be fed into a machine-learning model, categorical features should be represented numerically.


```python
for feature in ('make', 'fuel', 'gear'):
    print(
        f'Feature: {feature} - '
        f'Number of categories: {data[feature].nunique()}')
```

    Feature: make - Number of categories: 72
    Feature: fuel - Number of categories: 4
    Feature: gear - Number of categories: 3


To encode the features `'fuel'` and `'gear'`, we use the most commonly used method one-hot encoding (sometimes called "dummy encoding"). This algorithm creates a binary variable for each unique value of the categorical variables (for each sample, the new feature is 1 if the sample’s category matches the new feature, otherwise the value is 0). This algorithm adds a new feature for each unique category in `'fuel'` and `'gear'`.

Since the category `'make'` in our data set comes with **high cardinality** (i.e., it has too many unique categories), using one-hot encoding for this feature will create too many additional feature variables. This could lead to poor model performance - as data dimentionality increases, the model suffers from the **curse of dimensionality**).

To keep the number of dimensions under control, we use **leave-one-out** encoding method. It essentially calculates the mean of the *target* variables over all samples that have the same value for the categorical-feature variable in question. The encoding algorithm is slightly different between training and testing sets:
* For training set, it leaves the sample under consideration out, hence the name *Leave One Out*. This is done to avoid **target leakage**. Moreover, it adds random Gaussian noise with zero mean and $$\sigma$$ standard deviation into training data in order to decrease overfitting. $$\sigma$$ is a hyperparameter that should be tuned.
* For testing set, it does not leave the current sample out - the statistics about target variable for each value of each categorical variable calculated for the training data set is used. It also does not need the randomness factor.

Leave-one-out technique is implemented in the [Category Encoders](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html) library as a scikit-learn-style transformer:


```python
loo_encoder = LeaveOneOutEncoder(sigma=0.05)
encoded = loo_encoder.fit_transform(X_train['make'], y_train)
encoded.head(n=10)
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
      <th>make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38370</th>
      <td>9573.924156</td>
    </tr>
    <tr>
      <th>13498</th>
      <td>9029.417690</td>
    </tr>
    <tr>
      <th>27888</th>
      <td>8792.132242</td>
    </tr>
    <tr>
      <th>12999</th>
      <td>25944.844420</td>
    </tr>
    <tr>
      <th>7243</th>
      <td>9101.103968</td>
    </tr>
    <tr>
      <th>2653</th>
      <td>12284.006678</td>
    </tr>
    <tr>
      <th>7050</th>
      <td>13583.914503</td>
    </tr>
    <tr>
      <th>31847</th>
      <td>14364.409615</td>
    </tr>
    <tr>
      <th>1741</th>
      <td>8766.192993</td>
    </tr>
    <tr>
      <th>36229</th>
      <td>8188.430674</td>
    </tr>
  </tbody>
</table>
</div>



---
# Box-Cox Transformation of Targets

In the previous notebook, where we did the exploratory data analysis, we saw that the distribution of the target variables (prices) in our data set is very skewed to the left. We also saw that the target variables show *unequal* variability when monitored over different indipendent variables (see cone-shape scatter plots there). This is a sign of heteroscedasticity.

Although, we will use a tree-based learning algorithm that does not make any assumptions on normality of the targets, it would be useful to make targets more Gaussian-like for modeling issues related to heteroscedasticity (non-constant variance).

Here, we use Box-Cox transformation to map targets to as close to a Gaussian distribution as possible in order to stabilise the variance and minimize the skewness. Note that Box-Cox transform can only be applied to strictly positive data (such as income, price etc).


```python
pt = PowerTransformer(method='box-cox')
y_orig = target.values[:, np.newaxis]
y_norm = pt.fit_transform(y_orig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle('Targets Distribution', y=1)

hist_kw = dict(bins=30, facecolor='teal')

_ = ax1.hist(y_orig, **hist_kw)
ax1.set_title('Original')

_ = ax2.hist(y_norm, **hist_kw)
ax2.set_title('Transformed (Box-Cox)');
```


    
![png](/assets/img/2021-11-17-car-price-germany-pred/output_19_0.png)
    


---
# Creating Model Pipeline Step by Step

## Data-Preprocessing and Transformation

- Robust scaling for numerical features
- One-hot encoding for `'fuel'` and `'gear'`
- Leave-one-out encoding for `'make'`


```python
def create_preprocessor():
    """
    Data preprocessing and feature transformation pipeline

    Returns
    -------
    out : `sklearn.compose.ColumnTransformer` object
    """
    # Numerical features
    numerical_features = ['mileage', 'hp', 'age']
    numerical_transformer = RobustScaler()

    # Categorical features
    categorical_features_a = ['fuel', 'gear']
    categorical_transformer_a = OneHotEncoder(handle_unknown='ignore')

    categorical_features_b = ['make']
    categorical_transformer_b = Pipeline(
        steps=[
            ('loo_encoder', LeaveOneOutEncoder()),
            ('scaler', RobustScaler())])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('numer', numerical_transformer, numerical_features),
            ('categ_a', categorical_transformer_a, categorical_features_a),
            ('categ_b', categorical_transformer_b, categorical_features_b)])

    return preprocessor
```

## LightGBM Regression Estimator

[LightGBM](https://lightgbm.readthedocs.io/en/latest/) is a gradient boosting tree-based framework developed by Microsoft. Fast training, accuracy, and efficient memory utilisation are its main features. LightGBM uses a leaf-wise tree growth algorithm that tends to converge faster compared to depth-wise growth algorithms. It can be a go-to model for many tabular-data machine-learning problems.

LightGBM has several parameters which control how the decision trees are constructed, including:
* `num_leaves`: maximum number of leaves in one tree. Large values increase accuracy on the training set, but also increase the chance of getting hurt by overfitting.
* `max_depth`: limit the maximum depth for tree model. It controls the maximum number of splits in the decision trees.
* `n_estimators`: number of boosting iterations (trees to build). The more trees built the more accurate the model can be at the cost of longer training time and higher chance of overfitting.
* `learning_rate`: the shrinkage rate used for boosting.

We will use Bayesian optimization to find a good combination of these hyperparameters.

To create our regression estimator:
* We use `LightGBMRegressor` model with quantile loss function to estimate the predictive intervals on our dataset.
* As discussed before, we regress on transformed targets (Box-Cox transformation). To do so, we need to wrap LightGBM's regressor by scikit-learn's `TransformedTargetRegressor`.


```python
def create_quantile_regressor(alpha=0.5):
    """
    LightGBM regressor w/ quantile loss applied on transformed targets.
    In the regression problem, targets (prices) are transformed by using
    Box-Cox power transform method, `PowerTransform(method='box-cox')`.
    
    Parameters
    ----------
    alpha: float
        The coefficient used in quantile-based loss, should be between 0
        and 1. Default: 0.5 (median).

    Returns
    -------
    Quantile regressor to apply on transformed targets.
    """
    ttq_regressor = TransformedTargetRegressor(
        regressor=LGBMRegressor(objective='quantile', alpha=alpha),
        transformer=PowerTransformer(method='box-cox'))
    
    return ttq_regressor
```

## Full Training Pipeline


```python
def create_full_pipeline(alpha=0.5):
    """
    Full prediction pipeline.

    Parameters
    ----------
    alpha: float
        The coefficient used in quantile-based loss, should be between 0
        and 1. Default: 0.5 (median).

    Returns
    -------
    Pipeline of data transforms with the final estimator returned as
    `sklearn.pipeline.Pipeline` object.
    """
    preprocessor = create_preprocessor()
    ttq_regressor = create_quantile_regressor(alpha=alpha)
    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('ttq_regressor', ttq_regressor)])

    return model
```


```python
set_config(display='diagram')
create_full_pipeline()
```




<style>#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 {color: black;background-color: white;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 pre{padding: 0;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-toggleable {background-color: white;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-item {z-index: 1;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-parallel-item:only-child::after {width: 0;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-8251e01e-d5f0-4902-b53f-50d2e6409ed7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numer&#x27;, RobustScaler(),
                                                  [&#x27;mileage&#x27;, &#x27;hp&#x27;, &#x27;age&#x27;]),
                                                 (&#x27;categ_a&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;fuel&#x27;, &#x27;gear&#x27;]),
                                                 (&#x27;categ_b&#x27;,
                                                  Pipeline(steps=[(&#x27;loo_encoder&#x27;,
                                                                   LeaveOneOutEncoder()),
                                                                  (&#x27;scaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;make&#x27;])])),
                (&#x27;ttq_regressor&#x27;,
                 TransformedTargetRegressor(regressor=LGBMRegressor(alpha=0.5,
                                                                    objective=&#x27;quantile&#x27;),
                                            transformer=PowerTransformer(method=&#x27;box-cox&#x27;)))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="083acb57-7744-478d-960e-f14dae938ead" type="checkbox" ><label for="083acb57-7744-478d-960e-f14dae938ead" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numer&#x27;, RobustScaler(),
                                                  [&#x27;mileage&#x27;, &#x27;hp&#x27;, &#x27;age&#x27;]),
                                                 (&#x27;categ_a&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;fuel&#x27;, &#x27;gear&#x27;]),
                                                 (&#x27;categ_b&#x27;,
                                                  Pipeline(steps=[(&#x27;loo_encoder&#x27;,
                                                                   LeaveOneOutEncoder()),
                                                                  (&#x27;scaler&#x27;,
                                                                   RobustScaler())]),
                                                  [&#x27;make&#x27;])])),
                (&#x27;ttq_regressor&#x27;,
                 TransformedTargetRegressor(regressor=LGBMRegressor(alpha=0.5,
                                                                    objective=&#x27;quantile&#x27;),
                                            transformer=PowerTransformer(method=&#x27;box-cox&#x27;)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f2d55177-699a-4059-9cac-79f6a67519e6" type="checkbox" ><label for="f2d55177-699a-4059-9cac-79f6a67519e6" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numer&#x27;, RobustScaler(),
                                 [&#x27;mileage&#x27;, &#x27;hp&#x27;, &#x27;age&#x27;]),
                                (&#x27;categ_a&#x27;,
                                 OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                 [&#x27;fuel&#x27;, &#x27;gear&#x27;]),
                                (&#x27;categ_b&#x27;,
                                 Pipeline(steps=[(&#x27;loo_encoder&#x27;,
                                                  LeaveOneOutEncoder()),
                                                 (&#x27;scaler&#x27;, RobustScaler())]),
                                 [&#x27;make&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="07410962-f391-40eb-93a1-28b49317fb91" type="checkbox" ><label for="07410962-f391-40eb-93a1-28b49317fb91" class="sk-toggleable__label sk-toggleable__label-arrow">numer</label><div class="sk-toggleable__content"><pre>[&#x27;mileage&#x27;, &#x27;hp&#x27;, &#x27;age&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3b2fe679-5042-4faa-8dfb-63b89e96a6c5" type="checkbox" ><label for="3b2fe679-5042-4faa-8dfb-63b89e96a6c5" class="sk-toggleable__label sk-toggleable__label-arrow">RobustScaler</label><div class="sk-toggleable__content"><pre>RobustScaler()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="82946402-986b-4cab-b021-b26034028f6b" type="checkbox" ><label for="82946402-986b-4cab-b021-b26034028f6b" class="sk-toggleable__label sk-toggleable__label-arrow">categ_a</label><div class="sk-toggleable__content"><pre>[&#x27;fuel&#x27;, &#x27;gear&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="49fdb73b-ef36-4370-9cb1-94b2ad40aecf" type="checkbox" ><label for="49fdb73b-ef36-4370-9cb1-94b2ad40aecf" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="26c26953-5451-49fe-8406-02531fd55bb7" type="checkbox" ><label for="26c26953-5451-49fe-8406-02531fd55bb7" class="sk-toggleable__label sk-toggleable__label-arrow">categ_b</label><div class="sk-toggleable__content"><pre>[&#x27;make&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="eba3c326-e4e1-4908-bee9-f614390f5e2b" type="checkbox" ><label for="eba3c326-e4e1-4908-bee9-f614390f5e2b" class="sk-toggleable__label sk-toggleable__label-arrow">LeaveOneOutEncoder</label><div class="sk-toggleable__content"><pre>LeaveOneOutEncoder()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a4cc51cc-3c5e-47a8-bf26-ade7b4a47f81" type="checkbox" ><label for="a4cc51cc-3c5e-47a8-bf26-ade7b4a47f81" class="sk-toggleable__label sk-toggleable__label-arrow">RobustScaler</label><div class="sk-toggleable__content"><pre>RobustScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c2e3e660-2cc4-4b0f-8a60-7cbc2d3876c9" type="checkbox" ><label for="c2e3e660-2cc4-4b0f-8a60-7cbc2d3876c9" class="sk-toggleable__label sk-toggleable__label-arrow">ttq_regressor: TransformedTargetRegressor</label><div class="sk-toggleable__content"><pre>TransformedTargetRegressor(regressor=LGBMRegressor(alpha=0.5,
                                                   objective=&#x27;quantile&#x27;),
                           transformer=PowerTransformer(method=&#x27;box-cox&#x27;))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="843d3c77-3f64-48e4-8758-a6766fac6f65" type="checkbox" ><label for="843d3c77-3f64-48e4-8758-a6766fac6f65" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMRegressor</label><div class="sk-toggleable__content"><pre>LGBMRegressor(alpha=0.5, objective=&#x27;quantile&#x27;)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="74479ffc-b53e-4fcc-8674-a06dae472f24" type="checkbox" ><label for="74479ffc-b53e-4fcc-8674-a06dae472f24" class="sk-toggleable__label sk-toggleable__label-arrow">PowerTransformer</label><div class="sk-toggleable__content"><pre>PowerTransformer(method=&#x27;box-cox&#x27;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



---
# Hyperparameter Tuning with **scikit-optimize**
## Objective


```python
def hyperparameter_tuner(model, X, y, search_space, cvs_params, gp_params):
    """
    Parameters
    ----------
    model : estimator object implementing `fit()` method
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict.
    search_space : list, shape (n_dims,)
        List of hyperparameter search space dimensions.
    cvs_params : dict
        Parameters for Scikit-Learn's `cross_val_score()` method.
    gp_params : dict
        Parameters for Scikit-Optimize's `gp_minimize()` method.

    Returns
    -------
    The Gaussian Processes optimization result with the required
    information, returned as SciPy's `OptimizeResult` object.
    """
    @use_named_args(search_space)
    def objective_func(**params):
        """
        Function to *minimize*. Decorated by the `use_named_args()`, it
        takes named (keyword) arguments and returns the objective value.
        """
        model.set_params(**params)
        scores = cross_val_score(model, X, y, **cvs_params)
        return -1.0 * np.mean(scores)
        
    # We are now ready for sequential model-based optimization
    optim_result = gp_minimize(objective_func, search_space, **gp_params)
    return optim_result
```

## Optimization


```python
%%time

# Parameters of a Scikit-Learn estimator in a pipeline are accessed
# using the `<estimator>__<parameter>` syntax. Valid parameter keys
# can be listed with model's `get_params()` method.

# Hyperparameters search space
search_space = [
    space.Real(
        0.01, 0.6, prior='log-uniform',
        name='preprocessor__categ_b__loo_encoder__sigma'),
    space.Integer(10, 50, name='ttq_regressor__regressor__num_leaves'),
    space.Integer(2, 30, name='ttq_regressor__regressor__max_depth'),
    space.Real(
        0.01, 0.5, prior='log-uniform',
        name='ttq_regressor__regressor__learning_rate')]

# Parameters that'll be passed to sklearn's `cross_val_score()`
cvs_params = {'cv': 3, 'n_jobs': 4}

# Parameters that'll be passed to skopt's `gp_minimize()`
gp_params = {'n_calls': 50,
             'n_initial_points': 10,   
             'acq_func': 'EI',
             'n_jobs': 4,
             'verbose': False}

# Lower, middle and higher quantiles to evaluate: lower/upper bounds + median
quantiles = {'q_lower': 0.025, 'q_middle': 0.5, 'q_upper': 0.975}

# Cache Gaussian Process optimization results
cache_gp_results = dict()

for q_level, alpha in quantiles.items():
    print(f'\nHyperparameter optimization for quantile regressor with '
          f'alpha={alpha:.3f} ..... ', end='')

    # Create model pipeline
    model = create_full_pipeline(alpha=alpha)

    # Cross-validation scoring strategy 
    scorer = make_scorer(
        mean_pinball_loss, alpha=alpha, greater_is_better=False)
    cvs_params['scoring'] = scorer

    # Find optimal hyperparameters
    gp_result = hyperparameter_tuner(
        model, X_train, y_train, search_space, cvs_params, gp_params)

    # Cache current results
    cache_gp_results[q_level] = gp_result
    print('Done')
    
    # Report GP optimization result
    print('Best parameters:')
    for i_param, param_value in enumerate(gp_result.x):
        param_name = search_space[i_param].name.split('__')[-1]
        print(''.rjust(4) + f'{param_name}: {param_value:.6f}')
    print('=' * 40)
```

    
    Hyperparameter optimization for quantile regressor with alpha=0.025 ..... Done
    Best parameters:
        sigma: 0.010000
        num_leaves: 10.000000
        max_depth: 2.000000
        learning_rate: 0.320519
    ========================================
    
    Hyperparameter optimization for quantile regressor with alpha=0.500 ..... Done
    Best parameters:
        sigma: 0.010000
        num_leaves: 50.000000
        max_depth: 27.000000
        learning_rate: 0.197304
    ========================================
    
    Hyperparameter optimization for quantile regressor with alpha=0.975 ..... Done
    Best parameters:
        sigma: 0.010000
        num_leaves: 10.000000
        max_depth: 30.000000
        learning_rate: 0.157132
    ========================================
    CPU times: user 3min 1s, sys: 3.07 s, total: 3min 4s
    Wall time: 2min 15s


## Plotting Convergence Traces


```python
n_models = len(quantiles)
fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
fig.tight_layout(w_pad=1)

colors = ['sandybrown', 'chocolate', 'saddlebrown']

for i, (q_level, gp_result) in enumerate(cache_gp_results.items()):
    call_idx = np.arange(1, gp_result.func_vals.size + 1)
    min_vals = [gp_result.func_vals[:i].min() for i in call_idx]

    ax = axes[i]
    ax.plot(call_idx, min_vals, '-o', color=colors[i])
    
    alpha = quantiles[q_level]
    ax.set(xlabel=r'Number of calls, $n$', title=fr'$\alpha$={alpha}')

axes[0].set_ylabel(r'$f_{min}(x)$ after $n$ calls')
plt.suptitle('GP Convergence Traces', y=1.1);
```


    
![png](/assets/img/2021-11-17-car-price-germany-pred/output_32_0.png)
    


---
# Final Estimators: Computing Predictive Intervals


```python
trained_models = dict()

for q_level, alpha in quantiles.items():
    # Create model for this quantile
    model = create_full_pipeline(alpha=alpha)
    
    # Retrieve best (optimized) parameters for this model
    gp_res = cache_gp_results[q_level]
    n_params = len(search_space)
    best_params = dict(
        (search_space[i].name, gp_res.x[i]) for i in range(n_params))
    
    # Set the parameters of this model
    model.set_params(**best_params)
    
    model.fit(X_train, y_train)
    trained_models[q_level] = model
```


```python
def coverage_percentage(
        y: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray) -> float:
    """
    Function to calculate the coverage percentage of the predictive
    interval, i.e. the percentage of observations that fall between the
    lower and upper bounds of the predictions.
    
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    y_lower : ndarray of shape (n_samples,)
        Lower predictive intervals.
    y_upper : ndarray of shape (n_samples,)
        Upper predictive intervals.
    """
    return np.mean(np.logical_and(y > y_lower, y < y_upper)) * 100.0
```


```python
# Compute predictive intervals

y_pred_lower = trained_models['q_lower'].predict(X_test)
y_pred_upper = trained_models['q_upper'].predict(X_test)

ctop = coverage_percentage(y_test, y_pred_lower, np.inf)
cbot = coverage_percentage(y_test, -np.inf, y_pred_upper)
cint = coverage_percentage(y_test, y_pred_lower, y_pred_upper)

print(f'Coverage of top 95%: {ctop:.1f}%',
      f'Coverage of bottom 95%: {cbot:.1f}%',
      f'Coverage of 95% predictive interval: {cint:.1f}%', sep='\n')
```

    Coverage of top 95%: 97.4%
    Coverage of bottom 95%: 97.0%
    Coverage of 95% predictive interval: 94.4%



```python
y_pred_middle = trained_models['q_middle'].predict(X_test)

print(
    f'Median Absolute Error: ',
    f'\tLower-quantile model: {int(median_absolute_error(y_test, y_pred_lower))}',
    f'\tMiddle-quantile model: {int(median_absolute_error(y_test, y_pred_middle))}',
    f'\tUpper-quantile model: {int(median_absolute_error(y_test, y_pred_upper))}',
    sep='\n')
```

    Median Absolute Error: 
    	Lower-quantile model: 2708
    	Middle-quantile model: 967
    	Upper-quantile model: 3655



```python
step = 25
idx = np.argsort(y_pred_middle)[::step]
dummy_x = np.arange(idx.size)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    dummy_x, y_test.values[idx], 'o', ms=3, alpha=0.8,
    label='Testing data')
ax.plot(
    dummy_x, y_pred_middle[idx], '-', color='chocolate', alpha=0.8,
    label='Predicted median')
ax.fill_between(
    dummy_x, y_pred_lower[idx], y_pred_upper[idx], color='gray', alpha=0.5,
    label='95% confidence interval')

ax.set(xlabel=f'Sample index (every {step}-th entry)', ylabel='Price')

ax.legend()
ax.set_yscale('log')
```


    
![png](/assets/img/2021-11-17-car-price-germany-pred/output_38_0.png)
    

