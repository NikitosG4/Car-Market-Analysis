#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicles Price Modeling

This script:
    1) Loads and cleans a vehicle listing dataset
    2) Explores the data with plots
    3) Builds several regression models to predict price
    4) Evaluates models with R^2, MAE, and RMSE 
    5) Shows feature importance visualization where applicable 
    
Code and Resources Used
    Python Version: 3.11.3
    Packages: pandas, matplotlib, seaborn, numpy, sklearn, xgboost
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import data file

df = pd.read_csv('vehicles.csv')


# Dropping unnecessary data
df = df.drop(['id', 'url', 'region_url', 'image_url', 'description', 'VIN', 'county', 'region', 'posting_date', 'long', 'lat', 'size'], axis=1)

# Dropping rows with missing data
df = df.dropna(subset=['price', 'year', 'odometer', 'manufacturer'])

# Removing rows with suspicious prices
df = df[(df['price'] > 1000) & (df['price'] < 150000)]

# Removing listings with very high mileage
df = df[df['odometer'] < 300000]

# Remove unrealistic years (too old or from the future)
df = df[(df['year'] >= 1990) & (df['year'] <= 2025)]

# New column called age
df['age'] = 2025 - df['year']

# Replacing 'unknown' and 'other' values with
cols_to_fill = ['condition', 'cylinders', 'fuel', 'title_status', 'transmission',
                'drive', 'type', 'paint_color']

for col in cols_to_fill:
    df[col].fillna('unknown', inplace=True) 


# Extracting the number of cylinders to a separate column
def extract_cylinders(val):
    if val in ['unknown', 'other']:
        return np.nan
    else:
        return int(val.split()[0])

df['cylinders_num'] = df['cylinders'].apply(extract_cylinders)


# Replace mising values of number of cylinders with median value
df['cylinders_num'].fillna(df['cylinders_num'].median(), inplace = True)

df.head()


# Show how much data is missing in percents
for col in ['condition', 'cylinders', 'fuel', 'title_status', 'transmission',
            'drive', 'type', 'paint_color']:
    percent = (df[col] == 'unknown').mean() * 100
    print(f"{col}: {percent:.2f}% unknown")
    

# Histogram of distribution of car prices
plt.hist(df['price'], bins=100)
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Distribution of Car Prices')
plt.show()

# Scatterplot of price vs year of production relationship
plt.scatter(df['year'], df['price'], alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Price vs. Year')
plt.show()

# Boxplot of prices based on fuel type
sns.boxplot(x='fuel', y='price', data = df)
plt.xticks(rotation = 45)
plt.title("Price by Fuel Type")
plt.show()

# SHow odometer histogram
df.odometer.hist()
plt.title('Odometer histogram')
plt.xlabel('Odometer reading')
plt.ylabel('Number of cars')
plt.show()

#Correlatuon matrix
sns.heatmap(df.corr(numeric_only=True),annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

df['manufacturer'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Manufacturers")
plt.ylabel('Listing Counts')
plt.show()


# Model Building

# Choosing relevant columns

df_model = df[['price', 'age', 'manufacturer', 'odometer', 'condition', 'cylinders_num', 'fuel', 
            'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state']]

# Getting dummy data

df_dum = pd.get_dummies(df_model)

# Train test split

from sklearn.model_selection import train_test_split

X = df_dum.drop('price', axis = 1)
y = df_dum['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print("R2:", model.score(X_test,y_test))

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting Linear Regression

plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, n_jobs=-1, random_state = 42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R2:", rf.score(X_test, y_test))
print('MAE:', mean_absolute_error(y_test, y_pred_rf))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('Final dataset shape: ', df.shape)

# feature importance plot

importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10,6))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# XGBoost


from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators = 100, learning_rate = 0.2, max_depth = 6,
                   random_state = 42, n_jobs=-1)

xgb.fit(X_train, y_train)


y_pred_xgb = xgb.predict(X_test)


print("XGBoost R2:", xgb.score(X_test, y_test))
print('MAE:', mean_absolute_error(y_test, y_pred_xgb))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-15:]

plt.figure(figsize = (10,6))
plt.title('Top 15 Feature Importances (XGBoost)')
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center')
plt.yticks(range(len(indices_xgb)), [X.columns[i] for i in indices_xgb])
plt.xlabel('Relative Importance')
plt.show()


# KNN Regressor

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Building a pipeline
knn_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(
        n_neighbors=5, 
        weights='distance', 
        metric='minkowski', 
        p=2, 
        n_jobs=-1))])

knn_regressor.fit(X_train, y_train)

# Making predictions on the test data
y_pred_knn = knn_regressor.predict(X_test)

# Evaluating the model
print('MSE:', mean_squared_error(y_test, y_pred_knn))
print('KNN Regressor R2:', r2_score(y_test, y_pred_knn))

# Visualizing Results
# Parity Plot - Actual vs Predicted 
plt.scatter(y_test, y_pred_knn , alpha = 0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('KNN: Actual vs Predicted')
lims = [min(y_test.min(), y_pred_knn.min()), max(y_test.max(), y_pred_knn.max())]
plt.plot(lims, lims, 'r--')
plt.show()




# Ridge/Lasso Regression

# Ridge Regression

from sklearn.linear_model import Ridge

ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha = 1.0, random_state=42))])


ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print('Ridge R2:', r2_score(y_test, y_pred_ridge))
print('Ridge MAE:', mean_absolute_error(y_test, y_pred_ridge))
print('Ridge RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# Lasso Regression
from sklearn.linear_model import Lasso

lasso = Pipeline ([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha = 0.001, 
                    max_iter = 100000, 
                    tol=1e-3, 
                    random_state = 42))
    ])


lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print('Lasso R2:', r2_score(y_test, y_pred_lasso))
print('Lasso MAE:', mean_absolute_error(y_test, y_pred_lasso))
print('Lasso RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

lasso_model = lasso.named_steps['model']
coefs = lasso_model.coef_

feat_coefs = sorted(zip(X.columns, coefs), key=lambda x: abs(x[1]), reverse=True)

print('Top 15 coef Lasso:')
for name, w in feat_coefs[:15]:
    print(f'{name:40s} {w: .6f}')
    
zeros = (coefs == 0).sum()
print(f'Lasso zeroed out {zeros} features out of {len(coefs)}.')





# Neural Network (MLPRegressor)

from sklearn.neural_network import MLPRegressor

mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(
        hidden_layer_sizes = (128,64),
        activation = 'relu',
        solver = 'adam',
        alpha = 1e-4,
        learning_rate_init = 1e-3,
        max_iter = 500,
        early_stopping = True,
        validation_fraction = 0.1,
        n_iter_no_change = 20,
        random_state = 42,
        verbose = False
        ))
    ])

mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)


mlp_r2 = r2_score(y_test, y_pred_mlp)

mlp_mae = mean_absolute_error(y_test, y_pred_mlp)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))

print('MLP R2:', mlp_r2)
print('MLP MAE:', mlp_mae)
print('MLP RMSE:', mlp_rmse)

plt.scatter(y_test, y_pred_mlp, alpha = 0.5)
plt.xlabel('Actial Price')
plt.ylabel('Predicted Price')
plt.title('MLP: Actual vs Predicted')
lims = [min(y_test.min(), y_pred_mlp.min()), max(y_test.max(), y_pred_mlp.max())]
plt.plot(lims, lims, 'r--')
plt.show()













