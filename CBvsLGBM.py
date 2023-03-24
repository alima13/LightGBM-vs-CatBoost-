# set up
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso  
from sklearn.linear_model import BayesianRidge  
from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVR  
from lightgbm import LGBMRegressor
from sklearn import metrics

# function to evaluate and return values of r-squared, mean squared error, and plot of model accuracy
def my_regression_results(model):
    # Step 4 - assess model quality on test data

    # Step 5 - make predictions
    y_pred = model.predict(X_test)
    %matplotlib inline
    import matplotlib.pyplot as plt
    # The plot gives an idea of the model accuracy
    plt.plot(y_test,y_pred,'k.')
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values');
    y_pred2 = model.predict(X)

    res = pd.DataFrame(y_pred2)
    res.columns = ["prediction"]
    res.to_csv("prediction_results.csv")

    # Step 6: Assess accuracy on test-data.
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('Mean squared error on test data: {:0.7f}'.format(mse))
    print('Root mean squared error on test data: {:0.7f}'.format(rmse))
    print('R^2:', metrics.r2_score(y_test, y_pred))

# read data
df = pd.read_csv('table.csv', sep=",")
target_column = ['taget'] 
predictors = list(set(list(df.columns))-set(target_column))

print(df[predictors])
df.describe()
X = df[predictors].values
y = df[target_column].values
#Split data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
df.head()

import catboost as cb
train_dataset = cb.Pool(X_train, y_train) 
test_dataset = cb.Pool(X_test, y_test)
model = cb.CatBoostRegressor(loss_function='RMSE')
scores = cross_val_score(model,X=X_train,y=y_train,cv=KFold(n_splits=5))
grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)
my_regression_results(model)

estimator = LGBMRegressor(random_state = 42)
from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'num_leaves' : [10, 30, 50, 100, 200],
    'max_depth': [None, 5, 10, 20, 50],
    'n_estimators': [150, 200, 400, 600],
    'learning_rate': [0.05, 0.1, 0.25, 0.5]
    }
model = RandomizedSearchCV(estimator, parameters, random_state=42, scoring = 'r2', n_iter = 50)
model.fit(X_train, y_train)
scores = cross_val_score(model,X=X_train,y=y_train,cv=KFold(n_splits=5))
print ('LGBM',scores)
my_regression_results(model)

import xgboost as xgb
train_dataset = cb.Pool(X_train, y_train) 
test_dataset = cb.Pool(X_test, y_test)
model = cb.CatBoostRegressor(loss_function='RMSE')
scores = cross_val_score(model,X=X_train,y=y_train,cv=KFold(n_splits=5))
grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)
my_regression_results(model)

params = {
    "n_estimators": [50, 100, 135],
    "max_features": [0.6, 0.8, 1.0],
    "min_samples_split": [2, 8, 17],
    "min_samples_leaf": [1, 13, 18],
    "bootstrap": [True, False]
}

params = {
    "n_estimators": randint(10, 150),
    "max_features": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "bootstrap": [True, False]
}
# setup the random search 
random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=params,
    random_state=123,
    n_iter=25,
    cv=5,
    verbose=1,
    n_jobs=1,
    return_train_score=True)

# optimization objective
def cv_score(hyp_parameters):
    hyp_parameters = hyp_parameters[0]
    rf_model = RandomForestRegressor(random_state=0,
                                    n_estimators=int(hyp_parameters[0]),
                                    max_features=hyp_parameters[1],
                                    min_samples_split=int(hyp_parameters[2]),
                                    min_samples_leaf=int(hyp_parameters[3]),
                                    bootstrap=bool(hyp_parameters[4]))   
    scores = cross_val_score(rf_model,
                             X=X_train,
                             y=y_train,
                             cv=KFold(n_splits=5))
    return np.array(scores.mean())  # return average of 5-fold scores

# 3D Graph of Rastrigin with dimension n = 2

