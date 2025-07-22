import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

regressors = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(random_state=0)),
    ("Lasso Regression", Lasso(random_state=0)),
    ("ElasticNet Regression", ElasticNet(random_state=0)),
    ("Support Vector", SVR()),
    ("K-Nearest Neighbors", KNeighborsRegressor()),
    ("Decision Tree", DecisionTreeRegressor(random_state=0)),
    ("Random Forest", RandomForestRegressor(random_state=0)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=0)),
    ("Multi-layer Perceptron", MLPRegressor( 
        hidden_layer_sizes=(128,64,32),
        learning_rate_init=0.0001,
        activation='relu', solver='adam', 
        max_iter=10000, random_state=0)),
]

results = {}

data = pd.read_csv('CO-adsorption.csv')
#data

X = data.iloc[:,2:]
X = X.to_numpy()
#print(X)

Y = data.iloc[:,1]
Y = Y.to_numpy()
#print(Y)

X = StandardScaler().fit_transform(X)
#X

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#for i in [X_train, X_test, y_train, y_test]:
#  i = np.array(i)

train_data = [X_train,y_train]
test_data = [X_test,y_test]

for name, regressor in regressors:
    regressor.fit(X_train, y_train)
    pred_y_train = regressor.predict(X_train)
    pred_y_test = regressor.predict(X_test)
    pred_Y = regressor.predict(X)
    mse = mean_squared_error(y_test, pred_y_test)
    score = regressor.score(X_test, y_test)
    se = abs(y_test - pred_y_test)
    results[f"{name}"] = {"MSE": mse, "error": se}
    print(f"[{name}]\t  MSE:{mse:.4f}\t  R2:{score:.4f}")

    fig, ax = plt.subplots()

    ax.scatter(y_train,pred_y_train,label='Train')
    ax.scatter(y_test,pred_y_test,color='r',label='Test')
    ax.plot([-2,0.2], [-2,0.2], "--k")
    ax.set_ylabel("Target predicted")
    ax.set_xlabel("True Target")
    ax.set_title(f"Regression performance of the model {name}")
    ax.text(
        -1.9,
        -0.3,
        r"$R^2$=%.4f, MAE=%.2f eV, RMSE=%.2f eV"
        % (r2_score(Y, pred_Y), mean_absolute_error(Y, pred_Y), math.sqrt(mean_squared_error(Y, pred_Y))),
    )
    ax.set_xlim([-2,0.2])
    ax.set_ylim([-2,0.2])

    ax.legend()
    plt.savefig(f"{name}", dpi=300)

#print(results)
