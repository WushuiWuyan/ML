import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('ORR.csv')

X = data.iloc[:,9:]
X = X.to_numpy()
#print(X)

Y = data.iloc[:,2]
Y = Y.to_numpy()
#print(Y)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#for i in [X_train, X_test, y_train, y_test]:
#  i = np.array(i)

train_data = [X_train,y_train]
test_data = [X_test,y_test]

#print("train：",X_train.shape)
#print("test：",X_test.shape)

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr.score(X_test, y_test)

Y_pred = gbr.predict(X)
#print(Y_pred)

fig, ax = plt.subplots()

ax.scatter(Y,Y_pred)
ax.plot([-0.8,2.2], [-0.8,2.2], "--k")
ax.set_ylabel("Target predicted")
ax.set_xlabel("True Target") 
ax.set_title("Regression performance of the GBR model")
ax.set_xlim([-0.8,2.2])
ax.set_ylim([-0.8,2.2])
ax.text(
    -0.6,
    1.9,
    r"$R^2$=%.3f, MAE=%.4f eV, RMSE=%.4f eV"
    % (r2_score(Y, Y_pred), mean_absolute_error(Y, Y_pred), math.sqrt(mean_squared_error(Y, Y_pred))),
)
plt.show()

Y_pred = gbr.predict(X)
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

fig, ax = plt.subplots()

ax.scatter(y_train,y_train_pred)
ax.scatter(y_test,y_test_pred,color='r')
ax.plot([-0.8,2.2], [-0.8,2.2], "--k")
ax.set_ylabel("Target predicted")
ax.set_xlabel("True Target")
ax.set_title("Regression performance of the GBR model")
ax.text(
    -0.6,
    1.9,
    r"$R^2$=%.3f, MAE=%.4f eV, RMSE=%.4f eV"
    % (r2_score(Y, Y_pred), mean_absolute_error(Y, Y_pred), math.sqrt(mean_squared_error(Y, Y_pred))),
)
ax.set_xlim([-0.8,2.2])
ax.set_ylim([-0.8,2.2])

plt.savefig('Regression_performance_of_the_GBR_model', dpi=300)

