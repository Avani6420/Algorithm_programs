import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# created s regressor
# sklearn provides biltin datasets
# we get the diabetes dataset

diabetes = datasets.load_diabetes()
# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# we can get single feature
diabetes_x = diabetes.data # [:, np.newaxis, 2] if single then add scing

# x= features & y= label
diabetes_x_train = diabetes_x[:-30] # get last 30 element
diabetes_x_test = diabetes_x[-30:] # get first 20 element

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# sing sklearn we can create the linear model
model = linear_model.LinearRegression()

# fit the data : draw the line
model.fit(diabetes_x_train, diabetes_y_train)

# we cane draw the line and put featres value
diabetes_y_predicted = model.predict(diabetes_x_test)

print("mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("weights: ", model.coef_)
print("intercept: ", model.intercept_)

# if one ele then plot
# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predicted)
# plt.show()

