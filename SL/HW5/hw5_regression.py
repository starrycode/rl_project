import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Load the CSV file into a DataFrame
data = pd.read_csv('SL/HW5/train.csv')

# For Q1

# Separate the input features and the target variable
# drop the column with the target variable
X = data.drop('critical_temp', axis=1)
y = data['critical_temp']  # select the column with the target variable

# Split the dataset into a training set and a test set
# For (a)
X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=0.2)

# For (b)
X_train, X_val, y_train, y_val = train_test_split(
    X_test, y_test, test_size=0.25)


# For Q2

L1 = [0, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
L2 = [0, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]

train_MSE = []
val_MSE = []

for l1 in L1:
    for l2 in L2:
        alpha = l1 + l2
        l1_ratio = 0 if l1 == l2 == 0 else l1 / (l1 + l2)
        regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
        regr.fit(X_train, y_train)
        pred_train = regr.predict(X_train)
        train_MSE.append((l1, l2, mean_squared_error(y_train, pred_train)))
        pred_val = regr.predict(X_val)
        val_MSE.append((l1, l2, mean_squared_error(y_val, pred_val)))

# For (b)
print("For part (b)")
print(train_MSE)

# For (c)
print("For part (c)")
print(val_MSE)

# For (d)
best_model_train = min(train_MSE, key=lambda x: x[2])
best_model_val = min(val_MSE, key=lambda x: x[2])

print("For part (d)")
print(best_model_train)
print(best_model_val)

# best_model_val[0] = l1
# best_model_val[1] = l2

# For (e)
alpha = best_model_val[0] + best_model_val[1]
l1_ratio = best_model_val[0] / (best_model_val[0] + best_model_val[1])
regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
regr.fit(X_remain, y_remain)
pred_test = regr.predict(X_test)
print("For part (e)")
print(mean_squared_error(y_test, pred_test))
