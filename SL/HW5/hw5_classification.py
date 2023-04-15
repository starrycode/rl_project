import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the CSV file into a DataFrame
data = pd.read_csv('SL/HW5/Data_for_UCI_named.csv')

# For Q3
# For (a)
# drop the columns with the target variable
X = data.drop(columns=['p1', 'stab', 'stabf'])

# For (b)
target_var = {'unstable': 0, 'stable': 1}
# Changing the target variabel to a number
y = data['stabf'].map(target_var)

# For (c)
# Split the dataset into a training set and a test set
X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=0.2)

# For (d)
X_train, X_val, y_train, y_val = train_test_split(
    X_test, y_test, test_size=0.25)

# For Q4
# For (a)
L2_reg = LogisticRegression(penalty='l2')
L2_reg.fit(X_train, y_train)

no_reg = LogisticRegression(penalty='none')
no_reg.fit(X_train, y_train)

# For (b)
y_pred_L2_val = L2_reg.predict(X_val)
emp_risk_L2 = ('l2', 1 - accuracy_score(y_val, y_pred_L2_val))
confusion_L2 = confusion_matrix(y_val, y_pred_L2_val)

y_pred_no_reg_val = no_reg.predict(X_val)
emp_risk_no_reg = ('none', 1 - accuracy_score(y_val, y_pred_no_reg_val))
confusion_no_reg = confusion_matrix(y_val, y_pred_no_reg_val)

print("For part (b)")
print(emp_risk_L2)
print(confusion_L2)
print(emp_risk_no_reg)
print(confusion_no_reg)

# For (c)
better = min([emp_risk_L2, emp_risk_no_reg], key=lambda x: x[1])
final_model = LogisticRegression(penalty=better[0])
final_model.fit(X_remain, y_remain)

y_pred_final_val = no_reg.predict(X_test)
emp_risk_final = 1 - accuracy_score(y_test, y_pred_final_val)
confusion_final = confusion_matrix(y_test, y_pred_final_val)

print("For part (c)")
print(emp_risk_final)
print(confusion_final)
