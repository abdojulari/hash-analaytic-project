import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# reading the dataset 

dataset =  'companyx.xlsx'

# adding a column to the  excel sheets named attrition, denoting on existing employee sheet with 1 and on 
# Employees who have left with 0
existing_emp_df = pd.read_excel(dataset, 'Existing employees')
existing_emp_df.insert(10, "attrition", 0)
left_emp_df = pd.read_excel(dataset, 'Employees who have left')
left_emp_df.insert(10, "attrition", 1)
#matching the two sheets into one 
appended_df = existing_emp_df.append(left_emp_df)
appended_df = appended_df.drop('Emp ID', axis = 1)
appended_df.head()

#Transform non-numeric columns into numerical columns

for column in appended_df.columns:
        if appended_df[column].dtype == np.number:
            continue
        appended_df[column] = LabelEncoder().fit_transform(appended_df[column])
        
X = appended_df.iloc[:, :-1].values
Y = appended_df.iloc[:, 9 ].values

# data preprocessing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#fitting the logistic regression to training set 
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100, max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, Y_train)


Y_pred2 = clf_gini.predict(X_test)

print(accuracy_score(Y_test, Y_pred2))

importances = pd.DataFrame({'Feature':appended_df.iloc[:, :-1].columns,'Importance':np.round(clf_gini.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('Importance',ascending=False).set_index('Feature')
importances

#Visualize the importance
importances.plot.bar()