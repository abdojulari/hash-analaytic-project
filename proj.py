# random forest 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# reading the dataset 

dataset =  'case-study.xlsx'

# adding a column to the  excel sheets named attrition, denoting on existing employee sheet with 1 and on 
# Employees who have left with 0
df_existing = pd.read_excel(dataset, 'Existing employees')
df_existing.insert(10, "attrition", 0)
df_left = pd.read_excel(dataset, 'Employees who have left')
df_left.insert(10, "attrition", 1)
#matching the two sheets into one 
new_df = df_existing.append(df_left)
new_df = new_df.drop('Emp ID', axis = 1)
new_df.head()

#Get the correlation of the columns
print(new_df.corr())

#Visualize the correlation
import seaborn as sns
plt.figure(figsize=(14,14))  #14in by 14in
sns.heatmap(new_df.corr(), annot=True, fmt='.0%')

#Transform non-numeric columns into numerical columns

for column in new_df.columns:
        if new_df[column].dtype == np.number:
            continue
        new_df[column] = LabelEncoder().fit_transform(new_df[column])

X = new_df.iloc[:, :-1].values
Y = new_df.iloc[:, 9 ].values

# data preprocessing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#fitting the random forest to training set 
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

#Get the accuracy on the training data
forest.score(X_train, Y_train)

#Show the confusion matrix and accuracy for  the model on the test data
#Classification accuracy is the ratio of correct predictions to total predictions made.
#from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, forest.predict(X_test))
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
print()     # Print a new line

# Return the feature importances (the higher, the more important the feature).
importances = pd.DataFrame({'feature':new_df.iloc[:, :-1].columns,'importance':np.round(forest.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances

#Visualize the importance
importances.plot.bar()