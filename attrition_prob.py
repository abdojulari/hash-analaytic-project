

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



dataset =  'companyx.xlsx'

# creating a column (attrition_status)to each sheet of the excel file, denote the existing 
# employee sheet with 0 and on Employees who have left with 1

existing_empdf = pd.read_excel(dataset, 'Existing employees')
existing_empdf.insert(10, "attrition_status", 0)

df_emp_left = pd.read_excel(dataset, 'Employees who have left')
df_emp_left.insert(10, "attrition_status", 1)


#matching the two sheets into one 
appended_df = existing_empdf.append(df_emp_left)

# drop a column considered unnecessary 
appended_df = appended_df.drop('Emp ID', axis = 1)


# classify the attrition status and count the unique values 
print(appended_df['attrition_status'].value_counts())

print(appended_df.describe())

# printing out the column with type object 
for column in appended_df.columns:
    if appended_df[column].dtype == object:
        print(str(column) + ' : ' + str(appended_df[column].unique()))
        print(appended_df[column].value_counts())
        print("_________________________________________________________________")   
        
# find the correlation 
        
print(appended_df.corr())

# Transform categorical columns (text values) into numerical columns

for column in appended_df.columns:
        if appended_df[column].dtype == np.number:
            continue
        appended_df[column] = LabelEncoder().fit_transform(appended_df[column])

X = appended_df.iloc[:, :-1].values
Y = appended_df.iloc[:, 9 ].values

# Splitting the dataset into traing and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# fitting random forest classification to training set 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

# confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(accuracy_score(Y_test, Y_pred))


from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(Y_test, Y_pred, target_names=target_names))

feature = pd.DataFrame({'feature':appended_df.iloc[:, :-1].columns,'importance':np.round(classifier.feature_importances_,3)}) 
feature = feature.sort_values('importance',ascending=False).set_index('feature')
feature

feature.plot.bar()

