import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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


    
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# transformer -  list of tuples
ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [7,8])],
                       remainder='passthrough')
# we are dealing with index 8,9  i.e. state. We encoded state and pass it to array with float data type
X = new_df.iloc[:, :-1].values
Y = new_df.iloc[:, 9 ].values
X = np.array(ct.fit_transform(X), dtype=np.float)
print(X)


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
importance = classifier.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()