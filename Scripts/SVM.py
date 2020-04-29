import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC

## LOADING TRAINING AND TESTING DATA
df_path = "D:/Course Materials/Machine Learning/Project/Pickles/df.pickle"
with open(df_path, 'rb') as data:
    df = pickle.load(data)

X_train_path = "D:/Course Materials/Machine Learning/Project/Pickles/training_features.pickle"
with open(X_train_path, 'rb') as data:
    X_train = pickle.load(data)

y_train_path = "D:/Course Materials/Machine Learning/Project/Pickles/training_labels.pickle"
with open(y_train_path, 'rb') as data:
    y_train = pickle.load(data)

X_test_path = "D:/Course Materials/Machine Learning/Project/Pickles/test_features.pickle"
with open(X_test_path, 'rb') as data:
    X_test = pickle.load(data)

y_test_path = "D:/Course Materials/Machine Learning/Project/Pickles/test_labels.pickle"
with open(y_test_path, 'rb') as data:
    y_test = pickle.load(data)
    
"""
## HYPER-PARAMETER TUNING
from sklearn.model_selection import GridSearchCV
# Creating a grid of possible hyper-parameter values 
C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
  ]

svm = SVC(random_state = 0)

# Initialising grid search model and fitting training data 
grid_search = GridSearchCV(estimator = svm,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose = 1)
grid_search.fit(X_train, y_train)


print(grid_search.best_params_)
#{'C': 0.1, 'kernel': 'linear', 'probability': True}

print(grid_search.best_score_)
#0.9498666666666666
"""

# Initializing new SVM model with best combination of hyper parameters and fitting to data
best_svm = SVC(C = 0.1, kernel = 'linear', probability = True, random_state = 120)
best_svm.fit(X_train, y_train)

# Predicting Test Set results
y_pred = best_svm.predict(X_test)

# Training accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("The test accuracy is: ")
print(accuracy_score(y_test, y_pred), end = '\n')

# Classification report
print("Classification report")
print(classification_report(y_test, y_pred))

#Visualizing the confusion matrix
import seaborn as sns 

temp_df = df[['Category', 'Category_code']].drop_duplicates().sort_values('Category_code')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(cm, 
            annot = True,
            xticklabels = temp_df['Category'].values, 
            yticklabels = temp_df['Category'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion Matrix (SVM)')
plt.show()

# Saving SVM as pickled object
with open('Pickles/SVM.pickle', 'wb') as output:
    pickle.dump(best_svm, output)

