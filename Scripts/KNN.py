import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
## Hyper Parameter Tuning for KNN
## Only parameter to consider here in the number of neighbors
from sklearn.model_selection import GridSearchCV
    
n_neighbors = [1,2,3,4,5,6,7,8,9,10,11]
param_grid = {'n_neighbors': n_neighbors}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(estimator = knn,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose = 1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
#{'n_neighbors': 6}
#0.94

"""
# Initializing model with best hyper-parameters and fitting data
best_knn = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)   
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)

print("The test accuracy is: ")
print(accuracy_score(y_test, y_pred), end = '\n')

# Classification report
print("Classification report")
print(classification_report(y_test, y_pred))

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
plt.title('Confusion Matrix (KNN)')
plt.show()

# Saving model 
with open('Pickles/KNN.pickle', 'wb') as output:
    pickle.dump(best_knn, output)
