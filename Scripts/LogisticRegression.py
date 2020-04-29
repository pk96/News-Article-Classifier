import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
# Hyper Parameter Tuning using GridSearch:
    
C = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
multi_class = ['multinomial']
solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
class_weight = ['balanced']
penalty = ['l2']

param_grid = {'C': C,
              'multi_class': multi_class,
              'solver': solver,
              'class_weight': class_weight,
              'penalty': penalty
              }

clf = LogisticRegression()

grid_search = GridSearchCV(estimator = clf,
                           param_grid = param_grid,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose = 1)

grid_search.fit(X_train, y_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
# Output: {'C': 0.9, 'class_weight': 'balanced', 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'newton-cg'}
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)
# Output: 0.9561797752808989
# grid_search.best_estimator_
LogisticRegression(C=0.9, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='multinomial', n_jobs=None,
                   penalty='l2', random_state=None, solver='newton-cg',
                   tol=0.0001, verbose=0, warm_start=False)
"""

# Fitting Regression model with best hyperparameters found through GridSearch
best_lg = LogisticRegression(C=0.9, class_weight='balanced', dual=False,
                             fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                             max_iter=100, multi_class='multinomial', n_jobs=None,
                             penalty='l2', random_state=None, solver='newton-cg',
                             tol=0.0001, verbose=0, warm_start=False)

best_lg.fit(X_train, y_train)
y_pred = best_lg.predict(X_test)

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
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

with open('Pickles/LogRegression.pickle', 'wb') as output:
    pickle.dump(best_lg, output)