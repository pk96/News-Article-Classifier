import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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


nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

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
plt.title('Confusion Matrix (Naive Bayes)')
plt.show()

with open('Pickles/NaiveBayes.pickle', 'wb') as output:
    pickle.dump(nb, output)