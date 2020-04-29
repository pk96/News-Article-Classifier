import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## IMPORTING TABULAR DATASET
# Change path variable to read     
path = "D:\\Course Materials\Machine Learning\\Project\\Articles_corpus.csv"
df = pd.read_csv(path, sep = ';')

""" df.head(10)
  File_Name                                            Content  Category
0   001.txt  Ad sales boost Time Warner profit\r\n\r\nQuart...  business
1   002.txt  Dollar gains on Greenspan speech\r\n\r\nThe do...  business
2   003.txt  Yukos unit buyer faces loan claim\r\n\r\nThe o...  business
3   004.txt  High fuel prices hit BA's profits\r\n\r\nBriti...  business
4   005.txt  Pernod takeover talk lifts Domecq\r\n\r\nShare...  business
5   006.txt  Japan narrowly escapes recession\r\n\r\nJapan'...  business
6   007.txt  Jobs growth still slow in the US\r\n\r\nThe US...  business
7   008.txt  India calls for fair trade rules\r\n\r\nIndia,...  business
8   009.txt  Ethiopia's crop production up 24%\r\n\r\nEthio...  business
9   010.txt  Court rejects $280bn tobacco case\r\n\r\nA US ...  business
"""

## STEP 1: PREPROCESSING: PART 1 - TEXT CLEANING
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Removing special characters \r, \n and unwanted spaces

df['Content_parsed_1'] = df['Content'].str.replace('\r', ' ')
df['Content_parsed_1'] = df['Content_parsed_1'].str.replace('\n', ' ')
df['Content_parsed_1'] = df['Content_parsed_1'].str.replace('    ', ' ')
df['Content_parsed_1'] = df['Content_parsed_1'].str.replace('"', '') # removing quotes

df['Content_parsed_2'] = df['Content_parsed_1'].str.lower()

# 2. Removing punctuation signs and possesive pronoun terminations 

punctuation_signs = list("?:!.,;")
df['Content_parsed_3'] = df['Content_parsed_2']
for sign in punctuation_signs:
    df['Content_parsed_3'] = df['Content_parsed_3'].str.replace(sign, '')
    
df['Content_parsed_4'] = df['Content_parsed_3'].str.replace("'s", "")

# 3. Lemmatization 

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

n = len(df)
lemmatized_text_list = []

for row in range(n):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = df.loc[row]['Content_parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)

df['Content_parsed_5'] = lemmatized_text_list

# 4. Removing stop words

nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
"""
stop_words[0: 10]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
"""

df['Content_parsed_6'] = df['Content_parsed_5']

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_parsed_6'] = df['Content_parsed_6'].str.replace(regex_stopword, '')

# Done with text cleaning, removing intermediate columns
columns = ["File_Name", "Category", "Content", "Content_parsed_6"]
df = df[columns]

df = df.rename(columns={'Content_parsed_6': 'Content_parsed'})
"""
df.loc[2]['Content_parsed']
"yukos unit buyer face loan claim  owners  embattle russian oil giant yukos   ask  buyer   former production unit  pay back  $900m (â£479m) loan state-owned rosneft buy  yugansk unit  $93bn   sale force  russia  part settle  $275bn tax claim  yukos yukos' owner menatep group say   ask rosneft  repay  loan  yugansk  secure   assets rosneft already face  similar $540m repayment demand  foreign bank legal experts say rosneft purchase  yugansk would include  obligations  pledge assets   rosneft      pay real money   creditors  avoid seizure  yugansk assets say moscow-based us lawyer jamie firestone    connect   case menatep group manage director tim osborne tell  reuters news agency   default   fight    rule  law exist   international arbitration clauses   credit rosneft officials  unavailable  comment   company  say  intend  take action  menatep  recover    tax claim  debts owe  yugansk yukos  file  bankruptcy protection   us court   attempt  prevent  force sale   main production arm  sale go ahead  december  yugansk  sell   little-known shell company   turn  buy  rosneft yukos claim  downfall  punishment   political ambition   founder mikhail khodorkovsky   vow  sue  participant   sale"
"""

## STEP 1: PREPROCESSING: PART 2 - LABEL ENCODING
# Categories are currently represented as text, encoding them into numerical categories 

category_codes = {'business': 0,
                  'entertainment': 1,
                  'politics': 2,
                  'sport': 3,
                  'tech': 4
                  }

df['Category_code'] = df['Category']
df = df.replace({'Category_code':category_codes})
# Randomly shuffling the dataset before splitting
data = df


data = data.reindex(np.random.permutation(data.index))
#print(copy)

## STEP 1: PREPROCESSING: PART 3 - SPLITTING TRAINING AND TESTING DATA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['Content_parsed'],
                                                    data['Category_code'],
                                                    test_size = 0.20, random_state = 0)


## STEP 1: PREPROCESSING: PART 4 -  FEATURE ENGINEERING: CONVERTING TEXT TO TF-IDF VECTOR REPRESENTATION
from sklearn.feature_extraction.text import TfidfVectorizer

# Initializing parameters 
ngram_range = (1,1)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding = 'utf-8',
                        ngram_range = ngram_range,
                        stop_words = None,
                        lowercase = False,
                        max_df = max_df,
                        min_df = min_df,
                        max_features = max_features,
                        norm = 'l2',
                        smooth_idf = True,
                        sublinear_tf = True)

training_features = tfidf.fit_transform(X_train).toarray()
training_labels = y_train

test_features = tfidf.transform(X_test).toarray()
test_labels = y_test


# Saving training and testing data as pickles for easier access
# Currently saved to 'Pickles' folder in working directory 
import pickle

with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)
# TF-IDF 
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)

with open('Pickles/training_features.pickle', 'wb') as output:
    pickle.dump(training_features, output)
    
with open('Pickles/training_labels.pickle', 'wb') as output:
    pickle.dump(training_labels, output)
    
with open('Pickles/test_features.pickle', 'wb') as output:
    pickle.dump(test_features, output)
    
with open('Pickles/test_labels.pickle', 'wb') as output:
    pickle.dump(test_labels, output)

