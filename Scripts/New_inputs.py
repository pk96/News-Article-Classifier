## TAKING NEW INPUTS AND FEEDING THROUGH ENSEMBLE LEARNING FUNCTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading pre-trained models 
svm_path = "D:/Course Materials/Machine Learning/Project/Pickles/SVM.pickle"
with open(svm_path, 'rb') as data:
    svm = pickle.load(data)

knn_path = "D:/Course Materials/Machine Learning/Project/Pickles/KNN.pickle"
with open(knn_path, 'rb') as data:
    knn = pickle.load(data)

nb_path = "D:/Course Materials/Machine Learning/Project/Pickles/NaiveBayes.pickle"
with open(nb_path, 'rb') as data:
    nb = pickle.load(data)

lg_path = "D:/Course Materials/Machine Learning/Project/Pickles/LogRegression.pickle"
with open(lg_path, 'rb') as data:
    lg = pickle.load(data)

Xtrain_path = "D:/Course Materials/Machine Learning/Project/Pickles/X_train_pre.pickle"
with open(Xtrain_path, 'rb') as data:
    X_train = pickle.load(data)

category_codes = {'business': 0,
                  'entertainment': 1,
                  'politics': 2,
                  'sport': 3,
                  'tech': 4 }


punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

#nltk.download('punkt')
#nltk.download('wordnet')
# Function for creating tfidf vector(features) from new input text
def feature_engineering(text):
    
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
    #print(training_features)
    
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Content'])
    df.loc[0] = text
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Content_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)    
    lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_5'] = lemmatized_text_list
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    
    #print (tabulate(df,headers = 'firstrow'))
    df = df['Content_Parsed_6']
    #print(df.head(10))
    #df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    #print (features)
    #print(df)
    return features


# Getting category from category_id
def get_category(category_id):
     for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category


def most_frequent(votes): 
    return max(set(votes), key = votes.count)

def ensemble_majority_vote(res_svm, res_knn, res_lg):

    votes = [res_svm, res_knn, res_lg]
    
    if (res_svm != res_knn) and (res_knn != res_lg) and (res_svm != res_lg):
        print("Each classifier has predicted a different category.")
    
    print("Aggregated prediction of 3 classifiers: Category of input article is ", most_frequent(votes))
    

def predict_from_text(text):
    
    # Predict using the input model
    features = feature_engineering(text)
    
    # SVM RESULTS
    prediction_svc = svm.predict(features)[0]
    prediction_svc_proba = svm.predict_proba(features)[0]
    category_svc = get_category(prediction_svc)
    
    if (prediction_svc_proba.max()*100) > 65.00:
        print("The predicted category using the SVM model is %s." %(category_svc) )
        print("The probability is: %a" %(prediction_svc_proba.max()*100), "%")
        print("\n")
    else:
        print('The probability for the SVM model is below 65%, therefore the predicted category is "Other"')
        category_svc = 'Other'
        print("\n")
        
    # KNN RESULTS
    prediction_knn = knn.predict(features)[0]
    prediction_knn_proba = knn.predict_proba(features)[0]
    category_knn = get_category(prediction_svc)
    
    if (prediction_knn_proba.max()*100) > 65.00:
        print("The predicted category using the KNN model is %s." %(category_knn) )
        print("The probability is: %a" %(prediction_knn_proba.max()*100), "%")
        print("\n")
    else:
        print('The probability for the K nearest neighbors model is below 65%, therefore the predicted category is "Other"')
        category_knn = 'Other'
        print("\n")
        
    # NAIVE BAYES RESULTS
    prediction_nb = nb.predict(features)[0]
    prediction_nb_proba = nb.predict_proba(features)[0]
    category_nb = get_category(prediction_nb)
    
    if (prediction_nb_proba.max()*100) > 65.00:
        print("The predicted category using the Naive Bayes model is %s." %(category_nb) )
        print("The probability is: %a" %(prediction_nb_proba.max()*100), "%")
        print("\n")
    else:
        print('The probability for the Naive Bayes is below 65%, therefore the predicted category is "Other"')
        category_nb = 'Other'
        print("\n")
        
    # LOGISTIC REGRESSION RESULTS
    prediction_lg = lg.predict(features)[0]
    prediction_lg_proba = lg.predict_proba(features)[0]
    category_lg = get_category(prediction_lg)
    
    if (prediction_lg_proba.max()*100) > 65.00:
        print("The predicted category using the Logistic Regression model is %s." %(category_lg) )
        print("The probability is: %a" %(prediction_lg_proba.max()*100), "%")
        print("\n")
    else:
        print('The probability for the Logistic Regression model is below 65%, therefore the predicted category is "Other"')
        category_lg = 'Other'
        print("\n")
    
    ensemble_majority_vote(category_svc, category_knn, category_lg)

    
# new input articles
# more sample inputs can be found in the New Inputs folder
# article here is from business category ('001.txt' in New Input Articles/business)
text = """ 
Air France-KLM has secured at least €9bn (£7.9bn; $9.7bn) in government aid, as the Franco-Dutch airline group struggles to stay afloat because of the coronavirus outbreak.

The French authorities said Air France would get €3bn in loans and another €4bn in state-guaranteed funds.

Meanwhile, the Dutch government said it was preparing between €2bn and €4bn in aid to KLM.

Major world airlines all but halted passenger traffic around the world.

But they still have to pay to park and maintain planes that have been grounded.

Earlier this year, Air France-KLM estimated the outbreak would cost the group between €150m and €200m in February-April.

Company Chief Executive Ben Smith warned on Friday that the government aid was "not a blank cheque" and would require tough action on costs and performance, Reuters news agency reports.

"This financing will give us the opportunity to rebuild. Faced with the upheaval the world is going through, we are going to have to rethink our model immediately," he added.

The group is the result of a merger between Air France and KLM in 2004.

With a fleet of 550 aircraft, it covers 312 destinations in 116 countries around the world. In 2018, Air France-KLM's passenger traffic exceeded 100 million.
""" 


# unrelated category ('weather 001.txt' in New Input Articles/Other Categories)
""" text = 

Severe thunderstorms will hit the South and Midwest Tuesday into Tuesday night with threats of widespread damaging winds, large hail and a few tornadoes. Some severe weather could persist into Wednesday in the Southeast.

Tuesday morning, an isolated severe thunderstorm dropped large hail on the Dallas-Fort Worth Metroplex. Hail up to the size of ping-pong balls was reported at the National Weather Service office in Fort Worth.
The more widespread threat of severe weather will occur late Tuesday afternoon and Tuesday evening.

In the southern portion of the severe threat area shown below, large hail and an isolated tornado threat could be the initial concerns from southeastern Kansas into central and northeastern Oklahoma late Tuesday afternoon. This includes Oklahoma City and Tulsa, Oklahoma.

The storms will then congeal into a squall line with widespread damaging winds and an isolated tornado threat as they move southeastward Tuesday evening toward southeastern Oklahoma, north-central and northeastern Texas, Arkansas and Louisiana. It's possible this severe weather threat could meet the criteria for a damaging wind event called a derecho, but that will depend on how widespread it becomes.

In the Midwest threat area, strong to severe storms could have damaging winds, large hail and a few tornadoes. This includes parts of Missouri, southwestern Wisconsin, Illinois and eastern Iowa.

The low-pressure system will track eastward on Wednesday. Rain will continue in parts of the Midwest, while showers and thunderstorms move across parts of the South.

There could be a lingering threat of severe weather in coastal Texas and southern Louisiana in the early-morning hours. Strong thunderstorm winds would be the main concern.

Strong to severe storms could increase Wednesday afternoon in the Southeast, particularly in parts of southeastern Mississippi, Alabama and Georgia. Localized damaging wind gusts and large hail would be the main threats from any storms that do intensify, but a couple of tornadoes cannot be ruled out.

Atlanta; Columbus, Georgia; Birmingham and Montgomery, Alabama; and New Orleans are some of the cities that could have strong to severe storms on Wednesday.

Showers and thunderstorms may linger into Thursday in parts of the Southeast, mainly toward the coastal Carolinas and in Florida. A few severe storms are possible in these areas Thursday.
"""

predict_from_text(text)